# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Lightweight Perfetto trace collector for sglang.

Generates Chrome Trace Event Format JSON that can be loaded in
Perfetto UI (https://ui.perfetto.dev/) or chrome://tracing.

No external dependencies required (no OpenTelemetry).
"""

from __future__ import annotations

import glob as globmod
import gzip
import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class _Tid:
    SCHEDULER = 0
    BATCH_FORWARD = 1
    COUNTERS = 2
    REQ_BASE = 100


class _Cat:
    SCHEDULER = "scheduler"
    BATCH = "batch"
    REQUEST = "request"
    COUNTER = "counter"


def _ts_us() -> float:
    return time.perf_counter() * 1e6


# Global collector instance per process (set by Scheduler.__init__)
_collector: Optional[PerfettoTraceCollector] = None


def get_perfetto_trace_collector() -> Optional["PerfettoTraceCollector"]:
    return _collector


def set_perfetto_trace_collector(collector: "PerfettoTraceCollector"):
    global _collector
    _collector = collector


class PerfettoTraceCollector:
    """Collects trace events and exports as Perfetto-compatible JSON.

    Each rank (TP/PP/DP) has its own collector instance.
    Events are stored in memory and flushed to disk on demand.
    """

    def __init__(
        self,
        tp_rank: int = 0,
        pp_rank: int = 0,
        dp_rank: int = 0,
        tp_size: int = 1,
        pp_size: int = 1,
        dp_size: int = 1,
        max_req_tracks: int = 512,
    ):
        self.tp_rank = tp_rank
        self.pp_rank = pp_rank
        self.dp_rank = dp_rank
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.dp_size = dp_size
        self.max_req_tracks = max_req_tracks

        self.pid = dp_rank * 10000 + pp_rank * 100 + tp_rank

        self._enabled = False
        self._events: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

        self._rid_to_tid: Dict[str, int] = {}
        self._next_req_tid = _Tid.REQ_BASE
        self._req_tid_pool: List[int] = []

    # =================== Lifecycle ===================

    def enable(self):
        self._enabled = True
        self._add_metadata_events()

    def disable(self):
        self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _add_metadata_events(self):
        rank_label = f"TP{self.tp_rank}"
        if self.pp_size > 1:
            rank_label += f"-PP{self.pp_rank}"
        if self.dp_size > 1:
            rank_label += f"-DP{self.dp_rank}"

        self._emit(
            {
                "name": "process_name",
                "ph": "M",
                "pid": self.pid,
                "tid": 0,
                "args": {"name": f"Rank [{rank_label}]"},
            }
        )
        self._emit(
            {
                "name": "process_sort_index",
                "ph": "M",
                "pid": self.pid,
                "tid": 0,
                "args": {"sort_index": self.pid},
            }
        )
        for tid, label in [
            (_Tid.SCHEDULER, "Scheduler"),
            (_Tid.BATCH_FORWARD, "Batch Forward"),
        ]:
            self._emit(
                {
                    "name": "thread_name",
                    "ph": "M",
                    "pid": self.pid,
                    "tid": tid,
                    "args": {"name": label},
                }
            )

    def _emit(self, event: Dict[str, Any]):
        with self._lock:
            self._events.append(event)

    # =================== Request TID allocation ===================

    def _get_req_tid(self, rid: str) -> int:
        if rid in self._rid_to_tid:
            return self._rid_to_tid[rid]

        if self._req_tid_pool:
            tid = self._req_tid_pool.pop()
        else:
            tid = self._next_req_tid
            self._next_req_tid += 1
            if self._next_req_tid >= _Tid.REQ_BASE + self.max_req_tracks:
                self._next_req_tid = _Tid.REQ_BASE

        self._rid_to_tid[rid] = tid
        short_rid = rid.split("_")[-1][:12]
        self._emit(
            {
                "name": "thread_name",
                "ph": "M",
                "pid": self.pid,
                "tid": tid,
                "args": {"name": f"Req {short_rid}"},
            }
        )
        return tid

    def _release_req_tid(self, rid: str):
        if rid in self._rid_to_tid:
            tid = self._rid_to_tid.pop(rid)
            self._req_tid_pool.append(tid)

    # =================== Scheduler Events ===================

    def sched_begin(self, name: str, ts: float = None):
        if not self._enabled:
            return
        self._emit(
            {
                "name": name,
                "cat": _Cat.SCHEDULER,
                "ph": "B",
                "ts": ts or _ts_us(),
                "pid": self.pid,
                "tid": _Tid.SCHEDULER,
            }
        )

    def sched_end(self, name: str, ts: float = None, args: Dict = None):
        if not self._enabled:
            return
        ev = {
            "name": name,
            "cat": _Cat.SCHEDULER,
            "ph": "E",
            "ts": ts or _ts_us(),
            "pid": self.pid,
            "tid": _Tid.SCHEDULER,
        }
        if args:
            ev["args"] = args
        self._emit(ev)

    def sched_instant(self, name: str, args: Dict = None, ts: float = None):
        if not self._enabled:
            return
        ev = {
            "name": name,
            "cat": _Cat.SCHEDULER,
            "ph": "i",
            "s": "t",
            "ts": ts or _ts_us(),
            "pid": self.pid,
            "tid": _Tid.SCHEDULER,
        }
        if args:
            ev["args"] = args
        self._emit(ev)

    # =================== Batch Events ===================

    def batch_begin(
        self,
        forward_mode: str,
        batch_size: int,
        num_tokens: int = 0,
        ts: float = None,
        extra: Dict = None,
    ):
        if not self._enabled:
            return
        a = {
            "forward_mode": forward_mode,
            "batch_size": batch_size,
            "num_tokens": num_tokens,
        }
        if extra:
            a.update(extra)
        self._emit(
            {
                "name": f"batch_{forward_mode}",
                "cat": _Cat.BATCH,
                "ph": "B",
                "ts": ts or _ts_us(),
                "pid": self.pid,
                "tid": _Tid.BATCH_FORWARD,
                "args": a,
            }
        )

    def batch_end(self, forward_mode: str, ts: float = None, args: Dict = None):
        if not self._enabled:
            return
        ev = {
            "name": f"batch_{forward_mode}",
            "cat": _Cat.BATCH,
            "ph": "E",
            "ts": ts or _ts_us(),
            "pid": self.pid,
            "tid": _Tid.BATCH_FORWARD,
        }
        if args:
            ev["args"] = args
        self._emit(ev)

    # =================== Request Events ===================

    def req_begin(self, rid: str, name: str, ts: float = None, args: Dict = None):
        if not self._enabled:
            return
        tid = self._get_req_tid(rid)
        ev = {
            "name": name,
            "cat": _Cat.REQUEST,
            "ph": "B",
            "ts": ts or _ts_us(),
            "pid": self.pid,
            "tid": tid,
        }
        if args:
            ev["args"] = args
        self._emit(ev)

    def req_end(self, rid: str, name: str, ts: float = None, args: Dict = None):
        if not self._enabled:
            return
        tid = self._get_req_tid(rid)
        ev = {
            "name": name,
            "cat": _Cat.REQUEST,
            "ph": "E",
            "ts": ts or _ts_us(),
            "pid": self.pid,
            "tid": tid,
        }
        if args:
            ev["args"] = args
        self._emit(ev)

    def req_instant(self, rid: str, name: str, ts: float = None, args: Dict = None):
        if not self._enabled:
            return
        tid = self._get_req_tid(rid)
        ev = {
            "name": name,
            "cat": _Cat.REQUEST,
            "ph": "i",
            "s": "t",
            "ts": ts or _ts_us(),
            "pid": self.pid,
            "tid": tid,
        }
        if args:
            ev["args"] = args
        self._emit(ev)

    def req_complete(
        self,
        rid: str,
        name: str,
        ts_us: float,
        dur_us: float,
        args: Dict = None,
    ):
        """Record a complete event (X) for a request phase."""
        if not self._enabled:
            return
        tid = self._get_req_tid(rid)
        ev = {
            "name": name,
            "cat": _Cat.REQUEST,
            "ph": "X",
            "ts": ts_us,
            "dur": dur_us,
            "pid": self.pid,
            "tid": tid,
        }
        if args:
            ev["args"] = args
        self._emit(ev)

    def req_finish(self, rid: str):
        if not self._enabled:
            return
        self._release_req_tid(rid)

    # =================== Counter Events ===================

    def counter(self, name: str, values: Dict[str, Any], ts: float = None):
        if not self._enabled:
            return
        self._emit(
            {
                "name": name,
                "ph": "C",
                "ts": ts or _ts_us(),
                "pid": self.pid,
                "tid": _Tid.COUNTERS,
                "args": values,
            }
        )

    # =================== Export ===================

    def dump(self, output_path: str, compress: bool = True) -> str:
        with self._lock:
            events = list(self._events)

        trace_data = {
            "traceEvents": events,
            "displayTimeUnit": "ms",
            "metadata": {
                "tp_rank": self.tp_rank,
                "pp_rank": self.pp_rank,
                "dp_rank": self.dp_rank,
                "tp_size": self.tp_size,
                "pp_size": self.pp_size,
                "dp_size": self.dp_size,
            },
        }

        json_str = json.dumps(trace_data)
        if compress:
            if not output_path.endswith(".gz"):
                output_path += ".gz"
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with gzip.open(output_path, "wb") as f:
                f.write(json_str.encode("utf-8"))
        else:
            if output_path.endswith(".gz"):
                output_path = output_path[:-3]
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "w") as f:
                f.write(json_str)

        logger.info(
            f"Perfetto trace dumped to {output_path} ({len(events)} events, "
            f"TP{self.tp_rank}/PP{self.pp_rank}/DP{self.dp_rank})"
        )
        return output_path

    def clear(self):
        with self._lock:
            self._events.clear()
            self._rid_to_tid.clear()
            self._req_tid_pool.clear()
            self._next_req_tid = _Tid.REQ_BASE

    def event_count(self) -> int:
        with self._lock:
            return len(self._events)


class PerfettoTraceMerger:
    """Merge Perfetto trace files from multiple ranks into a single file."""

    def merge(
        self,
        trace_files: List[str],
        output_path: str,
        compress: bool = True,
    ) -> str:
        all_events = []
        metadata_list = []

        for path in sorted(trace_files):
            trace_data = self._load_trace(path)
            if trace_data is None:
                continue
            all_events.extend(trace_data.get("traceEvents", []))
            metadata_list.append(trace_data.get("metadata", {}))

        merged = {
            "traceEvents": all_events,
            "displayTimeUnit": "ms",
            "metadata": {
                "merged": True,
                "num_ranks": len(metadata_list),
                "ranks": metadata_list,
            },
        }

        json_str = json.dumps(merged)
        if compress:
            if not output_path.endswith(".gz"):
                output_path += ".gz"
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with gzip.open(output_path, "wb") as f:
                f.write(json_str.encode("utf-8"))
        else:
            if output_path.endswith(".gz"):
                output_path = output_path[:-3]
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "w") as f:
                f.write(json_str)

        logger.info(
            f"Merged {len(trace_files)} trace files -> {output_path} "
            f"({len(all_events)} total events)"
        )
        return output_path

    def _load_trace(self, path: str) -> Optional[Dict]:
        try:
            if path.endswith(".gz"):
                with gzip.open(path, "rt", encoding="utf-8") as f:
                    return json.load(f)
            else:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load trace file {path}: {e}")
            return None

    def discover_and_merge(
        self,
        trace_dir: str,
        prefix: str = "perfetto-trace",
        output_path: Optional[str] = None,
        compress: bool = True,
    ) -> str:
        patterns = [
            os.path.join(trace_dir, f"{prefix}*.json.gz"),
            os.path.join(trace_dir, f"{prefix}*.json"),
        ]
        trace_files = []
        for pattern in patterns:
            trace_files.extend(globmod.glob(pattern))

        trace_files = [
            f for f in trace_files if "merged" not in os.path.basename(f)
        ]
        trace_files = list(set(trace_files))

        if not trace_files:
            raise ValueError(
                f"No trace files found in {trace_dir} with prefix {prefix}"
            )

        if output_path is None:
            ext = ".json.gz" if compress else ".json"
            output_path = os.path.join(trace_dir, f"merged-{prefix}{ext}")

        return self.merge(trace_files, output_path, compress)
