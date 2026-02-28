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

Generates Chrome Trace Event Format JSON loadable in
Perfetto UI (https://ui.perfetto.dev/) or chrome://tracing.

No external dependencies beyond the Python standard library.
Controlled via the existing start_profile / stop_profile API
by including "PERFETTO" in the activities list.
"""

from __future__ import annotations

import glob as globmod
import gzip
import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

try:
    import pynvml
except ImportError:
    pynvml = None

BYTES_TO_GIB = 1024**3
US_PER_SEC = 1_000_000.0


def _ts_us() -> float:
    """Current wall-clock timestamp in microseconds."""
    return time.time() * US_PER_SEC


# ---------------------------------------------------------------------------
# GPU stats background monitor
# ---------------------------------------------------------------------------


class GPUStatsMonitor:
    """Periodically samples GPU metrics via NVML in a daemon thread.

    Produces Chrome-trace counter ("C") events that are merged into the
    final Perfetto trace.
    """

    DEFAULT_INTERVAL_MS = 100

    def __init__(self, gpu_id: int, pid: Union[str, int]) -> None:
        self._pid = pid
        self._gpu_id = gpu_id
        self._available = False
        self._events: List[Dict] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._handle = None
        self._nvml_ok = False
        self._interval_s = self.DEFAULT_INTERVAL_MS / 1000.0

        if pynvml is None:
            return
        try:
            pynvml.nvmlInit()
            self._nvml_ok = True
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            self._available = True
        except Exception as exc:
            logger.warning("GPUStatsMonitor: NVML init failed – %s", exc)
            if self._nvml_ok:
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    pass
                self._nvml_ok = False

    # -- lifecycle --

    def start(self) -> None:
        if not self._available or self._running:
            return
        self._running = True
        self._stop_event.clear()
        self._ready_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self._ready_event.wait(timeout=5.0)

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def take_events(self) -> List[Dict]:
        events = self._events
        self._events = []
        return events

    # -- internals --

    def _loop(self) -> None:
        import torch

        self._ready_event.set()
        next_tick = time.perf_counter()
        while not self._stop_event.is_set():
            next_tick += self._interval_s
            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                temp = pynvml.nvmlDeviceGetTemperature(
                    self._handle, pynvml.NVML_TEMPERATURE_GPU
                )
                torch_alloc = float(torch.cuda.memory_allocated(self._gpu_id))
                torch_reserved = float(torch.cuda.memory_reserved(self._gpu_id))

                args = {
                    "mem_total_gib": round(mem.total / BYTES_TO_GIB, 2),
                    "mem_free_gib": round(mem.free / BYTES_TO_GIB, 2),
                    "mem_used_gib": round(mem.used / BYTES_TO_GIB, 2),
                    "torch_alloc_gib": round(torch_alloc / BYTES_TO_GIB, 2),
                    "torch_reserved_gib": round(torch_reserved / BYTES_TO_GIB, 2),
                    "gpu_util_pct": util.gpu,
                    "mem_util_pct": util.memory,
                    "temperature_c": temp,
                }
                self._events.append(
                    {
                        "name": "gpu_stats",
                        "cat": "gpu_stats",
                        "ph": "C",
                        "ts": _ts_us(),
                        "pid": self._pid,
                        "tid": "gpu_stats",
                        "args": args,
                    }
                )
            except Exception as exc:
                logger.debug("GPUStatsMonitor sample error: %s", exc)

            sleep_t = next_tick - time.perf_counter()
            if sleep_t > 0:
                self._stop_event.wait(timeout=sleep_t)

    def __del__(self) -> None:
        if self._nvml_ok and pynvml is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Thread-ID constants
# ---------------------------------------------------------------------------

class _Tid:
    SCHEDULER = 0
    BATCH_FORWARD = 1
    REQ_BASE = 100


# ---------------------------------------------------------------------------
# Core Perfetto trace collector
# ---------------------------------------------------------------------------


class PerfettoTraceCollector:
    """Collects Chrome Trace Event Format events and exports to JSON.

    Lifecycle:
        1. Construct with rank info and output path.
        2. ``start()`` – begin recording events (and GPU monitor).
        3. Events flow in from the scheduler mixin hooks.
        4. ``stop()``  – stop recording and flush to disk.
    """

    def __init__(
        self,
        output_path: str,
        tp_rank: int = 0,
        pp_rank: int = 0,
        dp_rank: int = 0,
        tp_size: int = 1,
        pp_size: int = 1,
        dp_size: int = 1,
        gpu_id: Optional[int] = None,
        max_req_tracks: int = 512,
    ):
        self.tp_rank = tp_rank
        self.pp_rank = pp_rank
        self.dp_rank = dp_rank
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.dp_size = dp_size
        self._output_path = output_path
        self.max_req_tracks = max_req_tracks

        self.pid = dp_rank * 10000 + pp_rank * 100 + tp_rank

        self._enabled = False
        self._events: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

        # req-tid pool
        self._rid_to_tid: Dict[str, int] = {}
        self._next_req_tid = _Tid.REQ_BASE
        self._req_tid_pool: List[int] = []

        # GPU monitor
        self._gpu_monitor: Optional[GPUStatsMonitor] = None
        if gpu_id is not None:
            self._gpu_monitor = GPUStatsMonitor(gpu_id, self.pid)

    # =================== lifecycle ===================

    def start(self) -> None:
        self._enabled = True
        self._add_metadata()
        if self._gpu_monitor is not None:
            self._gpu_monitor.start()

    def stop(self) -> str:
        """Stop collecting, merge GPU events, dump to disk, return path."""
        self._enabled = False
        if self._gpu_monitor is not None:
            self._gpu_monitor.stop()
            gpu_events = self._gpu_monitor.take_events()
            with self._lock:
                self._events.extend(gpu_events)
        return self._dump()

    @property
    def enabled(self) -> bool:
        return self._enabled

    # =================== internal helpers ===================

    def _add_metadata(self) -> None:
        label = f"TP{self.tp_rank}"
        if self.pp_size > 1:
            label += f"-PP{self.pp_rank}"
        if self.dp_size > 1:
            label += f"-DP{self.dp_rank}"

        self._emit(
            {"name": "process_name", "ph": "M", "pid": self.pid, "tid": 0,
             "args": {"name": f"Rank [{label}]"}}
        )
        self._emit(
            {"name": "process_sort_index", "ph": "M", "pid": self.pid, "tid": 0,
             "args": {"sort_index": self.pid}}
        )
        for tid, name in [
            (_Tid.SCHEDULER, "Scheduler"),
            (_Tid.BATCH_FORWARD, "Batch Forward"),
        ]:
            self._emit(
                {"name": "thread_name", "ph": "M", "pid": self.pid, "tid": tid,
                 "args": {"name": name}}
            )

    def _emit(self, ev: Dict[str, Any]) -> None:
        with self._lock:
            self._events.append(ev)

    def _get_req_tid(self, rid: str) -> int:
        if rid in self._rid_to_tid:
            return self._rid_to_tid[rid]
        tid = (
            self._req_tid_pool.pop()
            if self._req_tid_pool
            else self._alloc_req_tid()
        )
        self._rid_to_tid[rid] = tid
        short = rid.split("_")[-1][:12]
        self._emit(
            {"name": "thread_name", "ph": "M", "pid": self.pid, "tid": tid,
             "args": {"name": f"Req {short}"}}
        )
        return tid

    def _alloc_req_tid(self) -> int:
        tid = self._next_req_tid
        self._next_req_tid += 1
        if self._next_req_tid >= _Tid.REQ_BASE + self.max_req_tracks:
            self._next_req_tid = _Tid.REQ_BASE
        return tid

    def _release_req_tid(self, rid: str) -> None:
        if rid in self._rid_to_tid:
            self._req_tid_pool.append(self._rid_to_tid.pop(rid))

    # =================== scheduler events ===================

    def sched_begin(self, name: str) -> None:
        if not self._enabled:
            return
        self._emit({"name": name, "cat": "scheduler", "ph": "B",
                     "ts": _ts_us(), "pid": self.pid, "tid": _Tid.SCHEDULER})

    def sched_end(self, name: str, args: Dict = None) -> None:
        if not self._enabled:
            return
        ev: Dict[str, Any] = {"name": name, "cat": "scheduler", "ph": "E",
                               "ts": _ts_us(), "pid": self.pid, "tid": _Tid.SCHEDULER}
        if args:
            ev["args"] = args
        self._emit(ev)

    # =================== batch events ===================

    def batch_begin(self, mode: str, bs: int, ntokens: int = 0,
                    extra: Dict = None) -> None:
        if not self._enabled:
            return
        a: Dict[str, Any] = {"forward_mode": mode, "batch_size": bs,
                              "num_tokens": ntokens}
        if extra:
            a.update(extra)
        self._emit({"name": f"batch_{mode}", "cat": "batch", "ph": "B",
                     "ts": _ts_us(), "pid": self.pid, "tid": _Tid.BATCH_FORWARD,
                     "args": a})

    def batch_end(self, mode: str, args: Dict = None) -> None:
        if not self._enabled:
            return
        ev: Dict[str, Any] = {"name": f"batch_{mode}", "cat": "batch", "ph": "E",
                               "ts": _ts_us(), "pid": self.pid, "tid": _Tid.BATCH_FORWARD}
        if args:
            ev["args"] = args
        self._emit(ev)

    # =================== request events ===================

    def req_begin(self, rid: str, name: str, args: Dict = None) -> None:
        if not self._enabled:
            return
        tid = self._get_req_tid(rid)
        ev: Dict[str, Any] = {"name": name, "cat": "request", "ph": "B",
                               "ts": _ts_us(), "pid": self.pid, "tid": tid}
        if args:
            ev["args"] = args
        self._emit(ev)

    def req_end(self, rid: str, name: str, args: Dict = None) -> None:
        if not self._enabled:
            return
        tid = self._get_req_tid(rid)
        ev: Dict[str, Any] = {"name": name, "cat": "request", "ph": "E",
                               "ts": _ts_us(), "pid": self.pid, "tid": tid}
        if args:
            ev["args"] = args
        self._emit(ev)

    def req_instant(self, rid: str, name: str, args: Dict = None) -> None:
        if not self._enabled:
            return
        tid = self._get_req_tid(rid)
        ev: Dict[str, Any] = {"name": name, "cat": "request", "ph": "i", "s": "t",
                               "ts": _ts_us(), "pid": self.pid, "tid": tid}
        if args:
            ev["args"] = args
        self._emit(ev)

    def req_finish(self, rid: str) -> None:
        if not self._enabled:
            return
        self._release_req_tid(rid)

    # =================== counter events ===================

    def counter(self, name: str, values: Dict[str, Any]) -> None:
        if not self._enabled:
            return
        self._emit({"name": name, "ph": "C", "ts": _ts_us(),
                     "pid": self.pid, "tid": 0, "args": values})

    # =================== dump ===================

    def _dump(self) -> str:
        with self._lock:
            events = list(self._events)
            self._events.clear()
            self._rid_to_tid.clear()
            self._req_tid_pool.clear()
            self._next_req_tid = _Tid.REQ_BASE

        data = {
            "traceEvents": events,
            "displayTimeUnit": "ms",
            "metadata": {
                "tp_rank": self.tp_rank, "pp_rank": self.pp_rank,
                "dp_rank": self.dp_rank, "tp_size": self.tp_size,
                "pp_size": self.pp_size, "dp_size": self.dp_size,
            },
        }
        path = self._output_path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        json_bytes = json.dumps(data).encode("utf-8")
        if path.endswith(".gz"):
            with gzip.open(path, "wb") as f:
                f.write(json_bytes)
        else:
            with open(path, "wb") as f:
                f.write(json_bytes)

        logger.info(
            "Perfetto trace: %d events -> %s (TP%d/PP%d/DP%d)",
            len(events), path, self.tp_rank, self.pp_rank, self.dp_rank,
        )
        return path

    def event_count(self) -> int:
        with self._lock:
            return len(self._events)


# ---------------------------------------------------------------------------
# Multi-rank merge
# ---------------------------------------------------------------------------


class PerfettoTraceMerger:
    """Merge Perfetto trace files from multiple TP/PP/DP ranks."""

    def merge(self, trace_files: List[str], output_path: str,
              compress: bool = True) -> str:
        all_events: List[Dict] = []
        meta_list: List[Dict] = []
        for p in sorted(trace_files):
            d = self._load(p)
            if d is None:
                continue
            all_events.extend(d.get("traceEvents", []))
            meta_list.append(d.get("metadata", {}))

        merged = {
            "traceEvents": all_events,
            "displayTimeUnit": "ms",
            "metadata": {"merged": True, "num_ranks": len(meta_list),
                         "ranks": meta_list},
        }
        js = json.dumps(merged).encode("utf-8")
        if compress and not output_path.endswith(".gz"):
            output_path += ".gz"
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        if output_path.endswith(".gz"):
            with gzip.open(output_path, "wb") as f:
                f.write(js)
        else:
            with open(output_path, "wb") as f:
                f.write(js)
        logger.info("Merged %d files -> %s (%d events)",
                     len(trace_files), output_path, len(all_events))
        return output_path

    def _load(self, path: str) -> Optional[Dict]:
        try:
            opener = gzip.open if path.endswith(".gz") else open
            with opener(path, "rt", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.error("Failed to load %s: %s", path, exc)
            return None

    def discover_and_merge(self, trace_dir: str, prefix: str = "perfetto-trace",
                           output_path: Optional[str] = None,
                           compress: bool = True) -> str:
        files: List[str] = []
        for ext in ("*.json.gz", "*.json"):
            files.extend(globmod.glob(os.path.join(trace_dir, f"{prefix}{ext}")))
        files = [f for f in set(files) if "merged" not in os.path.basename(f)]
        if not files:
            raise ValueError(f"No trace files in {trace_dir} with prefix {prefix}")
        if output_path is None:
            sfx = ".json.gz" if compress else ".json"
            output_path = os.path.join(trace_dir, f"merged-{prefix}{sfx}")
        return self.merge(files, output_path, compress)
