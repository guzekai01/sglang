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
"""Perfetto trace integration mixin for the Scheduler."""

from __future__ import annotations

import atexit
import logging
import os
import time
from typing import TYPE_CHECKING, Optional

from sglang.srt.observability.perfetto_trace import (
    PerfettoTraceCollector,
    _ts_us,
    set_perfetto_trace_collector,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


class SchedulerPerfettoTraceMixin:
    """Mixin that adds Perfetto tracing to the Scheduler."""

    def init_perfetto_trace(self: "Scheduler"):
        self.perfetto_enabled = getattr(
            self.server_args, "enable_perfetto_trace", False
        )
        self.perfetto_collector: Optional[PerfettoTraceCollector] = None

        if not self.perfetto_enabled:
            return

        trace_dir = getattr(self.server_args, "perfetto_trace_dir", None)
        if trace_dir is None:
            trace_dir = "/tmp/sglang_perfetto_traces"
        self._perfetto_trace_dir = trace_dir
        os.makedirs(trace_dir, exist_ok=True)

        self.perfetto_collector = PerfettoTraceCollector(
            tp_rank=self.tp_rank,
            pp_rank=self.pp_rank,
            dp_rank=self.dp_rank if self.dp_rank is not None else 0,
            tp_size=self.tp_size,
            pp_size=self.pp_size,
            dp_size=self.dp_size,
        )
        self.perfetto_collector.enable()
        set_perfetto_trace_collector(self.perfetto_collector)

        atexit.register(self._dump_perfetto_trace_on_exit)

        logger.info(
            f"Perfetto trace enabled for TP{self.tp_rank}/PP{self.pp_rank}/"
            f"DP{self.dp_rank if self.dp_rank is not None else 0}, "
            f"output dir: {trace_dir}"
        )

    def _dump_perfetto_trace_on_exit(self):
        if self.perfetto_collector and self.perfetto_collector.enabled:
            self._dump_perfetto_trace()

    def _dump_perfetto_trace(self) -> Optional[str]:
        if not self.perfetto_collector:
            return None

        dp = self.dp_rank if self.dp_rank is not None else 0
        filename = f"perfetto-trace-TP{self.tp_rank}-PP{self.pp_rank}-DP{dp}.json.gz"
        path = os.path.join(self._perfetto_trace_dir, filename)
        return self.perfetto_collector.dump(path)

    def dump_perfetto_trace(self: "Scheduler") -> Optional[str]:
        """Public method callable via RPC to dump the current trace."""
        return self._dump_perfetto_trace()

    # =================== Scheduler loop tracing ===================

    def perfetto_on_recv_requests_begin(self: "Scheduler"):
        if not self.perfetto_enabled:
            return
        self.perfetto_collector.sched_begin("recv_requests")

    def perfetto_on_recv_requests_end(self: "Scheduler", num_reqs: int):
        if not self.perfetto_enabled:
            return
        self.perfetto_collector.sched_end(
            "recv_requests",
            args={"num_reqs": num_reqs} if num_reqs > 0 else None,
        )

    def perfetto_on_get_batch_begin(self: "Scheduler"):
        if not self.perfetto_enabled:
            return
        self.perfetto_collector.sched_begin("get_next_batch")

    def perfetto_on_get_batch_end(self: "Scheduler", batch: Optional["ScheduleBatch"]):
        if not self.perfetto_enabled:
            return
        args = None
        if batch is not None:
            args = {
                "forward_mode": batch.forward_mode.name,
                "batch_size": batch.batch_size(),
            }
        self.perfetto_collector.sched_end("get_next_batch", args=args)

    def perfetto_on_run_batch_begin(self: "Scheduler", batch: "ScheduleBatch"):
        if not self.perfetto_enabled:
            return
        mode = batch.forward_mode.name.lower()
        bs = batch.batch_size()
        num_tokens = int(batch.seq_lens_sum) if hasattr(batch, "seq_lens_sum") and batch.seq_lens_sum else 0
        rids = [r.rid[:12] for r in batch.reqs[:8]]
        self.perfetto_collector.batch_begin(
            mode, bs, num_tokens,
            extra={"rids_sample": rids, "forward_ct": self.forward_ct},
        )

    def perfetto_on_run_batch_end(self: "Scheduler", batch: "ScheduleBatch"):
        if not self.perfetto_enabled:
            return
        mode = batch.forward_mode.name.lower()
        self.perfetto_collector.batch_end(mode)

    def perfetto_on_process_result_begin(self: "Scheduler", batch: "ScheduleBatch"):
        if not self.perfetto_enabled:
            return
        self.perfetto_collector.sched_begin("process_result")

    def perfetto_on_process_result_end(self: "Scheduler", batch: "ScheduleBatch"):
        if not self.perfetto_enabled:
            return
        self.perfetto_collector.sched_end("process_result")

    # =================== Request lifecycle tracing ===================

    def perfetto_on_req_queue(self: "Scheduler", req: "Req"):
        """Called when a request enters the waiting queue."""
        if not self.perfetto_enabled:
            return
        self.perfetto_collector.req_begin(
            req.rid,
            "lifecycle",
            args={
                "rid": req.rid[:16],
                "input_len": len(req.origin_input_ids),
            },
        )
        self.perfetto_collector.req_begin(req.rid, "queue_wait")

    def perfetto_on_req_prefill_begin(self: "Scheduler", req: "Req"):
        """Called when a request begins prefill (forward entry)."""
        if not self.perfetto_enabled:
            return
        self.perfetto_collector.req_end(req.rid, "queue_wait")
        self.perfetto_collector.req_begin(
            req.rid,
            "prefill",
            args={
                "extend_input_len": req.extend_input_len,
                "cached_tokens": req.cached_tokens,
            },
        )

    def perfetto_on_req_prefill_end(self: "Scheduler", req: "Req"):
        """Called when prefill phase completes for a request."""
        if not self.perfetto_enabled:
            return
        self.perfetto_collector.req_end(req.rid, "prefill")
        self.perfetto_collector.req_begin(req.rid, "decode")

    def perfetto_on_req_decode_step(self: "Scheduler", req: "Req", decode_ct: int):
        """Called after each decode step for a request."""
        if not self.perfetto_enabled:
            return
        self.perfetto_collector.req_instant(
            req.rid,
            f"decode_step_{decode_ct}",
            args={"output_len": len(req.output_ids)},
        )

    def perfetto_on_req_finish(
        self: "Scheduler",
        req: "Req",
        finish_reason: str = "",
    ):
        """Called when a request finishes."""
        if not self.perfetto_enabled:
            return
        self.perfetto_collector.req_end(
            req.rid,
            "decode",
            args={"output_len": len(req.output_ids)},
        )
        self.perfetto_collector.req_end(
            req.rid,
            "lifecycle",
            args={
                "finish_reason": finish_reason,
                "total_output_tokens": len(req.output_ids),
                "total_input_tokens": len(req.origin_input_ids),
            },
        )
        self.perfetto_collector.req_finish(req.rid)

    # =================== Counter events ===================

    def perfetto_counters(self: "Scheduler"):
        """Emit counter events for queue/running sizes."""
        if not self.perfetto_enabled:
            return
        ts = _ts_us()
        self.perfetto_collector.counter(
            "scheduler_stats",
            {
                "num_waiting": len(self.waiting_queue),
                "num_running": len(self.running_batch.reqs),
            },
            ts=ts,
        )
