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
"""Perfetto trace integration mixin for the Scheduler.

The collector is created on ``init_profile`` when the activities list
contains ``"PERFETTO"`` and follows the start / stop profile lifecycle.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Optional

from sglang.srt.observability.perfetto_trace import PerfettoTraceCollector

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


class SchedulerPerfettoTraceMixin:
    """Mixin that adds Perfetto tracing to the Scheduler.

    The public ``perfetto_*`` hooks are called from the scheduler event loops
    and output-processor.  They are all no-ops when the collector is ``None``
    or not enabled, so the hot-path cost is a single attribute check.
    """

    def init_perfetto_collector(
        self: "Scheduler",
        output_dir: str,
        profile_id: str,
    ) -> None:
        """Create the collector.  Called from ``init_profile``."""
        dp = self.dp_rank if self.dp_rank is not None else 0

        parts = [profile_id, f"TP-{self.tp_rank}"]
        if self.pp_size > 1:
            parts.append(f"PP-{self.pp_rank}")
        if self.dp_size > 1:
            parts.append(f"DP-{dp}")
        filename = "-".join(parts) + "-perfetto.json.gz"
        output_path = os.path.join(output_dir, filename)

        self.perfetto_collector = PerfettoTraceCollector(
            output_path=output_path,
            tp_rank=self.tp_rank,
            pp_rank=self.pp_rank,
            dp_rank=dp,
            tp_size=self.tp_size,
            pp_size=self.pp_size,
            dp_size=self.dp_size,
            gpu_id=self.gpu_id,
        )

    def start_perfetto_profile(self: "Scheduler") -> None:
        """Start recording.  Called from ``start_profile``."""
        if self.perfetto_collector is not None:
            self.perfetto_collector.start()

    def stop_perfetto_profile(self: "Scheduler") -> Optional[str]:
        """Stop recording and dump.  Called from ``stop_profile``.

        Returns the output file path, or None.
        """
        if self.perfetto_collector is not None and self.perfetto_collector.enabled:
            path = self.perfetto_collector.stop()
            self.perfetto_collector = None
            return path
        return None

    # ------------------------------------------------------------------ #
    #  Convenience predicate used in every hook below.                    #
    # ------------------------------------------------------------------ #

    @property
    def _pt(self: "Scheduler") -> Optional[PerfettoTraceCollector]:
        c = getattr(self, "perfetto_collector", None)
        if c is not None and c.enabled:
            return c
        return None

    # =================== scheduler loop ===================

    def perfetto_on_recv_requests_begin(self: "Scheduler") -> None:
        c = self._pt
        if c:
            c.sched_begin("recv_requests")

    def perfetto_on_recv_requests_end(self: "Scheduler", num_reqs: int) -> None:
        c = self._pt
        if c:
            c.sched_end("recv_requests",
                         args={"num_reqs": num_reqs} if num_reqs else None)

    def perfetto_on_get_batch_begin(self: "Scheduler") -> None:
        c = self._pt
        if c:
            c.sched_begin("get_next_batch")

    def perfetto_on_get_batch_end(
        self: "Scheduler", batch: Optional["ScheduleBatch"]
    ) -> None:
        c = self._pt
        if c:
            args = None
            if batch is not None:
                args = {"forward_mode": batch.forward_mode.name,
                        "batch_size": batch.batch_size()}
            c.sched_end("get_next_batch", args=args)

    def perfetto_on_run_batch_begin(
        self: "Scheduler", batch: "ScheduleBatch"
    ) -> None:
        c = self._pt
        if c:
            mode = batch.forward_mode.name.lower()
            bs = batch.batch_size()
            ntok = int(batch.seq_lens_sum) if getattr(batch, "seq_lens_sum", None) else 0
            c.batch_begin(mode, bs, ntok,
                          extra={"forward_ct": self.forward_ct})

    def perfetto_on_run_batch_end(
        self: "Scheduler", batch: "ScheduleBatch"
    ) -> None:
        c = self._pt
        if c:
            c.batch_end(batch.forward_mode.name.lower())

    def perfetto_on_process_result_begin(
        self: "Scheduler", batch: "ScheduleBatch"
    ) -> None:
        c = self._pt
        if c:
            c.sched_begin("process_result")

    def perfetto_on_process_result_end(
        self: "Scheduler", batch: "ScheduleBatch"
    ) -> None:
        c = self._pt
        if c:
            c.sched_end("process_result")

    # =================== request lifecycle ===================

    def perfetto_on_req_queue(self: "Scheduler", req: "Req") -> None:
        c = self._pt
        if c:
            c.req_begin(req.rid, "lifecycle",
                        args={"rid": req.rid[:16],
                              "input_len": len(req.origin_input_ids)})
            c.req_begin(req.rid, "queue_wait")

    def perfetto_on_req_prefill_begin(self: "Scheduler", req: "Req") -> None:
        c = self._pt
        if c:
            c.req_end(req.rid, "queue_wait")
            c.req_begin(req.rid, "prefill",
                        args={"extend_input_len": req.extend_input_len,
                              "cached_tokens": req.cached_tokens})

    def perfetto_on_req_prefill_end(self: "Scheduler", req: "Req") -> None:
        c = self._pt
        if c:
            c.req_end(req.rid, "prefill")
            c.req_begin(req.rid, "decode")

    def perfetto_on_req_decode_step(
        self: "Scheduler", req: "Req", decode_ct: int
    ) -> None:
        c = self._pt
        if c:
            c.req_instant(req.rid, f"decode_{decode_ct}",
                          args={"output_len": len(req.output_ids)})

    def perfetto_on_req_finish(
        self: "Scheduler", req: "Req", finish_reason: str = ""
    ) -> None:
        c = self._pt
        if c:
            c.req_end(req.rid, "decode",
                      args={"output_len": len(req.output_ids)})
            c.req_end(req.rid, "lifecycle",
                      args={"finish_reason": finish_reason,
                            "output_tokens": len(req.output_ids),
                            "input_tokens": len(req.origin_input_ids)})
            c.req_finish(req.rid)

    # =================== counters ===================

    def perfetto_counters(self: "Scheduler") -> None:
        c = self._pt
        if c:
            c.counter("scheduler_stats", {
                "num_waiting": len(self.waiting_queue),
                "num_running": len(self.running_batch.reqs),
            })
