import json
import os
import tempfile
import unittest
from dataclasses import dataclass, field
from typing import List

from sglang.srt.observability.perfetto_trace import PerfettoTraceCollector
from sglang.srt.observability.perfetto_trace_mixin import SchedulerPerfettoTraceMixin
from sglang.test.ci.ci_register import register_cuda_ci
try:
    # Prefer the project-standard base when available (CI env has torch).
    from sglang.test.test_utils import CustomTestCase as _BaseTestCase
except Exception:
    # Allow running this pure-Python test in minimal environments.
    _BaseTestCase = unittest.TestCase

register_cuda_ci(est_time=5, suite="stage-b-test-small-1-gpu")


@dataclass
class _DummyReq:
    rid: str
    origin_input_ids: List[int]
    extend_input_len: int = 0
    cached_tokens: int = 0
    output_ids: List[int] = field(default_factory=list)


class _DummyScheduler(SchedulerPerfettoTraceMixin):
    def __init__(self, output_path: str) -> None:
        self.perfetto_collector = PerfettoTraceCollector(output_path=output_path)
        self.perfetto_collector.start()


class TestPerfettoChunkedPrefillTrace(_BaseTestCase):
    def test_chunked_prefill_only_one_prefill_begin(self):
        with tempfile.TemporaryDirectory() as td:
            out_path = os.path.join(td, "perfetto.json")
            sched = _DummyScheduler(out_path)

            req = _DummyReq(
                rid="rid_test_0001",
                origin_input_ids=list(range(8)),
                extend_input_len=8,
                cached_tokens=0,
            )

            # Request queued -> enters queue_wait span.
            sched.perfetto_on_req_queue(req)

            # First prefill chunk: should end queue_wait and begin prefill.
            sched.perfetto_on_req_prefill_begin(req)

            # Subsequent prefill chunks must NOT open another prefill span.
            req.extend_input_len = 16
            req.cached_tokens = 8
            sched.perfetto_on_req_prefill_begin(req)

            # Prefill completes once (last chunk) -> end prefill, begin decode.
            sched.perfetto_on_req_prefill_end(req)
            req.output_ids.extend([42, 43])
            sched.perfetto_on_req_finish(req, "eos")

            sched.perfetto_collector.stop()

            with open(out_path, "rt", encoding="utf-8") as f:
                trace = json.load(f)

            events = trace.get("traceEvents", [])

            def _count(name: str, ph: str) -> int:
                return sum(
                    1
                    for e in events
                    if e.get("cat") == "request"
                    and e.get("name") == name
                    and e.get("ph") == ph
                )

            self.assertEqual(_count("queue_wait", "B"), 1)
            self.assertEqual(_count("queue_wait", "E"), 1)
            self.assertEqual(_count("prefill", "B"), 1)
            self.assertEqual(_count("prefill", "E"), 1)
            self.assertEqual(_count("prefill_chunk", "i"), 1)
            self.assertEqual(_count("decode", "B"), 1)
            self.assertEqual(_count("decode", "E"), 1)


if __name__ == "__main__":
    unittest.main(verbosity=3)

