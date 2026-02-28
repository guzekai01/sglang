#!/usr/bin/env python3
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
"""CLI tool to merge Perfetto trace files from multiple TP/PP/DP ranks.

Usage:
    python -m sglang.srt.tools.merge_perfetto_traces --trace-dir /tmp/sglang_perfetto_traces
    python -m sglang.srt.tools.merge_perfetto_traces --trace-dir /tmp/sglang_perfetto_traces --output merged.json.gz
    python -m sglang.srt.tools.merge_perfetto_traces file1.json.gz file2.json.gz --output merged.json.gz
"""

import argparse
import logging
import sys

from sglang.srt.observability.perfetto_trace import PerfettoTraceMerger

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Merge Perfetto trace files from multiple ranks (TP/PP/DP)."
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Trace files to merge. If not specified, use --trace-dir to auto-discover.",
    )
    parser.add_argument(
        "--trace-dir",
        type=str,
        default=None,
        help="Directory containing trace files to merge.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="perfetto-trace",
        help="File prefix to match when discovering trace files (default: perfetto-trace).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file path for the merged trace.",
    )
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Do not gzip compress the output.",
    )

    args = parser.parse_args()
    merger = PerfettoTraceMerger()

    compress = not args.no_compress

    if args.files:
        if not args.output:
            args.output = "merged-perfetto-trace.json.gz"
        output_path = merger.merge(args.files, args.output, compress=compress)
    elif args.trace_dir:
        output_path = merger.discover_and_merge(
            args.trace_dir,
            prefix=args.prefix,
            output_path=args.output,
            compress=compress,
        )
    else:
        parser.error("Provide trace files or --trace-dir")
        sys.exit(1)

    logger.info(f"Merged trace written to: {output_path}")
    logger.info("Open in Perfetto UI: https://ui.perfetto.dev/")


if __name__ == "__main__":
    main()
