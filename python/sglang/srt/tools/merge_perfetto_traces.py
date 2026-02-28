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
    python -m sglang.srt.tools.merge_perfetto_traces --trace-dir /tmp/traces
    python -m sglang.srt.tools.merge_perfetto_traces f1.json.gz f2.json.gz -o merged.json.gz
"""

import argparse
import logging
import sys

from sglang.srt.observability.perfetto_trace import PerfettoTraceMerger

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    parser = argparse.ArgumentParser(
        description="Merge Perfetto trace files from multiple ranks (TP/PP/DP)."
    )
    parser.add_argument(
        "files", nargs="*",
        help="Trace files to merge. Use --trace-dir for auto-discovery instead.",
    )
    parser.add_argument("--trace-dir", type=str, default=None)
    parser.add_argument("--prefix", type=str, default="perfetto",
                        help="Filename prefix for auto-discovery (default: perfetto).")
    parser.add_argument("-o", "--output", type=str, default=None)
    parser.add_argument("--no-compress", action="store_true")

    args = parser.parse_args()
    merger = PerfettoTraceMerger()
    compress = not args.no_compress

    if args.files:
        out = args.output or "merged-perfetto-trace.json.gz"
        path = merger.merge(args.files, out, compress=compress)
    elif args.trace_dir:
        path = merger.discover_and_merge(
            args.trace_dir, prefix=args.prefix,
            output_path=args.output, compress=compress,
        )
    else:
        parser.error("Provide trace files or --trace-dir")
        sys.exit(1)

    print(f"Merged trace: {path}")
    print("Open in Perfetto UI: https://ui.perfetto.dev/")


if __name__ == "__main__":
    main()
