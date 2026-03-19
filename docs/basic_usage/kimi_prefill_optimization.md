# Kimi K2.5 System-Level Prefill Optimization on 2x8 H20

This guide focuses on minimizing **prefill latency / TTFT** for a **Kimi K2.5 deployment** on **2 nodes x 8 H20 GPUs**.

Scope notes:

- In the current SGLang codebase, **Kimi K2.5 is not a plain DeepSeek/K2 text-only path**. Its text stack is a **hybrid linear/full-attention MoE path**, so backend-level behavior is not identical to a pure K2 or DeepSeek-style backbone.
- This guide therefore focuses on **system-level and shared-runtime tuning**: topology, chunking, PP balance, profiling, and online-vs-offline validation.
- The runtime optimization in this PR still applies to K2.5 because it happens in the shared `ScheduleBatch -> ForwardBatch -> compute_position` prefill metadata path before model-specific kernels run.
- If you deploy **Kimi K2 / DeepSeek-style text backbones**, the same high-level topology and chunking workflow still applies, but backend-specific numbers may differ.
- **Kimi Linear** remains a separate model family with its own linear-attention path and should be tuned independently for final performance numbers.

## Model Scope

Use this guide for:

- **Kimi K2.5 on 2x8 H20**, where you care about end-to-end prefill / TTFT
- nearby K2/K2-like deployments when you want a **system-tuning workflow**, not a backend-specific kernel analysis

Do **not** read this guide as:

- a proof that K2.5 and K2 have identical kernel behavior
- a claim that one set of benchmark numbers transfers unchanged across K2.5, K2, and Kimi Linear
- a replacement for model-specific profiling

## Executive Summary

If your target is **the shortest prefill time** on 2x8 H20, prioritize the following in order:

1. **Use `TP=8, PP=2` as the default topology**
2. **Sweep fixed `--chunked-prefill-size` first**
3. **Then try dynamic chunking**
4. **Fix PP stage imbalance with `SGLANG_PP_LAYER_PARTITION` if needed**
5. **Only after that, optimize scheduler/control-plane overhead**

The shared runtime optimization in this PR improves the common prefill metadata path, but it is **not** the primary driver of Kimi K2.5 TTFT.

## What This PR Improves

This PR already lands three useful pieces:

1. **Shared prefill metadata optimization**
   - Precomputes and reuses `extend_start_loc`
   - Removes repeated O(bs^2) prefix-sum behavior from the hot prefill metadata path
   - Applies to the shared `extend/prefill` runtime path used before entering the model backbone

2. **Mixed prefill correctness hardening**
   - Recomputes offsets after mixed prefill/decode merges
   - Prevents stale metadata from silently poisoning later prefill work

3. **Pipeline-parallel correctness fixes**
   - Fixes Qwen3.5 PP local-layer execution and shard-aware weight loading
   - Hardens KimiLinear PP weight loading so non-local PP layers are not touched

## Modeled Impact for Kimi K2.5

For Kimi K2.5 deployments, prefill time can be simplified as:

```text
T_prefill
= T_scheduler
+ T_prefill_metadata
+ T_gpu_compute
+ T_pp_boundary_comm
+ T_pipeline_bubble
+ T_postprocess
```

The code changes in this PR mainly reduce:

- `T_prefill_metadata`

They do **not** directly reduce the largest terms in a long-context Kimi K2.5 prefill run:

- `T_gpu_compute`
- `T_pp_boundary_comm`
- `T_pipeline_bubble`

### Expected End-to-End Gain From This PR Alone

Use the following as a planning model, not a promise:

| Workload shape | Expected gain from this PR |
| --- | --- |
| Single request, long prompt, TTFT-focused | **0% - 1%** |
| Small batched prefill | **1% - 3%** |
| Larger batch / chunked prefill / mixed prefill | **2% - 6%** |

Interpretation:

- For **single-request ultra-long prefill**, the workload is dominated by model compute and PP bubbles, so runtime metadata cleanup has limited headroom.
- For **batched or chunked prefill**, the shared metadata path matters more, so this PR becomes more visible.

## Bigger Levers Than This PR

If your only target is **shorter Kimi K2.5 prefill**, the larger gains usually come from the following.
Treat the following ranges as **empirical tuning headroom**, not guaranteed improvements:

| Lever | Typical upside | Why it matters |
| --- | --- | --- |
| Fixed `chunked-prefill-size` sweep | **5% - 20%** | Directly changes chunk count, stage occupancy, and bubble size |
| Dynamic chunking after fixed-size tuning | **3% - 12%** on top of baseline | Reduces bubble growth as prefix length increases |
| PP stage balancing | **3% - 10%** | The slowest stage sets the pipeline speed |
| PD disaggregation for online serving | **10% - 30%** in mixed traffic | Removes decode interference from the prefill path |
| PP control-plane optimization | **3% - 10%** | Helps when many chunks or many concurrent requests are in flight |

## Recommended Topology for 2x8 H20

Use:

```text
TP = 8
PP = 2
```

That means:

- **one PP stage per node**
- **single-node TP8 inside each stage**

Why this is the default bet:

- Cross-node TP tends to pay a higher communication tax during prefill
- PP only communicates across the stage boundary
- For long-context TTFT, `TP=8, PP=2` is usually a better starting point than `TP=16, PP=1`

## Tuning Strategy

### Step 0: Establish a Clean Baseline

Start with:

- `TP=8`
- `PP=2`
- fixed chunking
- no dynamic chunking
- one long-prompt benchmark workload

Recommended initial settings:

```text
--chunked-prefill-size 4096
--max-running-requests 128
--mem-fraction-static 0.8
```

If the model is significantly more compute-heavy or MoE-heavy than your baseline expectation, also test:

```text
--chunked-prefill-size 6144
```

### Step 1: Sweep Fixed Chunk Size

Test these first:

- `4096`
- `5120`
- `6144`
- `7168`

Why:

- Too small: too many chunks, more scheduler overhead, more PP bubbles
- Too large: later chunks become slower and stage imbalance grows

### Step 2: Enable Dynamic Chunking

After finding the best fixed chunk size, set:

- `--chunked-prefill-size` to roughly **2x - 3x** of that fixed optimum
- `--enable-dynamic-chunking`

Then sweep:

- `SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR=0.65`
- `0.75`
- `0.85`

Good starting examples:

- fixed optimum `4096` -> dynamic initial size `8192` or `12288`
- fixed optimum `6144` -> dynamic initial size `12288` or `18432`

### Step 3: Check PP Stage Balance

If one stage is consistently slower than the other, set:

```bash
export SGLANG_PP_LAYER_PARTITION=<stage0_layers>,<stage1_layers>
```

Use it when:

- the model layers are not evenly divisible
- one stage consistently dominates prefill time
- chunk tuning stops helping

Rule of thumb:

- Put the slightly **larger partition on the later PP rank** if it reduces downstream idle time

### Step 4: Separate Prefill and Decode for Online Traffic

If you care about **online TTFT** instead of offline one-batch latency:

- use **PD disaggregation**
- benchmark prefill separately from decode

This matters because decode traffic can hide the real prefill bottleneck and destroy TTFT gains that look good in isolated profiling.

## Recommended Launch Template

### Node 0

```bash
python3 -m sglang.launch_server \
  --model-path <KIMI_MODEL_PATH> \
  --trust-remote-code \
  --nnodes 2 \
  --node-rank 0 \
  --dist-init-addr <MASTER_IP:PORT> \
  --tp-size 8 \
  --pp-size 2 \
  --host 0.0.0.0 \
  --port 30000 \
  --mem-fraction-static 0.8 \
  --max-running-requests 128 \
  --chunked-prefill-size 4096
```

### Node 1

```bash
python3 -m sglang.launch_server \
  --model-path <KIMI_MODEL_PATH> \
  --trust-remote-code \
  --nnodes 2 \
  --node-rank 1 \
  --dist-init-addr <MASTER_IP:PORT> \
  --tp-size 8 \
  --pp-size 2 \
  --host 0.0.0.0 \
  --port 30000 \
  --mem-fraction-static 0.8 \
  --max-running-requests 128 \
  --chunked-prefill-size 4096
```

### Dynamic Chunking Variant

```bash
export SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR=0.75

python3 -m sglang.launch_server \
  --model-path <KIMI_MODEL_PATH> \
  --trust-remote-code \
  --nnodes 2 \
  --node-rank <0_or_1> \
  --dist-init-addr <MASTER_IP:PORT> \
  --tp-size 8 \
  --pp-size 2 \
  --host 0.0.0.0 \
  --port 30000 \
  --mem-fraction-static 0.8 \
  --max-running-requests 128 \
  --chunked-prefill-size 12288 \
  --enable-dynamic-chunking
```

## Benchmark Plan

Use all three layers of measurement.

### A. Online TTFT

Use `bench_serving`:

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --base-url http://<SERVER_IP>:30000 \
  --dataset-name random \
  --random-input-len 8192 \
  --random-output-len 16 \
  --num-prompts 128 \
  --max-concurrency 8
```

Sweep input lengths:

- `8192`
- `16384`
- `32768`
- `65536`
- `131072`

Track:

- TTFT p50 / p95
- throughput
- max memory

### B. End-to-End Single Batch

Use `bench_one_batch_server`:

```bash
python3 -m sglang.bench_one_batch_server \
  --base-url http://<SERVER_IP>:30000 \
  --model-path <KIMI_MODEL_PATH> \
  --batch-size 1 \
  --input-len 32768 \
  --output-len 16
```

This is the cleanest way to compare fixed vs dynamic chunking for TTFT-oriented serving behavior.

### C. Low-Level Profiling

Use `bench_one_batch` only when you want to confirm where the time moved:

- metadata path
- attention backend
- PP bubble

## How to Interpret Results

If fixed chunk tuning gives the biggest win and this PR only adds a small extra gain, that is expected.

Use this decision tree:

1. **If TTFT improves a lot when chunk size changes:** your bottleneck is pipeline bubble / chunk planning
2. **If TTFT barely changes with chunk size but scales poorly with concurrency:** your bottleneck is scheduler/control-plane overhead
3. **If one PP stage is consistently slower:** fix layer partitioning next
4. **If online TTFT is much worse than one-batch latency:** prefill is being interrupted by decode traffic, so move to PD disaggregation

## Practical Optimization Roadmap

Use this order:

1. Land shared prefill metadata optimizations (this PR)
2. Tune fixed chunk size for `TP=8, PP=2`
3. Turn on dynamic chunking
4. Balance PP stages with `SGLANG_PP_LAYER_PARTITION`
5. If serving online traffic, move to PD disaggregation
6. Only then invest in deeper PP control-plane optimization

## What Not to Do First

Avoid these until the basic tuning loop is done:

- starting from `TP=16, PP=1`
- over-optimizing decode when prefill TTFT is the target
- assuming the shared runtime patch alone should deliver double-digit gains
- tuning dynamic chunking before you know the best fixed chunk size

## Related Docs

- [Pipeline Parallelism for Long Context](../advanced_features/pipeline_parallelism.md)
- [Benchmark and Profiling](../developer_guide/benchmark_and_profiling.md)
- [PD Disaggregation](../advanced_features/pd_disaggregation.md)
