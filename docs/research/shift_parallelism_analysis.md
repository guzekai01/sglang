# Shift Parallelism 论文总结与 SGLang 可行性分析

> 论文: *Shift Parallelism: Low-Latency, High-Throughput LLM Inference for Dynamic Workloads*
> 作者: Mert Hidayetoglu, Aurick Qiao, Michael Wyatt, Jeff Rasley, Yuxiong He, Samyam Rajbhandari (Snowflake AI Research)
> 链接: https://arxiv.org/abs/2509.16495

---

## 1. 核心问题

LLM 推理面临 **延迟 vs 吞吐量** 的根本矛盾：

| 并行策略 | TTFT | TPOT | 吞吐量 | 通信模式 |
|---------|------|------|--------|---------|
| **TP (Tensor Parallelism)** | 快 | 快 | 低（all-reduce 开销大） | 每层 2 次 all-reduce |
| **DP (Data Parallelism)** | 慢（无法加速单请求） | 慢 | 高（无通信） | 无 |
| **SP (Sequence Parallelism / Ulysses)** | 快 | 慢（小 batch 负载不均） | 高（all-to-all 开销低） | 每层 2 次 all-to-all |

关键洞察：**TP 和 DP 的 KV cache 布局不兼容**，无法在二者之间动态切换。但 **SP 和 TP 的 KV cache 布局是不变的（KV cache invariance）**，因此可以在它们之间无缝切换。

---

## 2. Shift Parallelism 的核心设计

### 2.1 基本思路

维护两套并行配置，根据当前 batch 大小动态选择：

- **Base 配置** (SP 或 SP×TP)：用于大 batch — 优化 TTFT 和吞吐量
- **Shift 配置** (全 TP)：用于小 batch — 优化 TPOT（decode 延迟）

```
if batch_tokens > threshold:
    use SP (or SP×TP)   # 大 batch，高吞吐
else:
    use full TP          # 小 batch，低延迟
```

### 2.2 KV Cache 不变性

SP 和 TP 的核心共同点：都使用 **Head Parallelism**，即将 attention heads 均匀分配到 GPU 上。

- TP=2 时：GPU0 持有 head 0,1；GPU1 持有 head 2,3
- SP=2 时：同样 GPU0 持有 head 0,1；GPU1 持有 head 2,3（但 sequence 被分片）

因此 KV cache 的内存布局完全一致，切换时无需迁移 KV cache。

当使用混合配置 (SP×TP) 时，head 的顺序可能会交错（例如 SP=3, TP=2 时顺序变为 0,2,4,1,3,5），需要通过 process-to-data mapping 保证 shift 配置时保持一致的 head 顺序。

### 2.3 SP 在推理中的适配

论文解决了 SP（原为训练设计的 Ulysses）用于推理的几个关键问题：

1. **GQA 支持**：将 `3×h` 替换为 `h + 2×h_kv`，通过 all-to-all 实现 KV cache replication
2. **KV Cache 复制**：当 SP 度 > KV head 数时，通过 all-to-all 通信复制 KV heads
3. **小 batch 负载均衡**：对 batch padding 到 SP 度的倍数（代价是 decode 时有冗余 token）
4. **通信融合**：将 Q、K、V 的 all-to-all 融合为单次通信

### 2.4 权重管理

两种方案：
- **On-the-fly slicing**：运行时切片权重矩阵（无额外内存，但有运行时开销）
- **Weight replication**：预加载两套权重（SP 和 TP 各一份，共享 attention 权重）

论文选择方案 2（weight replication），额外内存开销约 **15-25%**（MLP 权重需要复制，attention 权重因 KV cache invariance 共享）。

### 2.5 通信复杂度

| | 通信操作 | 通信量 |
|---|---------|-------|
| TP | 每层 2 次 all-reduce | O(n × d)，随 n 线性增长 |
| SP | 每层 2 次 all-to-all | O(n × d / P)，不随 P 增长 |

SP 的通信量不随并行度增长，这是其在高吞吐场景下的核心优势。

---

## 3. 实验结果

- **交互式负载（低流量）**：Shift Parallelism 比纯 TP 快 **1.51×**
- **批处理负载（高流量）**：吞吐量比纯 TP 高 **50%**
- **动态负载**：在真实生产 trace 中，Shift Parallelism 全面优于 TP-only 和 DP-only
- **测试模型**：Llama-3.3-70B, Llama-17B-16E (MoE), Qwen-30B-A3B
- **测试环境**：8×H100 / 8×H200 节点

---

## 4. SGLang 适配可行性分析

### 4.1 SGLang 现有架构概览

SGLang 的并行推理架构涉及以下关键组件：

| 组件 | 文件路径 | 当前功能 |
|------|---------|---------|
| 调度器 | `python/sglang/srt/managers/scheduler.py` | 管理 prefill/decode batch 调度 |
| DP Attention | `python/sglang/srt/layers/dp_attention.py` | 已实现 DP Attention（all-gather/all-reduce gather/scatter） |
| 通信层 | `python/sglang/srt/layers/communicator.py` | TP all-reduce, DP gather/scatter |
| 模型执行器 | `python/sglang/srt/model_executor/model_runner.py` | forward dispatch, CUDA graph |
| Forward Batch | `python/sglang/srt/model_executor/forward_batch_info.py` | ForwardMode: EXTEND/DECODE/MIXED/IDLE |
| KV Cache | `python/sglang/srt/mem_cache/memory_pool.py` | ReqToTokenPool, KVCache |
| 并行状态 | `python/sglang/srt/distributed/parallel_state.py` | TP/DP group 管理 |
| Attention 层 | `python/sglang/srt/layers/radix_attention.py` | RadixAttention |

### 4.2 技术可行性评估

#### 已有基础（有利因素）

1. **DP Attention 已实现**：SGLang 已有成熟的 DP Attention 机制（`dp_attention.py`），包括 `dp_gather`/`dp_scatter` 等通信原语，说明 SGLang 已经有多种并行模式共存的架构基础。

2. **Communicator 抽象良好**：`communicator.py` 已抽象出通信模式（all-reduce, reduce-scatter, all-gather），添加 all-to-all 通信模式相对直接。

3. **Forward Mode 已支持多模式**：`ForwardMode` 枚举已有 EXTEND/DECODE/MIXED/IDLE 等模式，可扩展 SP 相关模式。

4. **调度器灵活**：`scheduler.py` 的 `get_next_batch_to_run()` 已能根据不同条件选择 batch，可以在此层面添加 SP/TP 切换逻辑。

5. **CUDA Graph 支持**：SGLang 对 decode（小 batch）已有 CUDA graph 支持，shift 到 TP 模式时可以复用。

#### 实现挑战

1. **All-to-All 通信集成**：
   - SGLang 当前主要使用 all-reduce（TP）和 all-gather/reduce-scatter（DP Attention）
   - SP 需要 all-to-all 通信，需要在 `communicator.py` 和分布式通信层中添加支持
   - 需要确保 NCCL all-to-all 的性能和兼容性

2. **双权重加载**：
   - Shift Parallelism 需要同时维护 SP 和 TP 两套 MLP 权重（attention 权重共享）
   - SGLang 当前的权重加载路径（`model_loader/weight_utils.py`）需要扩展
   - 额外的 ~20% GPU 内存开销可能影响 KV cache 容量

3. **KV Cache Head 排序**：
   - 混合 (SP, TP) 配置时，head 顺序可能交错
   - 需要修改 attention 层以支持非连续的 head 映射
   - RadixAttention 和底层 FlashInfer/FA3 kernel 可能需要适配

4. **CUDA Graph 兼容性**：
   - SP 和 TP 模式切换意味着需要两套不同的 CUDA graph
   - `CudaGraphRunner` 需要感知当前并行模式
   - Graph capture 时需要区分 SP/TP 通信模式

5. **Batch 调度逻辑**：
   - 调度器需要感知当前 batch 大小并决定使用 SP 还是 TP
   - Chunked prefill 和 SP 的交互需要仔细设计
   - 多个 SP rank 之间的 batch 信息同步

6. **与现有 DP Attention 的关系**：
   - SGLang 已有 DP Attention，其与 Shift Parallelism 的 SP 模式在概念上有重叠
   - DP Attention 在 MLP 层使用全 TP all-reduce，在 attention 层各 DP rank 独立计算
   - SP 在 attention 层使用 all-to-all，在 MLP 层使用 TP all-reduce
   - 两种方案的核心区别在于 **attention 层是否跨 GPU 并行化单个请求**

### 4.3 实现路径建议

如果决定在 SGLang 中实现 Shift Parallelism，建议分阶段进行：

#### 阶段 1：SP for Inference（基础）
- 在 `communicator.py` 中添加 SP 通信原语（all-to-all for QKV, all-to-all for attn output）
- 修改 attention 层支持 SP 模式下的 head 并行
- 实现 GQA 下的 KV cache replication
- 添加小 batch padding 逻辑

#### 阶段 2：Shift 机制
- 实现双权重加载（SP weights + TP weights，attention 权重共享）
- 在 `model_runner.py` 中添加 SP/TP forward path 切换
- 修改 `scheduler.py` 支持基于 batch size 的并行模式选择
- 实现 KV cache head 重排序

#### 阶段 3：性能优化
- SP/TP 双模式 CUDA Graph capture
- 通信与计算 overlap
- 阈值自动调优
- 与现有 DP Attention 的协调优化

### 4.4 与 DP Attention 的对比

| 特性 | SGLang DP Attention | Shift Parallelism (SP) |
|------|-------------------|----------------------|
| Attention 并行方式 | 各 DP rank 独立处理不同请求 | all-to-all 跨序列并行 |
| MLP 并行方式 | 全 TP all-reduce | 全 TP all-reduce |
| 单请求加速 | 否 | 是（prefill 可跨 GPU） |
| KV Cache 布局 | 每个 DP rank 独立 | 与 TP 共享（KV cache invariance） |
| 动态切换 | 不支持切 TP | 可切换到 TP |
| 适用场景 | 高吞吐批处理 | 动态混合负载 |

**关键区别**：DP Attention 本质上是多副本处理不同请求，而 SP 是对同一请求的序列维度做并行。SP 可以降低单个长 prompt 的 TTFT，而 DP Attention 不能。

### 4.5 总结与建议

#### 可行性结论：技术上可行，但实现复杂度较高

**核心价值**：
- 对于动态流量场景（低流量时要低延迟，高流量时要高吞吐），Shift Parallelism 提供了唯一能同时兼顾的方案
- 1.5× 的延迟改善和 50% 的吞吐量提升是有实际商业价值的

**主要挑战**：
- 需要在 SGLang 的通信层添加 all-to-all 支持
- 双权重占用额外 ~20% GPU 内存
- CUDA Graph 需要支持两套并行模式
- 与现有 DP Attention 和 speculative decoding 等特性的交互需要仔细设计

**建议**：
1. 可以先在 SGLang 中实现 **纯 SP 推理模式**（不含 Shift），验证 all-to-all 通信的性能和 GQA 兼容性
2. 评估 SP 模式在 SGLang 已有的 benchmark 场景下的表现
3. 如果 SP 单独带来的吞吐量提升显著，再考虑实现完整的 Shift 机制
4. 需要注意该论文的作者团队已在 vLLM 上实现并开源了 Shift Parallelism，可参考其实现细节

---

## 5. 参考资料

- 论文 PDF: https://arxiv.org/pdf/2509.16495
- Ulysses SP 原始论文: Jacobs et al., 2023
- GQA: Ainslie et al., 2023
- 该技术已在 vLLM 中通过 plugin 形式集成部署
