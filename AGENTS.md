# AGENTS.md — SGLang 人机协作与 Agent 约定

> **适用对象**：人类贡献者、Cursor Cloud Agent、Claude Code 等自动化工具。
> 本文件位于仓库根目录，[Cursor Cloud Agent](https://www.cursor.com/docs/cloud-agent/setup) 与 [Claude Code](https://docs.anthropic.com/en/docs/claude-code) 均会自动读取。若工具未加载，可用 `@AGENTS.md` 显式引用。

---

## 1. 项目结构速览

| 组件 | 路径 | 语言 | 说明 |
|------|------|------|------|
| SGLang Runtime (SRT) | `python/sglang/` | Python | LLM/VLM 推理引擎核心（调度、模型前向、PagedAttention 等） |
| SGLang Kernel | `sgl-kernel/` | C++/CUDA | 高性能 GPU 算子（MoE、Attention、GEMM、量化等） |
| SGLang Model Gateway | `sgl-model-gateway/` | Rust | 模型路由与负载均衡（gRPC/HTTP，OpenAI 兼容） |
| Diffusion 子系统 | `python/sglang/multimodal_gen/` | Python | 图像/视频扩散模型推理，架构独立于 SRT；详见其子目录 `CLAUDE.md` |

代码规范、测试约定、PR 流程等已有完整文档，勿重复造轮子：
- **贡献指南**：`docs/developer_guide/contribution_guide.md`
- **代码风格**：同上文件的 *Code style guidance* 小节
- **单元测试约定**：`test/registered/unit/README.md`
- **PR 合并流程**：`.github/MAINTAINER.md`
- **sgl-kernel 更新流程**：贡献指南中 *How to update sgl-kernel* 小节

---

## 2. 协作原则

- **单一事实源**：范围、结论、下一步以 **Issue / PR 描述** 为准；即时通讯（Slack 等）只做同步，不作最终记录。
- **工件驱动**：多 Agent 或多人不依赖实时群聊；协作靠 **同一分支、同一 PR、仓库内已写入的说明**。
- **人类裁决**：合并、发布、架构取舍由 **Maintainer / Codeowner** 决定（流程见 `.github/MAINTAINER.md`）；Agent 产出默认视为 **提案**。

---

## 3. 任务类型

每张工单只启用需要的类型，在 Issue / PR 描述中标明：

| 类型 | 职责 | 交付要点 |
|------|------|----------|
| **Implement** | 在约束内改代码 | 动机、影响面；CPU 侧可跑的测试/lint |
| **Verify** | 复现与量化 | 基线 vs 改后；环境信息；日志/指标摘要 |
| **Review** | 风险与可维护性 | 阻塞项 / 非阻塞项；可定位到文件或逻辑 |
| **Record** | 更新状态与文档 | Issue/PR 描述、必要 `docs/` 更新 |

可用 **专长标签** 标注关注域：`cuda-kernel`、`distributed`、`scheduler`、`python-frontend`、`diffusion` 等。

---

## 4. 工单最低信息

人类或 Agent 开任务时须写清：

- **一句话目标** + **非目标**（本轮明确不做的事）。
- **硬约束**：API 兼容、支持的后端/硬件、不得修改的公共接口等。
- **验收**：可重复的 **命令 + 期望结果**；无法量化时写 **等价验收**（如指定测例全绿、无行为回归）。
- **回滚**：revert 步骤，或 feature flag / 配置开关说明。

---

## 5. Git / PR 约定

- **小步、单主题**；大块改动拆分子 PR，描述中注明依赖关系。
- **合并门禁**：CI 必过、Codeowner 审批。凡声称性能相关，须具备 **GPU Verify 结果摘要**。
- **热点模块**：多人可能同改一处时，先在 Issue 中 **声明意向** 避免并行冲突。
- **sgl-kernel 更新**：必须分多个 PR 完成（代码 → 版本号 → 调用方），不可单 PR 完成，详见贡献指南。

---

## 6. 审查关注点

- **正确性**：边界条件、数值稳定性、分布式一致性、错误路径。
- **爆炸半径**：是否影响默认路径、多后端、多 GPU 拓扑。
- **KISS**：能否用更小改动达成目标；避免难测的过度抽象。
- **性能声明**：须 **有实测数据**；禁止仅凭理论合入「更快」类表述。
- **可观测性**：日志/指标/断言是否足以定位回归。

---

## 7. 节拍与停损

- **停损**：同一主题 Implement → Verify → Review 建议 **≤ 3 轮**；仍不达标则 **记录阻塞原因** 并交人类排期，不无限迭代。
- **并行**：仅当文件域或子目标明显分离时并行多 PR；同一热点 **串行** 或指定 integrator 负责合并顺序。

---

## 8. 无 GPU 环境：人机分工

Cloud Agent 典型运行在 **CPU 虚拟机** 上，不能作为 GPU / 多卡 / 端到端推理性能验证的权威环境。

| 环节 | Cloud Agent（无 GPU） | 人 / GPU 机器 / GPU CI |
|------|------------------------|--------------------------|
| 改代码、CPU 可跑测试、lint、静态检查 | ✅ 可做 | 可选抽查 |
| 性能、kernel、多卡、真实推理负载 | ❌ 不宣称「已 GPU 验证」 | ✅ 权威 Verify |
| 合入决策 | 提案 | Maintainer 依据 Verify 与 Review |

**规则**：与性能相关的改动，**缺少 GPU Verify 摘要不得合并**。

---

## 9. Cursor Cloud 专用指令

### 9.1 安装与依赖

从仓库根目录执行：

```bash
pip install --upgrade pip
pip install uv pre-commit
UV_SYSTEM_PYTHON=true uv pip install -e "python[dev]" --index-strategy unsafe-best-match --prerelease allow
```

> `pyproject.toml` 在 `python/` 目录下，因此安装路径是 `"python[dev]"`，不是 `".[dev]"`。

### 9.2 开发命令

| 操作 | 命令 |
|------|------|
| **Lint（全量）** | `pre-commit run --all-files` |
| **单元测试（全量）** | `python3 -m pytest test/registered/unit/ -v` |
| **单元测试（CPU 安全子集）** | `python3 -m pytest test/registered/unit/function_call/ test/registered/unit/layers/ test/registered/unit/observability/ test/registered/unit/parser/ test/registered/unit/server_args/ test/registered/unit/utils/ --ignore=test/registered/unit/utils/test_profile_merger.py -v` |
| **启动推理服务（需 GPU）** | `sglang serve --model-path <model>` |
| **E2E 测试（需 GPU）** | 见 `docs/developer_guide/contribution_guide.md` |

### 9.3 已知坑点

- **`compressed_tensors` 在 import 时创建 CUDA tensor**：任何间接 import `sglang.srt.managers.schedule_batch` 的测试文件在无 GPU 时会 collection error。上方 "CPU 安全子集" 命令已排除这些文件。
- **`core.hooksPath` 冲突**：若 `pre-commit install` 报错，先执行 `git config --unset-all core.hooksPath`。
- **`pytest.ini` 位于 `test/` 目录**：从仓库根运行 pytest 并指向 `test/registered/unit/` 即可，无需 `cd test`。
- **torch CUDA 版本**：PyPI `torch==2.9.1` 捆绑 cu128；CI 用 cu129 index。CI 脚本 `scripts/ci/cuda/ci_install_dependency.sh` 会自动修正 torchaudio/torchvision 的 CUDA 版本匹配。CPU 环境无影响。

---

## 10. 版本

- **文档版本**：v1
- **修订**：由核心维护者更新并通告团队。
