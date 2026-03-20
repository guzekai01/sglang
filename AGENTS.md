# AGENTS.md

## Cursor Cloud specific instructions

### Overview

SGLang is a high-performance LLM/multimodal serving framework. The monorepo contains three main components:

| Component | Path | Language |
|---|---|---|
| SGLang Runtime (SRT) | `python/sglang/` | Python |
| SGLang Kernel | `sgl-kernel/` | C++/CUDA |
| SGLang Model Gateway | `sgl-model-gateway/` | Rust |

### Development commands

- **Lint**: `pre-commit run --all-files` (21 hooks: isort, ruff, black, codespell, clang-format, etc.)
- **Unit tests (CPU-safe)**: `python3 -m pytest test/registered/unit/ -v` — note that tests importing `sglang.srt.managers.schedule_batch` or `compressed_tensors` will fail without GPU due to CUDA tensor initialization at import time.
- **CPU-safe unit test subset**: `python3 -m pytest test/registered/unit/function_call/ test/registered/unit/layers/ test/registered/unit/observability/ test/registered/unit/parser/ test/registered/unit/server_args/ test/registered/unit/utils/ --ignore=test/registered/unit/utils/test_profile_merger.py -v`
- **E2E tests**: Require a running SGLang server with GPU — see `docs/developer_guide/contribution_guide.md`.
- **Launch server**: `python3 -m sglang.launch_server --model-path <model>` or `sglang serve --model-path <model>` (requires GPU + HF model access).

### Key gotchas

- **No GPU = limited testing.** The `compressed_tensors` package creates CUDA tensors at module import time, preventing any test that transitively imports `sglang.srt.managers.schedule_batch` from collecting on CPU-only machines.
- **Editable install**: Use `pip install -e "python[dev]"` from the repo root. The `pyproject.toml` is in `python/`, not the repo root.
- **Pre-commit hooks**: Run `git config --unset-all core.hooksPath` before `pre-commit install` if `core.hooksPath` is set globally.
- **pytest config**: The `pytest.ini` lives in `test/` (not repo root), so run pytest from `/workspace` and point to `test/registered/unit/`.
- **torch version**: The package pins `torch==2.9.1`. PyPI torch bundles cu128, but CI uses cu129 index. The CI script (`scripts/ci/cuda/ci_install_dependency.sh`) handles the CUDA version mismatch by reinstalling torchaudio/torchvision from the matching index.
