import contextlib
import types
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

import sglang.srt.models.qwen3_5 as qwen3_5_model
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors, compute_position
from sglang.srt.utils.common import compute_start_loc_from_lens
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=8, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=8, suite="stage-b-test-small-1-gpu-amd")


class _Recorder:
    def with_current_layer(self, _layer_idx: int):
        return contextlib.nullcontext()


class _FakeLayer(nn.Module):
    def __init__(self, delta: float):
        super().__init__()
        self.delta = delta
        self.calls = 0

    def forward(self, positions, hidden_states, residual, forward_batch):
        del positions, forward_batch
        self.calls += 1
        return hidden_states + self.delta, residual


class TestPrefillMetadataAndPPRegressions(CustomTestCase):
    def test_compute_position_reuses_precomputed_start_loc(self):
        extend_seq_lens = torch.tensor([4, 2, 5], dtype=torch.int32)
        extend_prefix_lens = torch.tensor([1, 0, 3], dtype=torch.int32)
        extend_start_loc = compute_start_loc_from_lens(extend_seq_lens)

        positions, returned_start_loc = compute_position(
            "torch_native",
            extend_prefix_lens,
            extend_seq_lens,
            int(extend_seq_lens.sum().item()),
            extend_start_loc,
        )

        expected_positions = torch.tensor(
            [1, 2, 3, 4, 0, 1, 3, 4, 5, 6, 7], dtype=torch.int64
        )
        torch.testing.assert_close(positions, expected_positions)
        self.assertEqual(returned_start_loc.data_ptr(), extend_start_loc.data_ptr())

    def test_qwen35_forward_only_runs_local_pp_layers(self):
        layers = nn.ModuleList(
            [_FakeLayer(10.0), _FakeLayer(1.0), _FakeLayer(2.0), _FakeLayer(20.0)]
        )
        pp_group = types.SimpleNamespace(
            rank_in_group=1,
            world_size=3,
            is_first_rank=False,
            is_last_rank=False,
        )
        config = types.SimpleNamespace(
            hidden_size=4,
            vocab_size=16,
            num_hidden_layers=4,
            layers_block_type=["attention"] * 4,
            rms_norm_eps=1e-6,
        )

        with (
            patch.object(qwen3_5_model, "get_pp_group", return_value=pp_group),
            patch.object(
                qwen3_5_model, "make_layers", return_value=(layers, 1, 3)
            ) as mock_make_layers,
            patch.object(
                qwen3_5_model,
                "get_global_expert_distribution_recorder",
                return_value=_Recorder(),
            ),
        ):
            model = qwen3_5_model.Qwen3_5ForCausalLM(config)
            output = model(
                input_ids=torch.empty(0, dtype=torch.int64),
                positions=torch.arange(3, dtype=torch.int64),
                forward_batch=object(),
                pp_proxy_tensors=PPProxyTensors(
                    {
                        "hidden_states": torch.zeros(3, 4),
                        "residual": None,
                    }
                ),
            )

        self.assertEqual(mock_make_layers.call_args.kwargs["pp_rank"], 1)
        self.assertEqual(mock_make_layers.call_args.kwargs["pp_size"], 3)
        self.assertEqual(mock_make_layers.call_args.kwargs["prefix"], "layers")
        self.assertEqual([layer.calls for layer in layers], [0, 1, 1, 0])
        torch.testing.assert_close(
            output["hidden_states"],
            torch.full((3, 4), 3.0),
        )


if __name__ == "__main__":
    unittest.main(verbosity=3)
