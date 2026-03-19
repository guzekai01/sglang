import contextlib
import types
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

import sglang.srt.models.qwen3_5 as qwen3_5_model
from sglang.srt.managers.schedule_batch import ScheduleBatch
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


class _FakeSamplingInfo:
    def merge_batch(self, other):
        del other


class _FakeReq:
    def __init__(self, origin_input_ids, output_ids):
        self.origin_input_ids = origin_input_ids
        self.output_ids = output_ids
        self.fill_ids = origin_input_ids + output_ids
        self.extend_input_len = len(output_ids)

    def set_extend_input_len(self, extend_input_len):
        self.extend_input_len = extend_input_len


class TestPrefillMetadataAndPPRegressions(CustomTestCase):
    def test_mix_with_running_recomputes_extend_start_loc(self):
        prefill_batch = ScheduleBatch(
            reqs=[_FakeReq([1, 2], [3, 4, 5])],
            model_config=types.SimpleNamespace(is_encoder_decoder=False),
            sampling_info=_FakeSamplingInfo(),
            input_ids=torch.tensor([3, 4, 5], dtype=torch.int64),
            req_pool_indices=torch.tensor([0], dtype=torch.int64),
            seq_lens=torch.tensor([5], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([5], dtype=torch.int64),
            out_cache_loc=torch.tensor([10, 11, 12], dtype=torch.int64),
            orig_seq_lens=torch.tensor([5], dtype=torch.int32),
            seq_lens_sum=5,
            prefix_lens=[2],
            extend_lens=[3],
            extend_num_tokens=3,
            extend_start_loc=torch.tensor([0], dtype=torch.int32),
            extend_logprob_start_lens=[0],
            enable_overlap=False,
            return_logprob=False,
        )
        running_batch = ScheduleBatch(
            reqs=[_FakeReq([9, 8, 7, 6, 5], [4])],
            model_config=types.SimpleNamespace(is_encoder_decoder=False),
            sampling_info=_FakeSamplingInfo(),
            input_ids=torch.tensor([4], dtype=torch.int64),
            req_pool_indices=torch.tensor([1], dtype=torch.int64),
            seq_lens=torch.tensor([6], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([6], dtype=torch.int64),
            out_cache_loc=torch.tensor([20], dtype=torch.int64),
            orig_seq_lens=torch.tensor([6], dtype=torch.int32),
            seq_lens_sum=6,
            prefix_lens=[],
            extend_lens=[],
            extend_num_tokens=1,
            extend_start_loc=torch.tensor([0], dtype=torch.int32),
            extend_logprob_start_lens=[],
            enable_overlap=False,
            return_logprob=False,
        )

        prefill_batch.mix_with_running(running_batch)

        self.assertEqual(prefill_batch.extend_lens, [3, 1])
        torch.testing.assert_close(
            prefill_batch.extend_start_loc,
            torch.tensor([0, 3], dtype=torch.int32),
        )

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
