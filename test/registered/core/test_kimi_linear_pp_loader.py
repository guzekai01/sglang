import types
import unittest

import torch
from torch import nn

from sglang.srt.models.kimi_linear import KimiLinearForCausalLM
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=8, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=8, suite="stage-b-test-small-1-gpu-amd")


class _RemoteLayer:
    @property
    def self_attn(self):
        raise AssertionError("out-of-shard PP layer should not be accessed")


class _FakeParam:
    def __init__(self, name: str, calls: list[tuple]):
        self.name = name
        self.calls = calls

    def weight_loader(self, param, loaded_weight, *args, **kwargs):
        del param
        self.calls.append((self.name, loaded_weight.clone(), args, kwargs))


class TestKimiLinearPPLoader(CustomTestCase):
    def _build_stub_model(
        self,
        *,
        is_first_rank: bool = False,
        is_last_rank: bool = False,
        extra_param_names: tuple[str, ...] = (),
    ):
        model = KimiLinearForCausalLM.__new__(KimiLinearForCausalLM)
        nn.Module.__init__(model)

        local_fused_attn = types.SimpleNamespace(do_fuse_qkvbfg=True)
        local_full_attn = types.SimpleNamespace(
            qk_nope_head_dim=1,
            v_head_dim=1,
            kv_b_proj=types.SimpleNamespace(
                weight=torch.arange(4, dtype=torch.float32).reshape(2, 2),
                weight_scale=torch.tensor(0.5),
            ),
        )
        model.model = types.SimpleNamespace(
            start_layer=1,
            end_layer=3,
            layers=[
                _RemoteLayer(),
                types.SimpleNamespace(self_attn=local_fused_attn),
                types.SimpleNamespace(self_attn=local_full_attn),
            ],
        )
        model.config = types.SimpleNamespace(
            is_moe=False,
            is_linear_attn=False,
            full_attention_layer_ids=[0, 2],
            is_kda_layer=lambda layer_id: layer_id == 1,
        )
        model.pp_group = types.SimpleNamespace(
            is_first_rank=is_first_rank, is_last_rank=is_last_rank
        )

        calls = []
        local_param_name = "model.layers.1.self_attn.fused_qkvbfg_a_proj.weight"
        params = [(local_param_name, _FakeParam(local_param_name, calls))]
        for extra_name in extra_param_names:
            params.append((extra_name, _FakeParam(extra_name, calls)))
        model.named_parameters = lambda: params

        return model, calls, local_full_attn

    def test_load_weights_filters_non_local_pp_layers(self):
        model, calls, local_full_attn = self._build_stub_model()

        model.load_weights(
            [
                (
                    "model.layers.0.self_attn.b_proj.weight",
                    torch.ones((2, 2), dtype=torch.float32),
                ),
                (
                    "model.layers.1.self_attn.b_proj.weight",
                    torch.full((2, 2), 7.0, dtype=torch.float32),
                ),
                ("model.embed_tokens.weight", torch.ones((1,), dtype=torch.float32)),
                ("model.norm.weight", torch.ones((1,), dtype=torch.float32)),
                ("lm_head.weight", torch.ones((1,), dtype=torch.float32)),
            ]
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "model.layers.1.self_attn.fused_qkvbfg_a_proj.weight")
        self.assertEqual(calls[0][2], (3,))
        torch.testing.assert_close(
            calls[0][1], torch.full((2, 2), 7.0, dtype=torch.float32)
        )
        self.assertTrue(hasattr(local_full_attn, "w_kc"))
        self.assertTrue(hasattr(local_full_attn, "w_vc"))
        torch.testing.assert_close(local_full_attn.w_scale, torch.tensor(0.5))

    def test_first_rank_keeps_embed_tokens_loading(self):
        model, calls, _ = self._build_stub_model(
            is_first_rank=True,
            is_last_rank=False,
            extra_param_names=("model.embed_tokens.weight",),
        )

        model.load_weights(
            [("model.embed_tokens.weight", torch.full((1,), 3.0, dtype=torch.float32))]
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "model.embed_tokens.weight")
        torch.testing.assert_close(
            calls[0][1], torch.full((1,), 3.0, dtype=torch.float32)
        )

    def test_last_rank_keeps_norm_and_lm_head_loading(self):
        model, calls, _ = self._build_stub_model(
            is_first_rank=False,
            is_last_rank=True,
            extra_param_names=("model.norm.weight", "lm_head.weight"),
        )

        model.load_weights(
            [
                ("model.norm.weight", torch.full((1,), 5.0, dtype=torch.float32)),
                ("lm_head.weight", torch.full((1,), 9.0, dtype=torch.float32)),
            ]
        )

        self.assertEqual([call[0] for call in calls], ["model.norm.weight", "lm_head.weight"])
        torch.testing.assert_close(
            calls[0][1], torch.full((1,), 5.0, dtype=torch.float32)
        )
        torch.testing.assert_close(
            calls[1][1], torch.full((1,), 9.0, dtype=torch.float32)
        )


if __name__ == "__main__":
    unittest.main(verbosity=3)
