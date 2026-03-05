import unittest

import torch

from src.model.attention import HoPE, ODEAttentionBlock, RoPE, SigmoidAttention, StandardAttention, TitanAttention
from src.model.tormented_bert_frankestein import TormentedBertFrankenstein, UltraConfig


class AttentionRefactorTests(unittest.TestCase):
    def test_import_smoke(self):
        self.assertTrue(callable(TormentedBertFrankenstein))
        self.assertTrue(callable(UltraConfig))
        self.assertTrue(callable(TitanAttention))
        self.assertTrue(callable(StandardAttention))
        self.assertTrue(callable(SigmoidAttention))
        self.assertTrue(callable(ODEAttentionBlock))
        self.assertTrue(callable(HoPE))
        self.assertTrue(callable(RoPE))

    def test_default_forward_compat(self):
        config = UltraConfig(
            vocab_size=100,
            hidden_size=48,
            num_layers=2,
            num_loops=1,
            num_heads=6,
            retention_heads=6,
            num_experts=2,
            top_k_experts=1,
            dropout=0.0,
            layer_pattern=["titan_attn", "standard_attn"],
            ode_solver="rk4",
            ode_steps=1,
            use_bitnet=False,
            norm_type="layer_norm",
            use_factorized_embedding=False,
            factorized_embedding_dim=16,
            use_embedding_conv=False,
            embedding_conv_kernel=3,
            use_hope=True,
            use_moe=False,
            ffn_hidden_size=96,
            ffn_activation="gelu",
        )
        model = TormentedBertFrankenstein(config)
        x = torch.randint(0, config.vocab_size, (2, 8))
        y = model(x)
        self.assertEqual(y.shape, (2, 8, config.vocab_size))

    def test_legacy_use_hope_false_maps_to_rope(self):
        config = UltraConfig(
            vocab_size=100,
            hidden_size=48,
            num_layers=1,
            num_loops=1,
            num_heads=6,
            retention_heads=6,
            num_experts=2,
            top_k_experts=1,
            dropout=0.0,
            layer_pattern=["titan_attn"],
            ode_solver="rk4",
            ode_steps=1,
            use_bitnet=False,
            norm_type="layer_norm",
            use_factorized_embedding=False,
            factorized_embedding_dim=16,
            use_embedding_conv=False,
            embedding_conv_kernel=3,
            use_hope=False,
            positional_encoding=None,
            use_moe=False,
            ffn_hidden_size=96,
            ffn_activation="gelu",
        )
        attn = TitanAttention(config)
        self.assertIsInstance(attn.pos_encoder, RoPE)

    def test_positional_encoding_override(self):
        base_kwargs = dict(
            vocab_size=100,
            hidden_size=48,
            num_layers=1,
            num_loops=1,
            num_heads=6,
            retention_heads=6,
            num_experts=2,
            top_k_experts=1,
            dropout=0.0,
            layer_pattern=["titan_attn"],
            ode_solver="rk4",
            ode_steps=1,
            use_bitnet=False,
            norm_type="layer_norm",
            use_factorized_embedding=False,
            factorized_embedding_dim=16,
            use_embedding_conv=False,
            embedding_conv_kernel=3,
            use_hope=True,
            use_moe=False,
            ffn_hidden_size=96,
            ffn_activation="gelu",
        )
        cfg_hope = UltraConfig(**base_kwargs, positional_encoding="hope")
        cfg_rope = UltraConfig(**base_kwargs, positional_encoding="rope")

        self.assertIsInstance(TitanAttention(cfg_hope).pos_encoder, HoPE)
        self.assertIsInstance(TitanAttention(cfg_rope).pos_encoder, RoPE)

    def test_invalid_positional_encoding_raises(self):
        with self.assertRaisesRegex(ValueError, "positional_encoding"):
            UltraConfig(
                vocab_size=100,
                hidden_size=48,
                num_layers=1,
                num_loops=1,
                num_heads=6,
                retention_heads=6,
                num_experts=2,
                top_k_experts=1,
                dropout=0.0,
                layer_pattern=["titan_attn"],
                ode_solver="rk4",
                ode_steps=1,
                use_bitnet=False,
                norm_type="layer_norm",
                use_factorized_embedding=False,
                factorized_embedding_dim=16,
                use_embedding_conv=False,
                embedding_conv_kernel=3,
                use_hope=True,
                positional_encoding="invalid",
                use_moe=False,
                ffn_hidden_size=96,
                ffn_activation="gelu",
            )

    def test_layer_type_coverage(self):
        for layer_type in ["titan_attn", "standard_attn", "sigmoid_attn", "ode", "retnet"]:
            config = UltraConfig(
                vocab_size=100,
                hidden_size=48,
                num_layers=1,
                num_loops=1,
                num_heads=6,
                retention_heads=6,
                num_experts=2,
                top_k_experts=1,
                dropout=0.0,
                layer_pattern=[layer_type],
                ode_solver="rk4",
                ode_steps=1,
                use_bitnet=False,
                norm_type="layer_norm",
                use_factorized_embedding=False,
                factorized_embedding_dim=16,
                use_embedding_conv=False,
                embedding_conv_kernel=3,
                use_hope=True,
                use_moe=False,
                ffn_hidden_size=96,
                ffn_activation="gelu",
            )
            model = TormentedBertFrankenstein(config)
            x = torch.randint(0, config.vocab_size, (1, 6))
            y = model(x)
            self.assertEqual(y.shape, (1, 6, config.vocab_size))


if __name__ == "__main__":
    unittest.main()
