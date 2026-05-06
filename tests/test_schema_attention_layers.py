import pathlib
import unittest

import yaml


class SchemaAttentionLayerTests(unittest.TestCase):
    def test_schema_includes_all_attention_layer_names(self):
        schema_path = pathlib.Path("configs/schema.yaml")
        with schema_path.open("r", encoding="utf-8") as handle:
            schema = yaml.safe_load(handle)

        enum_values = (
            schema["properties"]["model"]["properties"]["layer_pattern"]["items"]["enum"]
        )

        expected = {
            "retnet",
            "retnet_attn",
            "mamba",
            "ode",
            "titan_attn",
            "standard_attn",
            "sigmoid_attn",
            "sparse_transformer_attn",
            "longformer_attn",
            "bigbird_attn",
            "sparsek_attn",
            "nsa_attn",
            "sparge_attn",
            "fasa_attn",
            "gla_attn",
            "deltanet_attn",
            "gated_deltanet_attn",
            "hgrn2_attn",
            "fox_attn",
            "gated_softmax_attn",
            "engram_attn",
        }

        self.assertTrue(expected.issubset(set(enum_values)))

    def test_schema_includes_mixture_of_depths_fields(self):
        schema_path = pathlib.Path("configs/schema.yaml")
        with schema_path.open("r", encoding="utf-8") as handle:
            schema = yaml.safe_load(handle)

        model_properties = schema["properties"]["model"]["properties"]

        for field_name in [
            "use_mixture_of_depths",
            "mixture_of_depths_capacity_ratio",
            "mixture_of_depths_router_aux_loss_weight",
        ]:
            self.assertIn(field_name, model_properties)


if __name__ == "__main__":
    unittest.main()
