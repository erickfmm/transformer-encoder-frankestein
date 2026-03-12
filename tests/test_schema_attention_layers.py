import pathlib
import unittest

import yaml


class SchemaAttentionLayerTests(unittest.TestCase):
    def test_schema_includes_all_attention_layer_names(self):
        schema_path = pathlib.Path("src/training/configs/schema.yaml")
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
        }

        self.assertTrue(expected.issubset(set(enum_values)))


if __name__ == "__main__":
    unittest.main()
