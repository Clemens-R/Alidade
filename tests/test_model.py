from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

import model as geo


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PARAMETERS = 4_964_188
EXPECTED_INFERENCE_VALUES = 4_993_242


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.checkpoint = torch.load(
            ROOT / "final_model.pt", map_location="cpu", weights_only=True
        )

    def test_parameter_and_state_counts(self):
        model = geo.AdapterMoE(imagenet_init=False)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        state_count = sum(value.numel() for value in self.checkpoint["model"].values())
        center_count = np.asarray(self.checkpoint["centers"]).size
        self.assertEqual(parameter_count, EXPECTED_PARAMETERS)
        self.assertEqual(state_count + center_count, EXPECTED_INFERENCE_VALUES)

    def test_checkpoint_loads(self):
        model = geo.AdapterMoE(imagenet_init=False)
        model.load_state_dict(self.checkpoint["model"])

    def test_sparse_router_uses_three_experts(self):
        logits = torch.randn(5, len(geo.COUNTRIES))
        gates = geo.sparse_gates(logits, top_experts=3)
        torch.testing.assert_close(gates.sum(1), torch.ones(5))
        self.assertTrue(torch.equal((gates > 0).sum(1), torch.full((5,), 3)))

    def test_decoder_returns_unit_vectors(self):
        model = geo.AdapterMoE(imagenet_init=False).eval()
        model.load_state_dict(self.checkpoint["model"])
        output = {
            "cell": torch.randn(2, geo.FINAL_MODEL_CONFIG["cells"]),
            "country": torch.randn(2, len(geo.COUNTRIES)),
            "residual": torch.randn(2, len(geo.COUNTRIES), 3),
        }
        centers = torch.as_tensor(self.checkpoint["centers"], dtype=torch.float32)
        prediction = model.decode(output, centers)
        self.assertEqual(tuple(prediction.shape), (2, 3))
        torch.testing.assert_close(
            torch.linalg.vector_norm(prediction, dim=1), torch.ones(2),
            rtol=1e-5, atol=1e-5,
        )


if __name__ == "__main__":
    unittest.main()
