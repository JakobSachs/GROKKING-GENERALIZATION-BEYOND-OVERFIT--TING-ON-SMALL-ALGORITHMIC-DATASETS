from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from tinygrad import Tensor, nn

from prepare_blind_specimen import ROOT, digest, export_specimen
from transf import Transformer


class ExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.checkpoint = self.root / "private-task-name.safetensors"
        self.output = self.root / "neutral"
        self.receipt = self.root / "private-receipt.json"
        Tensor.manual_seed(0)
        model = Transformer(63, 3, 2, 128, 4, 512)
        self.state = nn.state.get_state_dict(model)
        nn.state.safe_save(self.state, str(self.checkpoint))

    def test_export_preserves_all_values_and_excludes_private_metadata(self) -> None:
        export_specimen(self.checkpoint, self.output, self.receipt)
        self.assertEqual(
            {p.name for p in self.output.iterdir()},
            {"specimen.npz", "architecture.json"},
        )
        with np.load(self.output / "specimen.npz", allow_pickle=False) as restored:
            self.assertEqual(set(restored.files), self.state.keys())
            for name, tensor in self.state.items():
                np.testing.assert_array_equal(restored[name], tensor.numpy())
        self.assertEqual(
            (self.output / "architecture.json").read_bytes(),
            (ROOT / "blind_architecture.json").read_bytes(),
        )
        receipt = json.loads(self.receipt.read_text())
        self.assertEqual(receipt["checkpoint_sha256"], digest(self.checkpoint))
        self.assertEqual(
            receipt["specimen_sha256"], digest(self.output / "specimen.npz")
        )
        self.assertEqual(receipt["parameters"], 413056)
        self.assertNotIn(
            "private-task-name", (self.output / "architecture.json").read_text()
        )

    def test_export_refuses_existing_output_or_receipt(self) -> None:
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            export_specimen(self.checkpoint, self.output, self.receipt)
        self.output.rmdir()
        self.receipt.write_text("preserve")
        with self.assertRaises(FileExistsError):
            export_specimen(self.checkpoint, self.output, self.receipt)
        self.assertEqual(self.receipt.read_text(), "preserve")
        self.assertFalse(self.output.exists())

    def test_export_rejects_incomplete_checkpoint(self) -> None:
        nn.state.safe_save(
            {"final": self.state["final"]}, str(self.root / "incomplete.safetensors")
        )
        with self.assertRaisesRegex(ValueError, "unexpected checkpoint architecture"):
            export_specimen(
                self.root / "incomplete.safetensors", self.output, self.receipt
            )
        self.assertFalse(self.output.exists())

    def test_export_rejects_nonfinite_weights(self) -> None:
        state = dict(self.state)
        state["final"] = Tensor.full(state["final"].shape, float("nan"))
        nn.state.safe_save(state, str(self.root / "nonfinite.safetensors"))
        with self.assertRaisesRegex(ValueError, "finite float32"):
            export_specimen(
                self.root / "nonfinite.safetensors", self.output, self.receipt
            )
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
