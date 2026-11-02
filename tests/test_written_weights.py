from __future__ import annotations

import io
import json
import shutil
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import numpy as np
from tinygrad import Tensor, nn

import evaluate_written_weights as evaluator
import experiment_config
import written_weights
from experiment_config import (
    MODEL_CONFIG,
    MODULUS,
    SOURCE_HASHES,
    digest,
    verify_sources,
)

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "artifacts/v1"
RECEIPT = ROOT / "experiments/v1-freeze.json"
_RESTORED: tempfile.TemporaryDirectory[str] | None = None


def setUpModule() -> None:
    global CANDIDATE, _RESTORED
    if not CANDIDATE.exists():
        # Fresh checkouts reconstruct only the exact bytes bound by the original receipt.
        _RESTORED = tempfile.TemporaryDirectory()
        CANDIDATE = Path(_RESTORED.name) / "v1"
        regenerated_receipt = Path(_RESTORED.name) / "receipt.json"
        written_weights.freeze_candidate(CANDIDATE, regenerated_receipt)
        if json.loads(regenerated_receipt.read_text()) != json.loads(
            RECEIPT.read_text()
        ):
            raise AssertionError(
                "reconstruction differs from the original frozen candidate"
            )


def tearDownModule() -> None:
    if _RESTORED is not None:
        _RESTORED.cleanup()


class InferenceTests(unittest.TestCase):
    """Exhaust the finite domain rather than sampling operand pairs."""

    def setUp(self) -> None:
        self.training = Tensor.training
        Tensor.training = False

    def tearDown(self) -> None:
        Tensor.training = self.training

    def test_frozen_checkpoint_covers_complete_domain_with_positive_margin(
        self,
    ) -> None:
        result = evaluator.evaluate(evaluator.load_candidate(CANDIDATE, RECEIPT))
        self.assertEqual(result.total, MODULUS**2)
        self.assertEqual(result.correct, result.total)
        self.assertEqual(result.mismatches, [])
        self.assertTrue(result.finite_logits)
        assert result.minimum_margin is not None
        self.assertGreater(result.minimum_margin, 0)
        self.assertEqual(result.unique_predictions, list(range(MODULUS)))

    def test_ablation_is_constant_zero_and_preserves_checkpoint(self) -> None:
        checkpoint = CANDIDATE / "model.safetensors"
        before = digest(checkpoint)
        result = evaluator.evaluate(
            evaluator.load_candidate(CANDIDATE, RECEIPT, ablate=True)
        )
        self.assertEqual(digest(checkpoint), before)
        self.assertTrue(result.finite_logits)
        self.assertEqual(result.unique_predictions, [0])
        self.assertEqual(result.correct, MODULUS)
        self.assertEqual(result.total, MODULUS**2)
        for mismatch in result.mismatches:
            self.assertEqual(mismatch.expected, (mismatch.a + mismatch.b) % MODULUS)
            self.assertNotEqual(mismatch.expected, mismatch.predicted)

    def test_random_seed_zero_control_repeats(self) -> None:
        Tensor.manual_seed(0)
        first = evaluator.evaluate(evaluator.Transformer(**MODEL_CONFIG))
        Tensor.manual_seed(0)
        second = evaluator.evaluate(evaluator.Transformer(**MODEL_CONFIG))
        self.assertEqual(first, second)
        self.assertTrue(first.finite_logits)

    def test_evaluation_rejects_training_mode(self) -> None:
        Tensor.training = True
        with self.assertRaisesRegex(ValueError, "inference mode"):
            evaluator.evaluate(evaluator.Transformer(**MODEL_CONFIG))

    def test_nonfinite_output_is_reported_without_a_numeric_margin(self) -> None:
        model = evaluator.load_candidate(CANDIDATE, RECEIPT)
        model.final = Tensor.full(model.final.shape, float("nan"))
        result = evaluator.evaluate(model)
        self.assertFalse(result.finite_logits)
        self.assertIsNone(result.minimum_margin)


class ArtifactTests(unittest.TestCase):
    def test_constructor_covers_all_parameters_and_is_seed_independent(self) -> None:
        Tensor.manual_seed(1)
        model = evaluator.Transformer(**MODEL_CONFIG)
        expected = nn.state.get_state_dict(model)
        first = written_weights.construct_parameters()
        Tensor.manual_seed(987654)
        second = written_weights.construct_parameters()
        self.assertEqual(first.keys(), expected.keys())
        self.assertEqual(first.keys(), second.keys())
        checkpoint = nn.state.safe_load(str(CANDIDATE / "model.safetensors"))
        for name, value in first.items():
            self.assertEqual(value.shape, expected[name].shape)
            self.assertEqual(value.dtype, np.float32)
            self.assertTrue(np.isfinite(value).all())
            np.testing.assert_array_equal(value, second[name])
            np.testing.assert_array_equal(value, checkpoint[name].numpy())

    def test_freeze_roundtrip_reproduces_original_artifact_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            candidate, receipt = base / "v1", base / "receipt.json"
            written_weights.freeze_candidate(candidate, receipt)
            self.assertEqual(
                json.loads(receipt.read_text()), json.loads(RECEIPT.read_text())
            )
            # Structural roundtrip only; predictions are tested on the frozen original.
            restored = evaluator.load_candidate(candidate, receipt)
            original = nn.state.safe_load(str(CANDIDATE / "model.safetensors"))
            for key, tensor in nn.state.get_state_dict(restored).items():
                np.testing.assert_array_equal(tensor.numpy(), original[key].numpy())

    def test_freeze_refuses_existing_output_or_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output, receipt = (
                Path(temporary) / "candidate",
                Path(temporary) / "receipt.json",
            )
            output.mkdir()
            with self.assertRaises(FileExistsError):
                written_weights.freeze_candidate(output, receipt)
            output.rmdir()
            receipt.write_text("preserve")
            with self.assertRaises(FileExistsError):
                written_weights.freeze_candidate(output, receipt)
            self.assertEqual(receipt.read_text(), "preserve")
            self.assertFalse(output.exists())

    def _copy_candidate(self) -> tuple[Path, Path]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        base = Path(temporary.name)
        candidate = base / "candidate"
        shutil.copytree(CANDIDATE, candidate)
        receipt = base / "receipt.json"
        shutil.copyfile(RECEIPT, receipt)
        return candidate, receipt

    def _rewrite_receipt(self, candidate: Path, receipt: Path) -> None:
        receipt.write_text(
            json.dumps(
                {
                    "manifest_sha256": digest(candidate / "manifest.json"),
                    "checkpoint_sha256": digest(candidate / "model.safetensors"),
                }
            )
        )

    def test_integrity_corruption_rejected(self) -> None:
        for filename in ("model.safetensors", "manifest.json"):
            with self.subTest(filename=filename):
                candidate, receipt = self._copy_candidate()
                path = candidate / filename
                path.write_bytes(path.read_bytes() + b"tamper")
                with self.assertRaisesRegex(ValueError, "digest"):
                    evaluator.load_candidate(candidate, receipt)

    def test_invalid_manifest_rejected_even_with_matching_receipt(self) -> None:
        changes = (
            {"schema_version": 2},
            {"schema_version": True},
            {"config": {**MODEL_CONFIG, "layers": 9}},
            {"upstream_commit": "0" * 40},
            {"source_sha256": {**SOURCE_HASHES, "transf.py": "0" * 64}},
            {"compiler_sha256": "0" * 64},
            {"config_source_sha256": "0" * 64},
            {"checkpoint_sha256": "0" * 64},
            {"attempt": 1},
            {"attempt": ""},
        )
        for change in changes:
            with self.subTest(change=change):
                candidate, receipt = self._copy_candidate()
                path = candidate / "manifest.json"
                manifest = json.loads(path.read_text())
                manifest.update(change)
                path.write_text(json.dumps(manifest))
                self._rewrite_receipt(candidate, receipt)
                with self.assertRaisesRegex(ValueError, "unsupported manifest"):
                    evaluator.load_candidate(candidate, receipt)

    def test_invalid_checkpoint_rejected_even_with_matching_receipt(self) -> None:
        def remove_parameter(state: dict[str, Tensor]) -> None:
            del state["final"]

        def wrong_shape(state: dict[str, Tensor]) -> None:
            state["final"] = Tensor.zeros(1)

        def nonfinite(state: dict[str, Tensor]) -> None:
            state["final"] = Tensor.full(state["final"].shape, float("nan"))

        mutations: tuple[Callable[[dict[str, Tensor]], None], ...] = (
            remove_parameter,
            wrong_shape,
            nonfinite,
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate.__name__):
                candidate, receipt = self._copy_candidate()
                checkpoint = candidate / "model.safetensors"
                state = nn.state.safe_load(str(checkpoint))
                mutate(state)
                altered = candidate / "altered.safetensors"
                nn.state.safe_save(state, str(altered))
                altered.replace(checkpoint)
                path = candidate / "manifest.json"
                manifest = json.loads(path.read_text())
                manifest["checkpoint_sha256"] = digest(checkpoint)
                path.write_text(json.dumps(manifest))
                self._rewrite_receipt(candidate, receipt)
                with self.assertRaisesRegex(
                    ValueError, "parameter keys|shape mismatch|nonfinite"
                ):
                    evaluator.load_candidate(candidate, receipt)

    def test_source_modification_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for filename in SOURCE_HASHES:
                shutil.copyfile(ROOT / filename, root / filename)
            (root / "grokking.py").write_text("altered")
            with (
                patch.object(experiment_config, "ROOT", root),
                self.assertRaisesRegex(
                    ValueError, "upstream source changed: grokking.py"
                ),
            ):
                verify_sources()

    def test_cli_generation_evaluation_and_existing_result_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            args = [
                "written_weights.py",
                "--output",
                str(base / "v1"),
                "--receipt",
                str(base / "receipt.json"),
            ]
            with patch("sys.argv", args), patch("sys.stdout", new_callable=io.StringIO):
                written_weights.main()
            self.assertEqual(
                json.loads((base / "receipt.json").read_text()),
                json.loads(RECEIPT.read_text()),
            )
            output = base / "evaluation.json"
            args = [
                "evaluate_written_weights.py",
                "--candidate",
                str(CANDIDATE),
                "--receipt",
                str(RECEIPT),
                "--output",
                str(output),
            ]
            with patch("sys.argv", args), patch("sys.stdout", new_callable=io.StringIO):
                evaluator.main()
            report = json.loads(output.read_text())
            self.assertEqual(report["candidate"]["correct"], MODULUS**2)
            self.assertTrue(report["ablation_control_passed"])
            before = output.read_bytes()
            with patch("sys.argv", args), self.assertRaises(FileExistsError):
                evaluator.main()
            self.assertEqual(output.read_bytes(), before)

    def test_cli_retains_a_failed_evaluation_for_inspection(self) -> None:
        # The real ablation supplies a failure fixture; only CLI orchestration is mocked.
        failed = evaluator.evaluate(
            evaluator.load_candidate(CANDIDATE, RECEIPT, ablate=True)
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "failed-evaluation.json"
            args = [
                "evaluate_written_weights.py",
                "--candidate",
                str(CANDIDATE),
                "--receipt",
                str(RECEIPT),
                "--output",
                str(output),
            ]
            with (
                patch("sys.argv", args),
                patch.object(evaluator, "evaluate", return_value=failed),
                patch("sys.stdout", new_callable=io.StringIO),
                self.assertRaises(SystemExit) as error,
            ):
                evaluator.main()
            self.assertEqual(error.exception.code, 1)
            report = json.loads(output.read_text())
            self.assertEqual(report["candidate"]["correct"], MODULUS)
            self.assertEqual(
                report["checkpoint_sha256"], digest(CANDIDATE / "model.safetensors")
            )

    def test_original_sources_are_unchanged(self) -> None:
        verify_sources()
        for filename, expected in SOURCE_HASHES.items():
            self.assertEqual(digest(ROOT / filename), expected)


if __name__ == "__main__":
    unittest.main()
