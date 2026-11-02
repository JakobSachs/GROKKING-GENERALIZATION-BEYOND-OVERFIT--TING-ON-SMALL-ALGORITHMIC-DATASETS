"""Evaluate a frozen checkpoint using only the original transformer forward pass."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from tinygrad import Tensor, nn

from experiment_config import (
    MODEL_CONFIG,
    MODULUS,
    ROOT,
    SOURCE_HASHES,
    UPSTREAM_COMMIT,
    digest,
    verify_sources,
)
from grokking import gen_data
from transf import Transformer


@dataclass(frozen=True)
class Mismatch:
    a: int
    b: int
    expected: int
    predicted: int


@dataclass(frozen=True)
class Evaluation:
    total: int
    correct: int
    accuracy: float
    finite_logits: bool
    minimum_margin: float | None
    unique_predictions: list[int]
    mismatches: list[Mismatch]


def load_candidate(
    candidate: Path, receipt_path: Path, *, ablate: bool = False
) -> Transformer:
    """Validate the freeze record before loading; never import or invoke the compiler."""
    verify_sources()
    receipt = json.loads(receipt_path.read_text())
    manifest_path = candidate / "manifest.json"
    checkpoint = candidate / "model.safetensors"
    checkpoint_hash = digest(checkpoint)
    if not isinstance(receipt, dict) or (
        receipt.get("manifest_sha256") != digest(manifest_path)
        or receipt.get("checkpoint_sha256") != checkpoint_hash
    ):
        raise ValueError("artifact digest does not match freeze receipt")
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict) or (
        type(manifest.get("schema_version")) is not int
        or manifest.get("schema_version") != 1
        or manifest.get("config") != MODEL_CONFIG
        or manifest.get("upstream_commit") != UPSTREAM_COMMIT
        or manifest.get("source_sha256") != SOURCE_HASHES
        or manifest.get("compiler_sha256") != digest(ROOT / "written_weights.py")
        or manifest.get("config_source_sha256") != digest(ROOT / "experiment_config.py")
        or manifest.get("checkpoint_sha256") != checkpoint_hash
        or not isinstance(manifest.get("attempt"), str)
        or not manifest["attempt"]
    ):
        raise ValueError("unsupported manifest or changed experiment source/config")
    state = nn.state.safe_load(str(checkpoint))
    model = Transformer(**MODEL_CONFIG)
    expected = nn.state.get_state_dict(model)
    if state.keys() != expected.keys():
        raise ValueError("checkpoint parameter keys do not match the original model")
    for name, tensor in state.items():
        if tensor.shape != expected[name].shape:
            raise ValueError(f"checkpoint shape mismatch: {name}")
        if not np.isfinite(tensor.numpy()).all():
            raise ValueError(f"nonfinite checkpoint weights: {name}")
    if ablate:
        # Replace before loading: never assign into storage mapped from the checkpoint.
        name = "tbs.0.out.0"
        state[name] = Tensor.zeros(*state[name].shape)
    nn.state.load_state_dict(model, state, verbose=False)
    return model


def evaluate(model: Transformer) -> Evaluation:
    """Check the entire original task domain against its independent arithmetic oracle."""
    if Tensor.training:
        raise ValueError("evaluation requires inference mode (dropout disabled)")
    pairs, expected = gen_data(MODULUS + 1)
    logits = np.concatenate(
        [
            model.forward(Tensor(pairs[start : start + 256])).numpy()[:, -1, :]
            for start in range(0, len(pairs), 256)
        ]
    )
    if logits.shape != (MODULUS**2, MODULUS + 1):
        raise ValueError("unexpected model output shape")
    finite_rows = np.isfinite(logits).all(axis=1)
    # Invalid rows have no prediction; argmax(NaN) must not earn accidental credit.
    predictions = np.where(finite_rows, logits.argmax(axis=1), -1)
    correct = predictions == expected
    finite = bool(finite_rows.all())
    rows = np.arange(len(expected))
    other = logits.copy()
    other[rows, expected] = -np.inf
    margin = (
        float((logits[rows, expected] - other.max(axis=1)).min()) if finite else None
    )
    return Evaluation(
        total=len(pairs),
        correct=int(correct.sum()),
        accuracy=float(correct.mean()),
        finite_logits=finite,
        minimum_margin=margin,
        unique_predictions=[int(value) for value in np.unique(predictions)],
        mismatches=[
            Mismatch(
                int(pairs[i, 0]),
                int(pairs[i, 1]),
                int(expected[i]),
                int(predictions[i]),
            )
            for i in np.flatnonzero(~correct)
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing existing evaluation result")
    candidate = load_candidate(args.candidate, args.receipt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        result = evaluate(candidate)
        Tensor.manual_seed(0)
        baseline = evaluate(Transformer(**MODEL_CONFIG))
        ablation = evaluate(load_candidate(args.candidate, args.receipt, ablate=True))
        control_passed = (
            ablation.finite_logits
            and ablation.unique_predictions == [0]
            and ablation.correct == MODULUS
        )
        report = {
            "candidate": asdict(result),
            "random_seed_0": asdict(baseline),
            "attention_ablation": asdict(ablation),
            "ablation_control_passed": control_passed,
            "checkpoint_sha256": digest(args.candidate / "model.safetensors"),
            "manifest_sha256": digest(args.candidate / "manifest.json"),
            "receipt_sha256": digest(args.receipt),
            "evaluator_sha256": digest(Path(__file__)),
            "source_sha256": SOURCE_HASHES,
            "config": MODEL_CONFIG,
        }
        json.dump(report, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    print(
        f"Candidate: {result.correct}/{result.total}, minimum margin {result.minimum_margin}; "
        f"random: {baseline.correct}/{baseline.total}; "
        f"ablated: {ablation.correct}/{ablation.total}, classes {ablation.unique_predictions}"
    )
    if not (
        result.finite_logits
        and baseline.finite_logits
        and result.correct == result.total
        and result.minimum_margin is not None
        and result.minimum_margin > 0
        and control_passed
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
