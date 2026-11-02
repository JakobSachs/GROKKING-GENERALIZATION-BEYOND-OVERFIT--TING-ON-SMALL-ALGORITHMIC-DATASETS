"""Write an arithmetic circuit into the original transformer, without inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
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
from transf import Transformer


def construct_parameters() -> dict[str, NDArray[np.float32]]:
    """Compile a homogeneous linear spline; no examples or model execution."""
    width, hidden = MODEL_CONFIG["embed_dim"], MODEL_CONFIG["ff_dim"]
    parameters: dict[str, NDArray[np.float32]] = {
        "embed": np.zeros((66, width), dtype=np.float32),
        "final": np.zeros((width, 63), dtype=np.float32),
    }
    for block in range(2):
        for name, shape in (
            ("query", (width, width)),
            ("key", (width, width)),
            ("value", (width, width)),
            ("out", (width, width)),
            ("ff1", (width, hidden)),
            ("ff2", (hidden, width)),
        ):
            parameters[f"tbs.{block}.{name}.0"] = np.zeros(shape, dtype=np.float32)
            parameters[f"tbs.{block}.{name}.1"] = np.zeros(shape[1], dtype=np.float32)
        for name in ("ln1", "ln2"):
            parameters[f"tbs.{block}.{name}.0"] = np.ones(width, dtype=np.float32)
            parameters[f"tbs.{block}.{name}.1"] = np.zeros(width, dtype=np.float32)

    # Signed pairs keep the SEP residual mean zero through the first attention.
    for token in range(MODULUS + 1):
        row = 3 + token
        operand = token if token < MODULUS else 0
        parameters["embed"][row, :4] = (operand, -operand, 1, -1)

    # Q=K=0 gives uniform attention. Three times the mean is the operand sum.
    parameters["tbs.0.value.0"][0, 0] = 1
    parameters["tbs.0.out.0"][0, 4:6] = (3, -3)

    # Both coordinates share the layernorm scale, so hinge locations survive it.
    for index, knot in enumerate(range(-1, 2 * MODULUS)):
        parameters["tbs.0.ff1.0"][4, index] = 1
        parameters["tbs.0.ff1.0"][2, index] = -knot

    # At integer sums, exactly one triangular hat is positive. Group by residue.
    for total in range(2 * MODULUS - 1):
        column = 6 + total % MODULUS
        for knot, coefficient in ((total - 1, 1), (total, -2), (total + 1, 1)):
            parameters["tbs.0.ff2.0"][knot + 1, column] += coefficient
    for residue in range(MODULUS):
        parameters["final"][6 + residue, residue] = 1
    # A mean is strictly below the unique maximum, even after layernorm centering.
    parameters["final"][6 : 6 + MODULUS, MODULUS] = 1 / MODULUS
    return parameters


def freeze_candidate(output: Path, receipt: Path) -> None:
    """Save a new checkpoint and provenance receipt, never replacing an attempt."""
    verify_sources()
    if output.exists() or receipt.exists():
        raise FileExistsError("refusing existing candidate directory or receipt")
    output.mkdir(parents=True)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    model = Transformer(**MODEL_CONFIG)
    parameters = construct_parameters()
    state = nn.state.get_state_dict(model)
    if state.keys() != parameters.keys():
        raise ValueError("compiled parameter keys do not match the original model")
    nn.state.load_state_dict(
        model,
        {key: Tensor(value, requires_grad=False) for key, value in parameters.items()},
        verbose=False,
    )
    checkpoint = output / "model.safetensors"
    nn.state.safe_save(nn.state.get_state_dict(model), str(checkpoint))
    manifest = {
        "schema_version": 1,
        "config": MODEL_CONFIG,
        "upstream_commit": UPSTREAM_COMMIT,
        "compiler_sha256": digest(Path(__file__)),
        "config_source_sha256": digest(ROOT / "experiment_config.py"),
        "source_sha256": SOURCE_HASHES,
        "checkpoint_sha256": digest(checkpoint),
        "attempt": output.name,
    }
    manifest_path = output / "manifest.json"
    with manifest_path.open("x") as handle:
        json.dump(manifest, handle, sort_keys=True, indent=2)
        handle.write("\n")
    with receipt.open("x") as handle:
        json.dump(
            {
                "checkpoint_sha256": digest(checkpoint),
                "manifest_sha256": digest(manifest_path),
            },
            handle,
            sort_keys=True,
            indent=2,
        )
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    freeze_candidate(args.output, args.receipt)
    print(f"Frozen {args.output}; commit {args.receipt} before any inference.")


if __name__ == "__main__":
    main()
