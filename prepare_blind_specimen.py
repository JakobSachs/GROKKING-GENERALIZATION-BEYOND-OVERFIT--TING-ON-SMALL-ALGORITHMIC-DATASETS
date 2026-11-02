"""Export checkpoint values without task labels, data, metrics, or source files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from tinygrad import nn

ROOT = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_specimen(checkpoint: Path, output: Path, receipt: Path) -> None:
    if output.exists() or receipt.exists():
        raise FileExistsError("refusing existing export or receipt")
    state = nn.state.safe_load(str(checkpoint))
    arrays = {name: tensor.numpy() for name, tensor in state.items()}
    if len(arrays) != 34 or sum(array.size for array in arrays.values()) != 413056:
        raise ValueError("unexpected checkpoint architecture")
    if any(
        array.dtype != np.float32 or not np.isfinite(array).all()
        for array in arrays.values()
    ):
        raise ValueError("weights must be finite float32 arrays")
    output.mkdir(parents=True)
    specimen = output / "specimen.npz"
    np.savez(specimen, **arrays)
    with np.load(specimen, allow_pickle=False) as restored:
        if set(restored.files) != arrays.keys() or any(
            not np.array_equal(restored[name], value) for name, value in arrays.items()
        ):
            raise ValueError("export did not preserve checkpoint values")
    architecture = output / "architecture.json"
    architecture.write_bytes((ROOT / "blind_architecture.json").read_bytes())
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with receipt.open("x") as handle:
        json.dump(
            {
                "checkpoint_sha256": digest(checkpoint),
                "specimen_sha256": digest(specimen),
                "architecture_sha256": digest(architecture),
                "exporter_sha256": digest(Path(__file__)),
                "parameters": sum(array.size for array in arrays.values()),
                "tensors": len(arrays),
            },
            handle,
            sort_keys=True,
            indent=2,
        )
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    export_specimen(args.checkpoint, args.output, args.receipt)
    print(
        "Exported anonymized tensors; receipt is kept outside the blind input directory."
    )


if __name__ == "__main__":
    main()
