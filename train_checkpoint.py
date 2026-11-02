"""Train the original model with fixed hyperparameters and save the final checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import signal
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from tinygrad import Device, Tensor, TinyJit, nn

from grokking import gen_data
from transf import Transformer

STEPS, BATCH, SEED = 5000, 512, 0
SOURCE_HASHES = {
    "grokking.py": "32a946ad312370621e974e0a90c92f5248dfe298d23910dd138f66b5129495d9",
    "transf.py": "97a7e02934f0709d0ce389a44dac3654a5ef9251ccd6035f00d42a4aced83bca",
}
ROOT = Path(__file__).resolve().parent


def finite_state(state: dict[str, Tensor]) -> None:
    for name, tensor in state.items():
        if not np.isfinite(tensor.numpy()).all():
            raise FloatingPointError(f"non-finite parameter: {name}")


def evaluate(model: Transformer, x: Tensor, y: Tensor) -> dict[str, float]:
    if Tensor.training:
        raise ValueError("evaluation must disable dropout")
    total_loss, correct, count = 0.0, 0, len(x)
    for start in range(0, count, 256):
        stop = min(start + 256, count)
        logits = model.forward(x[start:stop])[:, -1]
        values = logits.numpy()
        if not np.isfinite(values).all():
            raise FloatingPointError("non-finite evaluation logits")
        total_loss += logits.cross_entropy(y[start:stop]).item() * (stop - start)
        correct += int(
            (values.argmax(axis=1) == y[start:stop].numpy().argmax(axis=1)).sum()
        )
    loss, accuracy = total_loss / count, correct / count
    if not math.isfinite(loss):
        raise FloatingPointError("non-finite evaluation loss")
    return {"loss": loss, "accuracy": accuracy}


def save_state(state: dict[str, Tensor], path: Path) -> None:
    with path.open("xb") as handle:
        np.savez(handle, **{key: value.numpy() for key, value in state.items()})


def split_indices(count: int) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    order = Tensor.randperm(count).numpy().astype(np.int64)
    split = int(count * 0.4)
    return order[:split], order[split:]


def train(output: Path) -> None:
    actual_sources = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in SOURCE_HASHES
    }
    if actual_sources != SOURCE_HASHES:
        raise ValueError("original model or training source has changed")
    output.mkdir(parents=True, exist_ok=False)
    Tensor.manual_seed(SEED)
    model = Transformer(63, 3, 2, 128, 4, 512)
    data, labels = gen_data(63)
    x, y = Tensor(data), Tensor(labels).one_hot(63)
    train_indices, test_indices = split_indices(len(data))
    train_x, train_y = x[Tensor(train_indices)], y[Tensor(train_indices)]
    test_x, test_y = x[Tensor(test_indices)], y[Tensor(test_indices)]
    np.savez(output / "split_indices.npz", train=train_indices, test=test_indices)
    state = nn.state.get_state_dict(model)
    finite_state(state)
    # Host copies cannot alias optimizer-updated parameter storage.
    initial = {key: value.numpy().copy() for key, value in state.items()}
    save_state(state, output / "initial.npz")
    nn.state.safe_save(state, str(output / "initial.safetensors"))
    optimizer = nn.optim.AdamW(nn.state.get_parameters(model), weight_decay=1)
    metadata: dict[str, object] = {
        "status": "running",
        "seed": SEED,
        "steps_requested": STEPS,
        "actual_steps": 0,
        "batch_size": BATCH,
        "split_ratio": 0.4,
        "optimizer": "tinygrad.nn.optim.AdamW",
        "optimizer_settings": "original tinygrad 0.11.0 defaults, weight_decay=1",
        "model": {
            "syms": 63,
            "maxlen": 3,
            "layers": 2,
            "width": 128,
            "heads": 4,
            "ff_dim": 512,
            "dropout": 0.1,
        },
        "train_size": len(train_indices),
        "test_size": len(test_indices),
        "source_sha256": actual_sources,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "device": Device.DEFAULT,
        "maximum_seconds": 1800,
    }

    def save_metadata() -> None:
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    @TinyJit
    def train_step() -> None:
        samples = Tensor.randint(BATCH, low=0, high=train_x.shape[0])
        loss = model.forward(train_x[samples])[:, -1].cross_entropy(train_y[samples])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def terminate(_signal: int, _frame: object) -> None:
        raise KeyboardInterrupt("training process interrupted")

    previous_handler = signal.signal(signal.SIGTERM, terminate)
    save_metadata()
    started, warmed = time.monotonic(), None
    completed = 0
    try:
        with (output / "progress.jsonl").open("x") as log:
            for step in range(1, STEPS + 1):
                with Tensor.train():
                    train_step()
                completed = step
                elapsed = time.monotonic() - started
                if step == 3:
                    warmed = elapsed
                if step <= 3 or step % 100 == 0:
                    entry: dict[str, object] = {
                        "step": step,
                        "elapsed_seconds": elapsed,
                    }
                    if step % 100 == 0:
                        finite_state(nn.state.get_state_dict(model))
                        entry["train"] = evaluate(model, train_x, train_y)
                        entry["test"] = evaluate(model, test_x, test_y)
                    log.write(json.dumps(entry) + "\n")
                    log.flush()
                    metadata["actual_steps"] = step
                    metadata["elapsed_seconds"] = elapsed
                    save_metadata()
                    print(json.dumps(entry), flush=True)
                if step == 100:
                    steady = (elapsed - (warmed or 0)) / 97
                    projected = elapsed + steady * (STEPS - 100)
                    metadata["benchmark_100_steady_seconds_per_step"] = steady
                    metadata["projected_5000_seconds"] = projected
                    if projected > 1800:
                        metadata["status"] = "partial-time-budget"
                        break
                if elapsed > 1800:
                    metadata["status"] = "partial-time-budget"
                    break
        final = nn.state.get_state_dict(model)
        finite_state(final)
        save_state(final, output / "final.npz")
        nn.state.safe_save(final, str(output / "final.safetensors"))
        delta = math.sqrt(
            sum(
                float(np.square(value.numpy().astype(np.float64) - initial[key]).sum())
                for key, value in final.items()
            )
        )
        metadata.update(
            {
                "actual_steps": completed,
                "status": "complete" if completed == STEPS else "partial-time-budget",
                "initial_to_final_l2": delta,
                "full_domain": evaluate(model, x, y),
                "final_train": evaluate(model, train_x, train_y),
                "final_test": evaluate(model, test_x, test_y),
                "elapsed_seconds": time.monotonic() - started,
                "final_checkpoint_sha256": hashlib.sha256(
                    (output / "final.safetensors").read_bytes()
                ).hexdigest(),
            }
        )
        save_metadata()
        print(json.dumps(metadata), flush=True)
    except (Exception, KeyboardInterrupt):
        metadata.update(
            status="interrupted",
            actual_steps=completed,
            elapsed_seconds=time.monotonic() - started,
        )
        save_metadata()
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train(args.output)


if __name__ == "__main__":
    main()
