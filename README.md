# Grokking: generalization beyond overfitting on small algorithmic datasets

Original paper: [Power et al., 2022](https://arxiv.org/abs/2201.02177).
The original tinygrad training experiment remains in `grokking.py`, with its model in `transf.py`.

## Writing weights instead of training

This experiment attempts the same modular-addition task by directly constructing **every parameter** of the existing model. There is no optimizer, fitting to examples, or altered forward pass.

The task is `a + b mod 62`, with each number represented by a token, and input `[a, b, SEP]`. The existing configuration has two transformer blocks, width 128, four heads, feedforward width 512, post-layer normalization, ReLU, and unmasked attention. This is the repository's original model, not an exact reproduction of the original paper's causal transformer.

### How the circuit works

1. Token embeddings hold the operand value, a constant reference, and their negatives. Positional embedding weights are zero.
2. Zero queries and keys give uniform attention. The value/output projections write the sum of both operands into an accumulator at SEP.
3. Layer normalization scales the accumulator and reference equally. ReLU hinges of the form `ReLU(sum - threshold * reference)` therefore retain their thresholds.
4. Three adjacent hinges form a triangular score around each integer sum. Those scores feed the corresponding residue class.
5. The remaining block preserves class-score ordering; the final projection reads those scores. The SEP logit is the mean of the numeric logits, so it cannot beat their unique maximum.

The compiler uses arithmetic to set constants; **inference uses only the unchanged transformer**. The construction uses 125 hinges, not a table of the 3,844 operand pairs. This is an intentionally designed arithmetic circuit, not a novel algorithm or an independently discovered Fourier circuit.

### Freeze, then evaluate

Use Python 3.12+ and [uv](https://docs.astral.sh/uv/). From this directory:

```sh
uv sync --frozen
CPU=1 uv run python written_weights.py --output artifacts/v1 --receipt experiments/v1-freeze.json
# Commit the compiler and freeze receipt BEFORE any candidate inference.
git add experiment_config.py written_weights.py evaluate_written_weights.py experiments/v1-freeze.json docs/design/written-weights.md
# Also commit README/.gitignore changes if starting from the experiment's base commit.
git commit -m "Freeze first manually written modular-addition candidate"
CPU=1 uv run python evaluate_written_weights.py --candidate artifacts/v1 --receipt experiments/v1-freeze.json --output artifacts/v1/evaluation.json
```

Generation and evaluation refuse existing output paths. After the published v1 attempt, use a **new** directory and receipt (for example `artifacts/reproduction` and `experiments/reproduction-freeze.json`) rather than deleting or overwriting v1. The tracked receipt binds the checkpoint and manifest hashes; the manifest records the exact upstream source, compiler and configuration hashes. This is local provenance, not proof that nobody privately evaluated the algorithm beforehand.

The evaluator does not import or call the weight compiler. In a fresh process it loads the checkpoint and checks all **3,844 ordered pairs**, recording mismatches, finite logits and the minimum correct-versus-runner-up logit margin. It also evaluates:

- The original randomly initialized model with seed 0.
- The frozen model with only its first attention output matrix zeroed. This removes operand aggregation; it should predict zero for every pair and score 62/3,844.

Artifacts are local, ignored files. The first freeze receipt and a compact result record are tracked in `experiments/`. See `docs/design/written-weights.md` for the derivation, protocol and checks.

### First frozen attempt

| Model | Correct / total | Accuracy |
| --- | ---: | ---: |
| Directly written weights | 3,844 / 3,844 | 100% |
| Original random initialization, seed 0 | 62 / 3,844 | 1.61% |
| Written weights, aggregation ablated | 62 / 3,844 | 1.61% |

The correct answer beats every other token by at least **0.0645126 logits** over the complete domain. All logits are finite. The model has 413,056 parameters; 1,505 are nonzero in this construction, including normalization gains. The candidate was frozen at commit `74a2e3cf5fef7ebc57430d07357ee9e4fc198ba9` before evaluation. The observed result is recorded in [`experiments/v1-results.json`](experiments/v1-results.json).

### Verification

```sh
CPU=1 uv run python -m unittest discover -s tests -v
uv run --with ruff ruff check experiment_config.py written_weights.py evaluate_written_weights.py tests
uv run --with mypy mypy --follow-imports=silent --ignore-missing-imports experiment_config.py written_weights.py evaluate_written_weights.py tests
```

On a fresh checkout, the tests reconstruct the frozen artifact in a temporary directory and require its checkpoint **and manifest bytes** to match the original tracked receipt before any inference. A differing reconstruction fails rather than being accepted as a new candidate. Tests also check full-domain accuracy, serialization, seed independence, numerical failures, tamper rejection and the attention ablation.

### Scope

Success means that we can deliberately program this transformer to generalize over the entire modular-addition domain. It does **not** reproduce grokking's delayed learning transition, show natural-language understanding, establish that arbitrary pretrained weights can be interpreted, or compare compute fairly against training.
