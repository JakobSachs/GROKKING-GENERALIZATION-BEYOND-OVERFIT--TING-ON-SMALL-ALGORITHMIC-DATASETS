# Grokking: generalization beyond overfitting on small algorithmic datasets

Original paper: [Power et al., 2022](https://arxiv.org/abs/2201.02177).
The original `grokking.py` training script and `transf.py` model are unchanged.

## Blind identification of a trained network

Train the original model, export its numerical weights without task metadata, and ask a fresh `openai/gpt-6.1-sol` context what the network solves. This tests reading learned weights, unlike directly constructing an arithmetic circuit.

### Observed first blind result

Sol's leading hypothesis was:

> Two-operand modular addition, plausibly modulo 62, followed by a special readout token.

That matches the actual task. Sol assigned the exact hypothesis **about 50% subjective confidence**, with greater confidence in cyclic/arithmetic-like structure. This is one successful leading guess, not proof that Sol recovered the full circuit or can generally read arbitrary networks.

Before receiving weights, Sol listed broad short-context discrete prediction tasks, including symbolic operations, but did not identify an exact rule. After inspecting numerical tensors it cited:

- Sharp Fourier peaks across the first 62 token embeddings.
- Cosine similarity **0.99244** between the first two positional embeddings, suggesting similarly treated inputs.
- Cosine similarity **0.9999966** between the last positional embedding and the final token embedding, suggesting a dedicated readout token.
- A distinct last output column and cyclic structure in the ordinary output columns.

These numerical observations were independently checked against the exported tensors. Sol used only embeddings and output-projection analyses, not the intermediate circuit. It made 20 numerical requests: 16 succeeded and four hit output-selection bounds. Errors were retained, not replaced by a second attempt.

Visible assessments and compact provenance:
- [`experiments/blind-v1-architecture-only.md`](experiments/blind-v1-architecture-only.md)
- [`experiments/blind-v1-sol-assessment.md`](experiments/blind-v1-sol-assessment.md)
- [`experiments/blind-v1-results.json`](experiments/blind-v1-results.json)
- [`experiments/blind-v1-audit.json`](experiments/blind-v1-audit.json)

### Training and isolation

The completed run used seed 0, 5,000 AdamW updates, batch 512, the original 40% random training split, and the unchanged two-layer tinygrad transformer. It started from random initialization, not handwritten or pretrained parameters. All 34 parameter tensors changed; the parameter-space distance from initialization was **29.50847**. Final training accuracy was **1,537/1,537** and held-out accuracy **2,307/2,307**. Independent fresh-process evaluation confirmed **3,844/3,844** overall, with minimum correct-answer logit margin **4.60873**.

An earlier CPU command was interrupted by a worker's 30-second execution limit and did not save trained weights. That failed run is preserved locally. The completed run used the original JIT path and the same fixed seed/settings; the final step-5,000 checkpoint was used without selecting among seeds or checkpoints by held-out accuracy.

Checkpoint, export and interface hashes were committed at `25d03fab309a9ac1a4e21a23aa8563f5896d37c6` before the first real Sol request. The initial wire payload contained exactly a developer message and one architecture-only user message, with no tools, inherited history, project instructions or memory. Later tool access was limited to named numerical tensor operations. Sol received no task labels, training data, examples, model outputs, source files, scores or identifying filenames; it could not execute forward passes. The architecture itself was a legitimate source of clues.

The harness uses the existing Pi-managed subscription credentials. It never prints credentials or logs HTTP headers. It saves request bodies and normalized messages locally; shared reports contain only visible answers and numerical evidence, not reasoning blocks or SDK internals. Each run has a fresh session UUID, zero automatic retries and an exclusive output directory.

### Run a new experiment

Use Python 3.12+, Node.js 22+ and `uv`. Training is CPU-only in the recorded run and took about 16 minutes. These paths are new attempt names; do not delete or overwrite historical attempts.

```sh
uv sync --frozen
CPU=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 uv run python -u train_checkpoint.py --output artifacts/training-reproduction
CPU=1 uv run python prepare_blind_specimen.py \
  --checkpoint artifacts/training-reproduction/final.safetensors \
  --output artifacts/specimen-reproduction \
  --receipt experiments/reproduction-freeze.json
# Commit the checkpoint/export receipt and unchanged protocol BEFORE the model call.
git add experiments/reproduction-freeze.json
git commit -m "Freeze new blind-reading specimen before assessment"
node run_blind_sol.mjs \
  --sdk /path/to/pi-coding-agent/dist/index.js \
  --python "$(pwd)/.venv/bin/python" \
  --specimen "$(pwd)/artifacts/specimen-reproduction/specimen.npz" \
  --architecture "$(pwd)/artifacts/specimen-reproduction/architecture.json" \
  --inspector "$(pwd)/weight_inspector.py" \
  --receipt "$(pwd)/experiments/reproduction-freeze.json" \
  --output "$(pwd)/artifacts/readout-reproduction"
```

The SDK path points to an installed `earendil-works/pi-coding-agent` package. The runner uses its `ModelRuntime` directly, not an agent session with resource discovery. The requested model must be in that SDK's catalog, with existing OpenAI sign-in configured under the normal Pi agent directory.

Training stops and labels the checkpoint partial if its fixed time budget would be exceeded; do not describe a partial run as completed 5,000-step training. A blind call failure is retained as a failed attempt, not automatically retried.

### Checks

```sh
CPU=1 uv run --with pytest python -m pytest -q tests/test_weight_inspector.py tests/test_train_checkpoint.py tests/test_prepare_blind_specimen.py
node --test tests/test_blind_protocol.mjs
uv run --with ruff ruff check train_checkpoint.py weight_inspector.py prepare_blind_specimen.py tests/test_weight_inspector.py tests/test_train_checkpoint.py tests/test_prepare_blind_specimen.py
uv run --with mypy mypy --follow-imports=silent --ignore-missing-imports train_checkpoint.py weight_inspector.py prepare_blind_specimen.py
```

For the original training-only entry point, run `uv run python grokking.py`.
