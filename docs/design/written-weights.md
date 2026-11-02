# Written weights for modular addition

## Summary
Use Jakob's existing tinygrad transformer, unchanged, to attempt modular addition without training. The base is upstream `main` at `5fe20591e86bdde7a2f4323ee0225e28b69ceb39`. Construct a small arithmetic circuit directly in its parameters; freeze the first candidate before any candidate forward pass, then evaluate the complete finite domain in a fresh process. This reproduces the task, not grokking's learning transition or an independent discovery of an algorithm.

## Done when
- Original `transf.py` and `grokking.py` remain byte-for-byte unchanged.
- A CLI writes every parameter, saves a candidate checkpoint and provenance manifest before evaluation, and a separate command loads only that checkpoint into the original architecture.
- All 3,844 ordered input pairs modulo 62 are evaluated against Python's independent `(a+b)%62` oracle, including wraparound, and failures and minimum answer margin are reported without hiding them.
- A random-weight baseline and a targeted ablation provide controls. No optimizer, gradients, training examples or changes to forward computation are used to choose weights.

## Context
`grokking.py` uses `syms=63`: operands 0..61, separator token 62, inputs `[a,b,SEP]`, 2 layers, width 128, 4 heads and feedforward width 512. `transf.py` is post-layernorm, ReLU, unmasked self-attention; it concatenates positional one-hot features and token one-hot features before embedding. Dropout is inactive in inference (`Tensor.training=False`). These details differ from the original paper and must not be silently changed. `gen_data` already supplies the finite task domain but its inaccurate input-order comment is not a reason to alter the legacy training code.

## Design

### Circuit
Use six signed-coordinate channels: operand `x,-x`, reference `1,-1`, and accumulator `s,-s`. Remaining channels hold 62 class scores. All positional embeddings are zero. Operand embeddings carry their numeric values; SEP carries zero; every token has the reference pair. Signed pairs make the answer-position residual mean zero before the first layernorm.

Set first-block query/key matrices and biases to zero: each head attends uniformly over the three input tokens. Choose value/output matrices to read the operand coordinate and write three times its average into the accumulator pair. At SEP the accumulator is exactly `s=a+b`; the operand residual there is zero. The reference remains one.

First layernorm scales accumulator and reference by the same positive factor, giving coordinates `u=alpha*s` and `r=alpha`. In the first feedforward layer, use 125 ReLU hinges `H_t=ReLU(u-t*r)` for integer knots `t=-1..123`. For each possible integer sum `j=0..122`, form the triangular function `T_j=H_(j-1)-2H_j+H_(j+1)` in the output coordinate for residue `j mod 62`. At integer inputs only `T_s=r` is nonzero, so the correct residue has the largest class coordinate. This is a compact piecewise-linear arithmetic circuit with O(p) hinges, not an O(p^2) table of operand pairs. Modulo and arithmetic are permitted in the weight compiler and evaluator, never in inference.

Second-block attention/feedforward outputs are zero, all layernorm affine gains are one and biases zero. Subsequent layernorms may add a shared offset and positive scale to class coordinates, preserving their ordering. The final projection simply reads the 62 class coordinates; the unused separator class must be assigned a score that cannot win or tie for any valid input. One safe construction is a separator projection equal to the mean of all 62 class projections: strictly below their unique maximum. This also handles layernorm centering without a runtime mask.

No claim of novelty: this is a deliberately designed arithmetic circuit, not a Fourier-circuit reconstruction. The later Fourier work is useful prior art but not required for this model.

### Surfaces (new)
- `experiment_config.py`: fixed configuration, upstream source digests and shared digest/source-validation functions; contains no circuit construction or inference.
- `written_weights.py`: typed pure parameter constructor plus thin checkpoint-generation CLI. Fixed upstream configuration, no speculative arbitrary-modulus support. Zero every parameter and explicitly set normalization parameters; no random initialization may leak into the artifact.
- `evaluate_written_weights.py`: separate loader/evaluator using the existing `Transformer.forward`; validate manifest/config/checkpoint integrity and finite logits, evaluate in bounded batches, save machine-readable results. Include full-domain random and ablated controls. Do not invoke the weight constructor in the evaluation path.
- `tests/test_written_weights.py`: standard-library tests for the finite-domain correctness property, artifact roundtrip/integrity rejection and controls. Exhaustive testing is stronger than sampled property tests for this small domain.
- `README.md`: retain the original paper reference and document commands, derivation, exact result and limitations.
- `artifacts/` (ignored outputs): safetensors checkpoint (`model.safetensors`), UTF-8 JSON manifest (`manifest.json`) and evaluation JSON. Retain the first candidate and first-run result unchanged; later experiments use distinct names and an attempt log.
- `experiments/v1-freeze.json` (new, tracked): pre-inference freeze receipt containing SHA-256 of checkpoint and exact manifest bytes. The manifest contains schema version 1, the fixed architecture configuration, upstream commit, compiler and shared-configuration source SHA-256, both original source SHA-256 digests, checkpoint SHA-256 and attempt name. Generation refuses existing output directories or receipt files. Commit the receipt and constructor before any candidate forward pass. This is a local provenance record, not cryptographic proof of no prior private evaluation.

### Exact execution and controls
1. `uv sync --frozen`.
2. `CPU=1 uv run python written_weights.py --output artifacts/v1 --receipt experiments/v1-freeze.json` constructs parameters and serializes them; its call graph must contain no `Transformer.forward`, evaluation or optimizer calls. Only structural tests may run before freezing.
3. Commit constructor, design and receipt before inference; record the freeze commit in the later report.
4. `CPU=1 uv run python evaluate_written_weights.py --candidate artifacts/v1 --receipt experiments/v1-freeze.json --output artifacts/v1/evaluation.json` runs in a new process. Validate manifest/checkpoint hashes against receipt, validate config and unchanged transformer source hash, then load the checkpoint without calling the constructor. Write results exclusively (refuse existing output), including source/artifact hashes, number correct/total, mismatches, finite-logit checks, minimum correct-minus-best-other-class margin, and controls.
5. Run inference tests only after step 4. Tests that execute the candidate must first load the frozen receipt/checkpoint, never construct and evaluate a competing candidate.

The random baseline is the original `Transformer(63,3,2,128,4,512)` initialized with `Tensor.manual_seed(0)`, in inference mode. Report its complete-domain accuracy without assuming a fixed accuracy threshold. The ablation loads a fresh copy of the frozen checkpoint and zeros only `tbs.0.out.0` (first attention output matrix). This removes input aggregation at SEP while preserving all other weights. The ablated predictions must be input-independent (expected class 0) and accuracy must be 62/3844: each residue occurs exactly 62 times. Report the unique predicted classes and fail the control if it is not constant or its accuracy differs. No controls may influence the already-frozen candidate.

### Decisions
- Preserve the user's model rather than creating a separate transformer or rewriting it to make the construction easier.
- Keep the existing trained experiment intact as a distinct workflow; only the new weight-written workflow is added.
- Use CPU via `CPU=1`, existing locked tinygrad 0.11.0 and NumPy 2.4.0; no new runtime dependency.
- Freeze before observing any candidate prediction. If the first attempt fails, report it and save any revision separately.
- No deployment or server/UI changes. Git branch/worktree and a reviewable PR are the handover; no merge without approval.

## Phases
1. Implement the parameter constructor, serialization and manifest; freeze candidate v1 before any forward call. Test structural shapes and complete assignment without running inference beforehand.
2. Implement the standalone evaluator and run it in a fresh CPU process on the saved v1. Preserve results, run controls and exhaustive/roundtrip/error tests. Never silently overwrite the initial candidate.
3. Document the actual observed result and numerical limitations; run lint/type checks and independent reviews, fix verified findings, commit and open a PR without merging.

## Risks and checks
- Layernorm can invalidate naive arithmetic: paired zero-mean channels and homogeneous hinges preserve the sum/reference ratio. Verify the resulting answer margins across the entire domain.
- Floating-point cancellation near hinge knots can produce small spurious scores: record minimum correct-versus-runner-up margin and reject nonfinite logits.
- Tensor.linear uses row-vector dot products in this tinygrad version; follow the existing matrix orientations, not PyTorch's transposed convention.
- A correct oracle alone can conceal a bypass: verify the original model source digest, inspect all new inference code, and ablate the accumulator path.
- Token/position leakage: use existing token layout, zero positional weights and test swapped operands across the complete domain.

## Rollout
Standalone CLI experiment only. Re-run from a clean checkout with `uv sync --frozen`, generate a uniquely named checkpoint, then evaluate it with CPU enabled. No production service is involved; rollback is simply leaving the original training entry point unchanged. No unresolved user-owned scope choices.
