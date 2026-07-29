# EndLoss Cross-Layer GuideQuant Design

## Goal

Add a new LNQ-style scalar quantization method that keeps the existing GuideQuant curvature and SqueezeLLM initialization, then improves the row objective with sequential cross-layer EndLoss propagation.

## Architecture

The original `quantize.py` and `layerwise_nuq.py` flows remain usable. A new flow collects signed output gradients alongside the existing grouped squared saliency, accumulates GuideQuant Hessians as before, and runs a separate sequential quantizer that writes the same LUT/weight format expected by the existing packer.

## Components

- `any_precision/quantization/gradients.py` gains optional signed activation-gradient saving.
- `any_precision/quantization/crosslayer_stats.py` provides helpers for loading signed gradients, collecting layer/module inputs, computing propagated `R_l`, and updating accumulator `c`.
- `any_precision/quantization/endloss_crosslayer_quantize.py` contains the solver for `0.5 e^T H e + r^T e` and the sequential layer loop.
- `any_precision/quantization/endloss_crosslayer_main.py` mirrors `layerwise_main.py` while using the new signed-gradient cache and quantizer.
- `endloss_crosslayer_nuq.py` is the CLI entrypoint.
- `scripts/run_endloss_crosslayer_guidedquant_llama32_1b_c4_eval_ppl.sh` mirrors the provided runner and changes only the method entrypoint/output naming.

## Data Flow

1. `quantize.py` creates tokens, signed output-gradient cache, grouped squared saliency, and the existing SqueezeLLM initialization.
2. The new entrypoint loads tokens, builds or reuses GuideQuant Hessians from saliency, loads signed gradients, and quantizes layers in model order.
3. Before each linear module, compute `R = (D * c[:, None]).T @ X / T`.
4. Solve each row using the existing initialization plus the new CD/codebook objective.
5. After each module, update `c += rowsum(D * (X @ (W_hat - W).T))`.
6. Save LUT/weight files in the existing format and call the existing packer.

## Constraints

- Do not reintroduce raw mean gradients into the objective.
- Do not use a shifted target like `w - H^{-1}r`.
- Do not add new balancing, clipping, or scaling hyperparameters in version one.
- Keep Hessian damping and pack format compatible with the current LNQ code.
- Keep the old GuideQuant/LNQ scripts and functions intact.

## Testing

Unit tests cover the matrix-free propagation identities, CD candidate exactness, and closed-form codebook solve with `r`. A compile/import check verifies the new entrypoint is syntactically valid.
