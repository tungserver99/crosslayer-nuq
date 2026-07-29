# EndLoss Cross-Layer GuideQuant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a new sequential cross-layer EndLoss quantization flow on top of GuideQuant/LNQ without replacing the existing flow.

**Architecture:** Extend gradient collection to persist signed output gradients, reuse existing Hessian accumulation and packing, and add a new quantizer/entrypoint that optimizes `0.5 e^T H e + r^T e`. The new flow writes the same LUT/weight cache structure as `layerwise_nuq.py` so existing pack code continues to work.

**Tech Stack:** Python, PyTorch, NumPy, pytest, existing `any_precision` analyzer/packer utilities, bash runner.

## Global Constraints

- Keep existing `quantize.py`, `layerwise_nuq.py`, and `any_precision/quantization/layerwise_quantize.py` behavior compatible.
- Do not reintroduce raw calibration mean-gradient into the new objective.
- Do not compute a shifted target with `H^{-1}r`.
- Do not add propagated-error scaling or clipping hyperparameters.
- Save quantized weights/LUTs in the existing packer-compatible cache format.

---

### Task 1: Propagation Math Helpers

**Files:**
- Create: `any_precision/quantization/crosslayer_stats.py`
- Test: `tests/test_crosslayer_stats.py`

**Interfaces:**
- Produces: `flatten_calibration_tensor(tensor: torch.Tensor) -> torch.Tensor`
- Produces: `compute_propagated_R(X: torch.Tensor, D: torch.Tensor, c: torch.Tensor) -> torch.Tensor`
- Produces: `update_error_accumulator(c: torch.Tensor, X: torch.Tensor, D: torch.Tensor, error: torch.Tensor) -> torch.Tensor`

- [ ] **Step 1: Write failing tests**

```python
import torch
from any_precision.quantization.crosslayer_stats import (
    compute_propagated_R,
    update_error_accumulator,
)


def test_compute_propagated_R_matches_explicit_sum():
    X = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    D = torch.tensor([[[0.5, -1.0], [2.0, 0.25]]])
    c = torch.tensor([1.5, -0.5])
    expected = (D.reshape(-1, 2) * c[:, None]).T @ X.reshape(-1, 2) / 2
    assert torch.allclose(compute_propagated_R(X, D, c), expected)


def test_update_error_accumulator_matches_identity():
    X = torch.tensor([[[1.0, 0.0], [0.0, 2.0]]])
    D = torch.tensor([[[1.0, 3.0], [-2.0, 0.5]]])
    E = torch.tensor([[0.25, -0.5], [1.0, 0.75]])
    c = torch.zeros(2)
    delta_z = X.reshape(-1, 2) @ E.T
    expected = (D.reshape(-1, 2) * delta_z).sum(dim=1)
    assert torch.allclose(update_error_accumulator(c, X, D, E), expected)
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `pytest tests/test_crosslayer_stats.py -q`

- [ ] **Step 3: Implement helpers**

Implement flattening from `[N, S, C]` to `[T, C]`, `R = (D * c[:, None]).T @ X / T`, and accumulator update.

- [ ] **Step 4: Run tests and confirm pass**

Run: `pytest tests/test_crosslayer_stats.py -q`

### Task 2: Solver With Linear Term

**Files:**
- Create: `any_precision/quantization/endloss_crosslayer_quantize.py`
- Test: `tests/test_endloss_crosslayer_solver.py`

**Interfaces:**
- Consumes: `compute_propagated_R(...)`
- Produces: `objective_function_with_r(W, H, labels, C, R=None) -> torch.Tensor`
- Produces: `update_P_with_r(W, H, labels, C, R, cd_cycles, verbose=True) -> torch.Tensor`
- Produces: `update_C_with_r(W, H, labels, C, R, iteration) -> torch.Tensor`

- [ ] **Step 1: Write failing solver tests**

Test CD candidate exactness against brute-force full objective for one coordinate and test codebook solve gradient is near zero.

- [ ] **Step 2: Run tests and confirm failure**

Run: `pytest tests/test_endloss_crosslayer_solver.py -q`

- [ ] **Step 3: Implement solver**

Copy the smallest necessary structure from `layerwise_quantize.py`, then change residual to `s = H @ (q - w) + r`, CD candidate cost to `s_i * delta + 0.5 * H_ii * delta**2`, and codebook solve RHS to `P^T(Hw - r)`.

- [ ] **Step 4: Run tests and confirm pass**

Run: `pytest tests/test_endloss_crosslayer_solver.py -q`

### Task 3: Signed Gradient Cache

**Files:**
- Modify: `any_precision/quantization/gradients.py`
- Modify: `any_precision/quantization/main.py`
- Modify: `quantize.py`

**Interfaces:**
- Produces: optional `signed_grad_path` argument in `get_gradients(...)`
- Produces: optional `signed_gradients_path` / CLI `--signed_gradients_path`

- [ ] **Step 1: Add optional signed cache path**

Store signed output gradients before squaring/scaling for each hooked module as CPU tensors.

- [ ] **Step 2: Preserve old behavior**

When the new argument is not provided, current gradient/saliency behavior remains unchanged.

- [ ] **Step 3: Compile**

Run: `python -m compileall -q any_precision quantize.py`

### Task 4: New Pipeline Entry

**Files:**
- Create: `any_precision/quantization/endloss_crosslayer_main.py`
- Modify: `any_precision/quantization/__init__.py`
- Create: `endloss_crosslayer_nuq.py`

**Interfaces:**
- Consumes: existing Hessian cache, signed-gradient cache, initialization cache.
- Produces: packed model directory named with `_endlossxl`.

- [ ] **Step 1: Implement entrypoint**

Mirror `layerwise_main.py`, add signed-gradient cache path logging, call the new sequential seed function, then call existing `pack(...)`.

- [ ] **Step 2: Compile**

Run: `python -m compileall -q any_precision endloss_crosslayer_nuq.py`

### Task 5: Runner Script

**Files:**
- Create: `scripts/run_endloss_crosslayer_guidedquant_llama32_1b_c4_eval_ppl.sh`

**Interfaces:**
- Consumes: `quantize.py` and `endloss_crosslayer_nuq.py`
- Produces: C4 PPL eval runner with the same structure as the provided sample.

- [ ] **Step 1: Copy structure from sample runner**

Use the same variable defaults and eval block, replacing only the new entrypoint and packed model/tag names.

- [ ] **Step 2: Compile/check shell content**

Run: `Get-Content -Path scripts/run_endloss_crosslayer_guidedquant_llama32_1b_c4_eval_ppl.sh`

### Task 6: Final Verification

**Files:**
- Verify all touched files.

**Interfaces:**
- Consumes: all previous tasks.
- Produces: concise final status.

- [ ] **Step 1: Run unit tests**

Run: `pytest tests/test_crosslayer_stats.py tests/test_endloss_crosslayer_solver.py -q`

- [ ] **Step 2: Run compileall**

Run: `python -m compileall -q any_precision quantize.py endloss_crosslayer_nuq.py`

- [ ] **Step 3: Review git diff**

Run: `git diff -- docs any_precision quantize.py endloss_crosslayer_nuq.py scripts`
