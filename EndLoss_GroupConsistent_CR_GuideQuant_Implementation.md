# EndLoss Group-Consistent Cross-Layer Propagation for GuideQuant
## Implementation specification: keep the original EndLoss objective, redesign only the cross-layer \(\mathcal C/R\) estimator

> **For Codex / implementation agents**
>
> This document is the authoritative implementation specification for replacing the current scalar cross-layer accumulator in `EndLoss_Sequential_CrossLayer_GuideQuant_CD.md`.
>
> The original EndLoss, GuideQuant Hessian, CD assignment, exact codebook update, and SqueezeLLM/GuideQuant initialization **must remain unchanged**.
>
> Only the estimator of the cross-layer term
>
> \[
> r_l \approx F_{l,<l}e_{<l}
> \]
>
> is redesigned so that it uses the same output grouping and the same scale as the GuideQuant approximation of \(H_{ll}\).

---

# 1. Status and scope

This file supersedes the following parts of `EndLoss_Sequential_CrossLayer_GuideQuant_CD.md`:

- the scalar accumulator \(c_t\);
- the old update

  \[
  c\leftarrow c+\operatorname{rowsum}(D\odot\Delta Z);
  \]

- the old propagated term

  \[
  R=(D\odot c[:,None])^\top X;
  \]

- the runtime heuristic

  ```python
  R = R / D.shape[-1]
  ```

All other parts of the original method remain active unless this file explicitly says otherwise.

## 1.1. Keep unchanged

The implementation must keep:

1. the global NLL second-order objective;
2. the assumption that the raw calibration mean-gradient is omitted;
3. the sequential local objective

   \[
   \boxed{
   J_l(e_l)
   =
   \frac12e_l^\top H_{ll}e_l+r_l^\top e_l;
   }
   \]

4. the GuideQuant grouped empirical-Fisher Hessian;
5. the existing signed output-gradient cache;
6. the current CD assignment cost;
7. the current exact codebook update with \(R\);
8. the existing initialization;
9. the existing layer/module quantization order;
10. token-sum scaling currently used by the GuideQuant Hessian code.

## 1.2. Do not introduce

The implementation must not introduce:

- teacher/student reconstruction loss;
- activation matching as a replacement objective;
- a new tunable damping coefficient;
- a manually tuned cross-layer coefficient;
- normalization by total output dimension \(d_{\mathrm{out}}\);
- normalization by number of groups \(G\);
- quantized activations in place of the full-precision Taylor expansion point;
- changes to CD or codebook optimization before the new statistics are independently verified.

---

# 2. Naming convention

There are two unrelated objects commonly called `C` in this repository:

1. the **cross-layer propagation accumulator** introduced here;
2. the codebook/centroid tensor named `C` inside the solver.

To avoid ambiguity:

- mathematical notation for the propagation accumulator is \(\mathcal C\);
- Python runtime name must be `group_accumulator`;
- `C` inside `endloss_crosslayer_quantize.py` continues to mean codebook centroids.

Do not name the runtime propagation tensor `C` in production code.

---

# 3. Matrix convention

For one `nn.Linear` module \(l\):

\[
W_l\in\mathbb R^{d_{\mathrm{out}}\times d_{\mathrm{in}}},
\]

\[
X_l\in\mathbb R^{T\times d_{\mathrm{in}}},
\]

\[
Z_l=X_lW_l^\top,
\]

\[
D_l=
\frac{\partial L_{\mathrm{NLL}}}{\partial Z_l}
\in\mathbb R^{T\times d_{\mathrm{out}}}.
\]

After quantization:

\[
E_l=\widehat W_l-W_l
\in\mathbb R^{d_{\mathrm{out}}\times d_{\mathrm{in}}},
\]

and the linear output perturbation is:

\[
\boxed{
\Delta Z_l=X_lE_l^\top
\in\mathbb R^{T\times d_{\mathrm{out}}}.
}
\]

`T` is the flattened calibration token count. The flattening order of `X` and `D` must be identical.

---

# 4. GuideQuant grouped Hessian

Let the final GuideQuant Hessian use \(G\) output groups. For module \(l\):

\[
G=\texttt{num\_groups},
\]

\[
m_l=rac{d_{\mathrm{out}}^{(l)}}{G}.
\]

The module must satisfy:

\[
\boxed{
d_{\mathrm{out}}^{(l)}\bmod G=0.
}
\]

Let \(J_{l,g}\) be the contiguous set of output rows assigned to group \(g\), with:

\[
|J_{l,g}|=m_l.
\]

GuideQuant computes group saliency:

\[
\boxed{
s_{l,g}[t]
=
\frac1{m_l}
\sum_{j\in J_{l,g}}D_l[t,j]^2.
}
\]

The Hessian shared by rows in group \(g\) is:

\[
\boxed{
H_{l,g}
=
\sum_{t=1}^{T}
s_{l,g}[t]x_{l,t}x_{l,t}^\top.
}
\]

This is the current token-sum convention used by `SaliencyEngine.XTX`.

For an error row \(e_{l,j}\), the GuideQuant self-quadratic contribution is:

\[
\frac12e_{l,j}^\top H_{l,g(j)}e_{l,j}.
\]

No change is made to this Hessian.

---

# 5. Why the old scalar accumulator is incompatible

The previous implementation uses:

\[
c_l[t]
=
\sum_{m<l}
\sum_{j=1}^{d_{\mathrm{out}}^{(m)}}
D_m[t,j]\Delta Z_m[t,j].
\]

It then computes:

\[
R_l
=
(D_l\odot c_l[:,None])^\top X_l.
\]

This combines:

- a GuideQuant diagonal block based on **mean squared gradients inside each group**;
- a cross-layer block based on a **raw signed sum over every output row and every group**.

The raw signed aggregation may contain cross-output energy that the GuideQuant diagonal block deliberately discarded. Therefore the resulting global quadratic need not remain positive semidefinite. In practice this can produce:

- very large \(R\);
- very large \(H^{-1}R\) or its Cholesky-equivalent inside codebook update;
- centroid/weight explosion;
- NaN PPL.

The current line:

```python
R = R / D.shape[-1]
```

reduces the symptom, but it is not derived from the GuideQuant grouping and makes the cross-layer signal too weak or incorrectly scaled.

The new estimator must preserve group structure from the moment the previous-layer contribution is accumulated.

---

# 6. New group-consistent cross-layer estimator

## 6.1. Normalized signed contribution of one module and one group

For module \(l\), group \(g\), and token \(t\), define:

\[
\boxed{
a_{l,g}[t]
=
\frac1{\sqrt{m_l}}
\sum_{j\in J_{l,g}}
D_l[t,j]\Delta Z_l[t,j].
}
\]

This is the signed first-order contribution of one GuideQuant output group, normalized by the square root of that group's row count.

## 6.2. Why the normalization is \(1/\sqrt{m_l}\)

By Cauchy-Schwarz:

\[
\left(
\sum_{j\in J_{l,g}}
D_l[t,j]\Delta Z_l[t,j]
\right)^2
\leq
\left(
\sum_{j\in J_{l,g}}D_l[t,j]^2
\right)
\left(
\sum_{j\in J_{l,g}}\Delta Z_l[t,j]^2
\right).
\]

Because:

\[
\sum_{j\in J_{l,g}}D_l[t,j]^2
=
m_l s_{l,g}[t],
\]

we obtain:

\[
\boxed{
a_{l,g}[t]^2
\leq
s_{l,g}[t]
\sum_{j\in J_{l,g}}\Delta Z_l[t,j]^2.
}
\]

The right-hand side is exactly the per-token, per-group energy represented by the GuideQuant Hessian.

Therefore the new signed cross-layer contribution is bounded in the same metric used by the GuideQuant diagonal block.

## 6.3. Group accumulator

Before quantizing module \(l\), define:

\[
\boxed{
\mathcal C_l[t,g]
=
\sum_{m<l}a_{m,g}[t].
}
\]

Shape:

\[
\boxed{
\mathcal C_l\in\mathbb R^{T\times G}.
}
\]

The first module sees:

\[
\mathcal C_1=0.
\]

The accumulator keeps one signed propagation channel for each final GuideQuant Hessian group. It does not collapse all groups into one scalar.

## 6.4. Cross-layer linear term for the current module

The cross-layer contribution involving current module \(l\) is:

\[
\sum_{t,g}
\mathcal C_l[t,g]a_{l,g}[t].
\]

Substituting \(a_{l,g}\):

\[
\sum_{g}
\sum_{j\in J_{l,g}}
\left[
\frac1{\sqrt{m_l}}
\sum_t
\mathcal C_l[t,g]D_l[t,j]x_{l,t}
\right]^\top e_{l,j}.
\]

Therefore, for row \(j\in J_{l,g}\):

\[
\boxed{
r_{l,j}
=
\frac1{\sqrt{m_l}}
\sum_t
\mathcal C_l[t,g]D_l[t,j]x_{l,t}.
}
\]

Matrix form for one group:

\[
\boxed{
R_l[J_{l,g},:]
=
\frac1{\sqrt{m_l}}
\left(
D_l[:,J_{l,g}]
\odot
\mathcal C_l[:,g,None]
\right)^\top
X_l.
}
\]

The complete \(R_l\) has shape:

\[
R_l\in\mathbb R^{d_{\mathrm{out}}\times d_{\mathrm{in}}}.
\]

There is no additional division by `d_out`, `group_size`, or `num_groups` after this formula.

---

# 7. The loss and solver remain unchanged

For current row \(j\in J_{l,g}\):

\[
\boxed{
J_{l,j}(e_{l,j})
=
\frac12e_{l,j}^\top H_{l,g}e_{l,j}
+r_{l,j}^\top e_{l,j}.
}
\]

The implementation must continue to pass the new \(R_l\) to the existing solver.

## 7.1. CD assignment

Keep the current candidate cost:

\[
\Delta J
=
s_i\delta+
\frac12H_{ii}\delta^2,
\]

where the maintained residual is:

\[
s=He+r.
\]

## 7.2. Codebook update

Keep the current exact codebook update for fixed assignments:

\[
(P^\top HP)c
=
P^\top(Hw-r).
\]

No changes are required in:

- `choose_codeword_by_delta_cost`;
- `update_P_with_r`;
- `update_C_with_r`;
- `objective_function_with_r`;
- the SqueezeLLM/GuideQuant initialization.

Do not change the solver while validating the new statistics.

---

# 8. Global structured-Fisher interpretation

For module \(l\), group \(g\), token \(t\), define the GuideQuant group energy:

\[
q_{l,g}[t]
=
s_{l,g}[t]
\sum_{j\in J_{l,g}}\Delta Z_l[t,j]^2.
\]

The proposed global approximation is:

\[
\boxed{
\widetilde{\Delta L}
=
\frac12
\sum_{l,g,t}q_{l,g}[t]
+
\sum_{m<l}\sum_{g,t}
a_{m,g}[t]a_{l,g}[t].
}
\]

This keeps the GuideQuant diagonal blocks exactly and replaces only the cross-layer blocks.

It can be rewritten as:

\[
\widetilde{\Delta L}
=
\frac12
\sum_{g,t}
\left(
\sum_l a_{l,g}[t]
\right)^2
+
\frac12
\sum_{l,g,t}
\left(
q_{l,g}[t]-a_{l,g}[t]^2
\right).
\]

Because:

\[
a_{l,g}[t]^2\leq q_{l,g}[t],
\]

we have:

\[
\boxed{
\widetilde{\Delta L}\geq0.
}
\]

This is the key structural property missing from the old raw scalar accumulation.

---

# 9. Handling modules with different output dimensions

All modules use the same final number of GuideQuant groups \(G\), but their output dimensions and group sizes may differ.

For previous module \(m\):

\[
m_m=rac{d_{\mathrm{out}}^{(m)}}G.
\]

Its contribution is inserted into the accumulator with:

\[
\frac1{\sqrt{m_m}}.
\]

For current module \(l\):

\[
m_l=rac{d_{\mathrm{out}}^{(l)}}G.
\]

Its \(R_l\) is computed with:

\[
\frac1{\sqrt{m_l}}.
\]

Thus the cross coefficient between the two modules is automatically:

\[
\boxed{
\frac1{\sqrt{m_m m_l}}.
}
\]

No assumption that all modules have the same `d_out` is required.

## 9.1. Group-index alignment assumption

Version 1 intentionally pairs group index \(g\) across sequential modules:

\[
g_{\mathrm{previous}}\leftrightarrow g_{\mathrm{current}}.
\]

This is a structured approximation, not an assertion that the groups are semantically identical. It is selected because:

- it preserves the exact GuideQuant group partition;
- it avoids collapsing all groups into one scalar;
- it produces a bounded cross-layer surrogate;
- it introduces no new learned mapping or hyperparameter.

Do not add cross-group mixing in version 1.

---

# 10. Required implementation changes

## 10.1. File map

Modify:

- `any_precision/quantization/crosslayer_stats.py`
- `any_precision/quantization/endloss_crosslayer_main.py`
- `tests/test_crosslayer_stats.py`
- `tests/test_endloss_crosslayer_runtime_tokens.py`

Do not modify for this change:

- `any_precision/quantization/endloss_crosslayer_quantize.py`
- `any_precision/quantization/layerwise_quantize.py`
- initialization code;
- Hessian accumulation formulas;
- signed-gradient collection formulas.

## 10.2. New production interfaces

Replace the scalar-statistics interfaces with:

```python
from typing import Optional

import torch


def compute_grouped_propagated_R(
    X: torch.Tensor,
    D: torch.Tensor,
    group_accumulator: torch.Tensor,
    num_groups: int,
    normalize_by_tokens: bool = False,
) -> torch.Tensor:
    """Return R with shape [d_out, d_in] using GuideQuant-consistent groups."""


def update_grouped_error_accumulator(
    group_accumulator: torch.Tensor,
    X: torch.Tensor,
    D: torch.Tensor,
    error: torch.Tensor,
    num_groups: int,
) -> torch.Tensor:
    """Return updated propagation accumulator with shape [T, num_groups]."""
```

The old functions may be deleted or retained only as private compatibility wrappers. Production runtime must not call them:

```python
compute_propagated_R
update_error_accumulator
```

---

# 11. Exact implementation for `crosslayer_stats.py`

## 11.1. Shared validation

Add a private helper:

```python
def _validate_grouped_shapes(
    X_flat: torch.Tensor,
    D_flat: torch.Tensor,
    group_accumulator: torch.Tensor,
    num_groups: int,
) -> int:
    if num_groups < 1:
        raise ValueError(f"num_groups must be >= 1, got {num_groups}")
    if X_flat.dim() != 2 or D_flat.dim() != 2:
        raise ValueError("X_flat and D_flat must be 2D")
    if X_flat.shape[0] != D_flat.shape[0]:
        raise ValueError(
            f"X and D token counts differ: {X_flat.shape[0]} vs {D_flat.shape[0]}"
        )
    if group_accumulator.shape != (X_flat.shape[0], num_groups):
        raise ValueError(
            "group_accumulator must have shape "
            f"({X_flat.shape[0]}, {num_groups}), got {tuple(group_accumulator.shape)}"
        )
    output_dim = D_flat.shape[1]
    if output_dim % num_groups != 0:
        raise ValueError(
            f"output_dim {output_dim} must be divisible by num_groups {num_groups}"
        )
    return output_dim // num_groups
```

## 11.2. Compute grouped \(R\)

Use one final GEMM rather than a Python loop over rows:

```python
def compute_grouped_propagated_R(
    X: torch.Tensor,
    D: torch.Tensor,
    group_accumulator: torch.Tensor,
    num_groups: int,
    normalize_by_tokens: bool = False,
) -> torch.Tensor:
    X_flat = flatten_calibration_tensor(X).float()
    D_flat = flatten_calibration_tensor(D).float().to(device=X_flat.device)
    accumulator = group_accumulator.to(dtype=X_flat.dtype, device=X_flat.device)

    group_size = _validate_grouped_shapes(
        X_flat,
        D_flat,
        accumulator,
        num_groups,
    )

    token_count, output_dim = D_flat.shape
    D_grouped = D_flat.reshape(token_count, num_groups, group_size)

    weighted_grouped = (
        D_grouped
        * accumulator[:, :, None]
        / float(group_size) ** 0.5
    )

    weighted_rows = weighted_grouped.reshape(token_count, output_dim)
    result = weighted_rows.T @ X_flat

    if normalize_by_tokens:
        result = result / token_count

    return result
```

Important:

- do not divide `result` by `output_dim`;
- do not divide `result` by `num_groups`;
- do not divide `result` by `group_size` again;
- all arithmetic must be float32 after loading BF16 caches.

## 11.3. Update grouped accumulator

```python
def update_grouped_error_accumulator(
    group_accumulator: torch.Tensor,
    X: torch.Tensor,
    D: torch.Tensor,
    error: torch.Tensor,
    num_groups: int,
) -> torch.Tensor:
    X_flat = flatten_calibration_tensor(X).float()
    D_flat = flatten_calibration_tensor(D).float().to(device=X_flat.device)
    error = error.float().to(device=X_flat.device)
    accumulator = group_accumulator.to(dtype=X_flat.dtype, device=X_flat.device)

    group_size = _validate_grouped_shapes(
        X_flat,
        D_flat,
        accumulator,
        num_groups,
    )

    output_dim = D_flat.shape[1]
    if error.shape != (output_dim, X_flat.shape[1]):
        raise ValueError(
            "error must have shape "
            f"({output_dim}, {X_flat.shape[1]}), got {tuple(error.shape)}"
        )

    delta_z = X_flat @ error.T

    token_count = X_flat.shape[0]
    D_grouped = D_flat.reshape(token_count, num_groups, group_size)
    delta_grouped = delta_z.reshape(token_count, num_groups, group_size)

    contribution = (
        D_grouped * delta_grouped
    ).sum(dim=-1) / float(group_size) ** 0.5

    return accumulator + contribution
```

The output must remain shape `[T, G]`.

---

# 12. Exact runtime changes

## 12.1. Imports

In `any_precision/quantization/endloss_crosslayer_main.py`, replace:

```python
from .crosslayer_stats import compute_propagated_R, flatten_calibration_tensor, update_error_accumulator
```

with:

```python
from .crosslayer_stats import (
    compute_grouped_propagated_R,
    flatten_calibration_tensor,
    update_grouped_error_accumulator,
)
```

## 12.2. Constructor

Change the constructor signature to receive the same `num_groups` used by Hessian accumulation:

```python
def __init__(
    self,
    analyzer,
    tokens,
    signed_gradients_path: str,
    num_groups: int,
    initial_activations_cache_path: Optional[str] = None,
):
```

Store:

```python
if num_groups is None or num_groups < 1:
    raise ValueError(f"num_groups must be a positive integer, got {num_groups}")
self.num_groups = int(num_groups)
self.group_accumulator = None
```

Delete:

```python
self.c = None
```

## 12.3. Initialize accumulator

In `layer_R_provider`, replace scalar initialization:

```python
if self.c is None:
    self.c = torch.zeros(token_count, dtype=torch.float32)
```

with:

```python
if self.group_accumulator is None:
    self.group_accumulator = torch.zeros(
        token_count,
        self.num_groups,
        dtype=torch.float32,
    )
```

Also verify that every loaded signed-gradient tensor has the same token count.

## 12.4. Compute \(R\)

Replace:

```python
R = compute_propagated_R(X, D, self.c, normalize_by_tokens=False)
R = R / D.shape[-1]
```

with:

```python
R = compute_grouped_propagated_R(
    X,
    D,
    self.group_accumulator,
    self.num_groups,
    normalize_by_tokens=False,
)
```

There must be no subsequent output-row normalization.

## 12.5. Update accumulator

Replace:

```python
self.c = update_error_accumulator(self.c, X, D, error).cpu()
```

with:

```python
self.group_accumulator = update_grouped_error_accumulator(
    self.group_accumulator,
    X,
    D,
    error,
    self.num_groups,
).cpu()
```

## 12.6. Finite checks and logging

Replace scalar finite checks with:

```python
if torch.isfinite(self.group_accumulator).logical_not().any():
    raise ValueError(f"Non-finite EndLoss group accumulator after layer {layer_idx}")
```

Log both global and per-group magnitudes:

```python
group_max_abs = self.group_accumulator.abs().amax(dim=0)
logging.info(
    f"[Layer {layer_idx}] group accumulator "
    f"mean={self.group_accumulator.mean().item():.4e}, "
    f"std={self.group_accumulator.std().item():.4e}, "
    f"max_abs={self.group_accumulator.abs().max().item():.4e}, "
    f"per_group_max_abs={group_max_abs.tolist()}"
)
```

## 12.7. Runtime construction

Replace:

```python
runtime = CrossLayerPropagationRuntime(
    analyzer,
    tokens,
    signed_gradients_path,
    initial_activations_cache_path,
)
```

with:

```python
runtime = CrossLayerPropagationRuntime(
    analyzer,
    tokens,
    signed_gradients_path,
    num_groups,
    initial_activations_cache_path,
)
```

`num_groups` must be the exact same value passed to:

```python
accumulate_saliency_weighted_hessians(..., num_groups, ...)
```

---

# 13. Synchronizing \(X\), \(D\), and \(H\)

The new formulas assume that \(X_l\), \(D_l\), and \(H_{l,g}\) refer to the same full-precision Taylor expansion point and the same calibration-token ordering.

## 13.1. Required invariants

For each module:

- `X` is captured from the full-precision module input;
- `D` is the signed output gradient captured from the same full-precision model;
- Hessian saliency is formed from the square of that same signed gradient before final group merging;
- calibration samples are in the same order;
- sequence tokens are flattened in the same order;
- no token shuffle is allowed between gradient, Hessian, and propagation passes;
- the final `num_groups` is identical for Hessian and grouped propagation.

## 13.2. Do not change \(X\) to quantized activations

This method remains a Taylor/empirical-Fisher approximation around the full-precision model. Therefore `X` used by \(H\), \(R\), and accumulator updates remains the full-precision activation used by the current GuideQuant pipeline.

Using quantized-prefix activations would define a different approximation and is outside this change.

## 13.3. Cache compatibility

The existing raw signed-gradient cache can be reused because it stores `D` before grouping.

The existing Hessian cache can be reused if and only if all of the following are unchanged:

- model;
- dataset;
- calibration samples;
- sequence length;
- final `num_groups`;
- saliency collection scale.

The quantized EndLoss output cache must be deleted and recomputed because the propagated \(R\) changes.

---

# 14. Scale conventions

## 14.1. The existing `1e3` signed-gradient scale

The gradient collector currently stores:

```python
signed_grad = grad.float() * 1e3
```

Keep this behavior unchanged.

If \(D'=10^3D\), then:

\[
H'\propto (D')^2=10^6H.
\]

Each previous group contribution scales as:

\[
a'\propto D'=10^3a.
\]

The current \(R\) multiplies the accumulator by another \(D'\):

\[
R'\propto10^6R.
\]

Therefore \(H\) and \(R\) remain on the same scale.

Do not add or remove a `1e3` factor only in the grouped propagation path.

## 14.2. Token sum versus token mean

Current GuideQuant Hessian code accumulates a token sum:

\[
H=\sum_t\cdots.
\]

Therefore grouped \(R\) must also use a token sum:

```python
normalize_by_tokens=False
```

The per-token accumulator is not divided by `T`.

If Hessian accumulation is changed in the future to:

\[
H=\frac1T\sum_t\cdots,
\]

then `compute_grouped_propagated_R` must use:

```python
normalize_by_tokens=True
```

The accumulator update itself remains per-token and is still not divided by `T`.

---

# 15. Required tests

Implementation must follow test-first development. Add the tests below before changing runtime behavior.

At the top of `tests/test_crosslayer_stats.py`, add:

```python
import pytest
```

Keep the existing `load_crosslayer_stats()` helper and the flattening test. Remove or replace tests that call the old scalar APIs.

## 15.1. Grouped accumulator matches explicit formula

File: `tests/test_crosslayer_stats.py`

```python
def test_grouped_accumulator_matches_explicit_group_formula():
    stats = load_crosslayer_stats()

    X = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0]]],
        dtype=torch.float32,
    )
    D = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 4.0, 3.0]]],
        dtype=torch.float32,
    )
    error = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    accumulator = torch.zeros(2, 2, dtype=torch.float32)

    actual = stats.update_grouped_error_accumulator(
        accumulator,
        X,
        D,
        error,
        num_groups=2,
    )

    X_flat = X.reshape(-1, 2)
    D_flat = D.reshape(-1, 4)
    delta_z = X_flat @ error.T
    expected = torch.stack(
        [
            (D_flat[:, 0:2] * delta_z[:, 0:2]).sum(dim=1) / (2.0 ** 0.5),
            (D_flat[:, 2:4] * delta_z[:, 2:4]).sum(dim=1) / (2.0 ** 0.5),
        ],
        dim=1,
    )

    assert actual.shape == (2, 2)
    assert torch.allclose(actual, expected)
```

## 15.2. Grouped \(R\) matches explicit row formula

```python
def test_grouped_propagated_r_matches_explicit_row_formula():
    stats = load_crosslayer_stats()

    X = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0]]],
        dtype=torch.float32,
    )
    D = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 4.0, 3.0]]],
        dtype=torch.float32,
    )
    accumulator = torch.tensor(
        [[5.0, 7.0], [11.0, 13.0]],
        dtype=torch.float32,
    )

    actual = stats.compute_grouped_propagated_R(
        X,
        D,
        accumulator,
        num_groups=2,
        normalize_by_tokens=False,
    )

    X_flat = X.reshape(-1, 2)
    D_flat = D.reshape(-1, 4)
    expected_rows = []
    for row_idx in range(4):
        group_idx = row_idx // 2
        weighted = (
            D_flat[:, row_idx]
            * accumulator[:, group_idx]
            / (2.0 ** 0.5)
        )
        expected_rows.append(weighted @ X_flat)
    expected = torch.stack(expected_rows, dim=0)

    assert actual.shape == (4, 2)
    assert torch.allclose(actual, expected)
```

## 15.3. Zero accumulator gives zero \(R\)

```python
def test_zero_group_accumulator_gives_zero_r():
    stats = load_crosslayer_stats()
    X = torch.randn(2, 3, 5)
    D = torch.randn(2, 3, 8)
    accumulator = torch.zeros(6, 4)

    R = stats.compute_grouped_propagated_R(
        X,
        D,
        accumulator,
        num_groups=4,
        normalize_by_tokens=False,
    )

    assert torch.equal(R, torch.zeros_like(R))
```

## 15.4. Local \(R\) equals explicit cross-layer inner product

This test verifies the central identity:

\[
\sum_jr_{l,j}^\top e_{l,j}
=
\sum_{t,g}\mathcal C_l[t,g]a_{l,g}[t].
\]

```python
def test_r_dot_error_equals_explicit_group_cross_term():
    stats = load_crosslayer_stats()
    torch.manual_seed(0)

    T, d_in, d_out, groups = 5, 3, 8, 4
    X = torch.randn(T, d_in)
    D = torch.randn(T, d_out)
    accumulator = torch.randn(T, groups)
    error = torch.randn(d_out, d_in)

    R = stats.compute_grouped_propagated_R(
        X,
        D,
        accumulator,
        num_groups=groups,
        normalize_by_tokens=False,
    )
    actual = (R * error).sum()

    group_size = d_out // groups
    delta_z = X @ error.T
    D_grouped = D.reshape(T, groups, group_size)
    delta_grouped = delta_z.reshape(T, groups, group_size)
    current_group_score = (
        D_grouped * delta_grouped
    ).sum(dim=-1) / (float(group_size) ** 0.5)
    expected = (accumulator * current_group_score).sum()

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
```

## 15.5. Group contribution is bounded by GuideQuant group energy

```python
def test_group_score_is_bounded_by_guidequant_group_energy():
    torch.manual_seed(1)

    T, groups, group_size = 7, 4, 6
    D_grouped = torch.randn(T, groups, group_size)
    delta_grouped = torch.randn(T, groups, group_size)

    group_score = (
        D_grouped * delta_grouped
    ).sum(dim=-1) / (float(group_size) ** 0.5)

    saliency = D_grouped.pow(2).mean(dim=-1)
    guidequant_energy = saliency * delta_grouped.pow(2).sum(dim=-1)

    assert torch.all(group_score.pow(2) <= guidequant_energy + 1e-5)
```

## 15.6. Different module widths use symmetric square-root scaling

This test ensures a previous module with group size \(m_p\) and current module with group size \(m_c\) produce the intended coefficient:

\[
1/\sqrt{m_pm_c}.
\]

```python
def test_different_module_group_sizes_use_symmetric_scaling():
    stats = load_crosslayer_stats()
    torch.manual_seed(2)

    T, d_in, groups = 4, 3, 2

    X_prev = torch.randn(T, d_in)
    D_prev = torch.randn(T, 4)   # previous group size = 2
    E_prev = torch.randn(4, d_in)

    accumulator = stats.update_grouped_error_accumulator(
        torch.zeros(T, groups),
        X_prev,
        D_prev,
        E_prev,
        num_groups=groups,
    )

    X_cur = torch.randn(T, d_in)
    D_cur = torch.randn(T, 6)    # current group size = 3
    E_cur = torch.randn(6, d_in)

    R_cur = stats.compute_grouped_propagated_R(
        X_cur,
        D_cur,
        accumulator,
        num_groups=groups,
        normalize_by_tokens=False,
    )
    actual = (R_cur * E_cur).sum()

    prev_delta = (X_prev @ E_prev.T).reshape(T, groups, 2)
    cur_delta = (X_cur @ E_cur.T).reshape(T, groups, 3)
    prev_score = (
        D_prev.reshape(T, groups, 2) * prev_delta
    ).sum(dim=-1) / (2.0 ** 0.5)
    cur_score = (
        D_cur.reshape(T, groups, 3) * cur_delta
    ).sum(dim=-1) / (3.0 ** 0.5)
    expected = (prev_score * cur_score).sum()

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
```

## 15.7. Runtime no longer scales by output rows

Update `load_runtime_module()` in `tests/test_endloss_crosslayer_runtime_tokens.py` so its stub module exposes:

```python
modules["any_precision.quantization.crosslayer_stats"].compute_grouped_propagated_R = (
    lambda *args, **kwargs: None
)
modules["any_precision.quantization.crosslayer_stats"].update_grouped_error_accumulator = (
    lambda *args, **kwargs: None
)
```

The old scalar stub names are no longer sufficient after the production import changes.

Replace the old runtime test named:

```python
test_layer_r_provider_keeps_token_sum_and_scales_by_output_rows
```

with a grouped-runtime test:

```python
def test_layer_r_provider_uses_grouped_statistics_without_extra_row_scaling(monkeypatch):
    runtime_module = load_runtime_module()
    runtime = runtime_module.CrossLayerPropagationRuntime.__new__(
        runtime_module.CrossLayerPropagationRuntime
    )
    runtime.current_layer_idx = 0
    runtime.current_inputs = {"proj": torch.ones(1, 2, 3)}
    runtime.current_signed = {"proj": torch.ones(1, 2, 4)}
    runtime.num_groups = 2
    runtime.group_accumulator = torch.ones(2, 2)

    calls = []

    def fake_compute_grouped_r(
        X,
        D,
        group_accumulator,
        num_groups,
        normalize_by_tokens=False,
    ):
        calls.append((num_groups, normalize_by_tokens))
        return torch.full((D.shape[-1], X.shape[-1]), 8.0)

    monkeypatch.setattr(
        runtime_module,
        "compute_grouped_propagated_R",
        fake_compute_grouped_r,
    )

    (R,) = runtime.layer_R_provider(0, ["proj"])

    assert calls == [(2, False)]
    assert torch.equal(torch.from_numpy(R), torch.full((4, 3), 8.0))
```

## 15.8. Invalid grouping fails early

```python
def test_grouped_stats_reject_output_dim_not_divisible_by_groups():
    stats = load_crosslayer_stats()
    X = torch.randn(4, 3)
    D = torch.randn(4, 5)
    accumulator = torch.zeros(4, 2)

    with pytest.raises(ValueError, match="must be divisible"):
        stats.compute_grouped_propagated_R(
            X,
            D,
            accumulator,
            num_groups=2,
            normalize_by_tokens=False,
        )
```

---

# 16. Verification commands

Run focused tests first:

```bash
pytest -q \
  tests/test_crosslayer_stats.py \
  tests/test_endloss_crosslayer_runtime_tokens.py \
  tests/test_endloss_crosslayer_solver.py \
  tests/test_endloss_crosslayer_seed_order.py
```

Then run syntax verification:

```bash
python -m compileall \
  any_precision/quantization/crosslayer_stats.py \
  any_precision/quantization/endloss_crosslayer_main.py
```

If the repository-wide suite depends on unavailable compiled QTIP kernels, report that separately. Do not treat an unrelated QTIP import failure as a failure of these grouped-statistics tests.

---

# 17. Runtime diagnostics required for the first experiment

For every module, log before solving:

```text
R mean
R std
R max_abs
H diagonal median
H diagonal max
ratio = R.norm() / (H diagonal norm + eps)
```

After each module update, log:

```text
group_accumulator global max_abs
per-group max_abs
quantized weight max_abs
codebook max_abs
```

The first full-model run must compare:

1. GuideQuant baseline;
2. EndLoss pipeline with `group_accumulator` forced to zero;
3. new grouped propagation;
4. old scalar normalized result only as a historical reference.

The zero-accumulator run must reproduce the GuideQuant-style objective path closely enough to isolate solver-schedule differences from propagation differences.

---

# 18. Cache and execution instructions

The new method can reuse:

- tokens cache;
- signed-gradient cache;
- saliency cache;
- Hessian cache;
- GuideQuant/SqueezeLLM initialization cache;

provided all metadata matches section 13.3.

It must recompute:

- EndLoss quantized weights;
- LUT/codebook outputs;
- packed model.

Use the existing overwrite flags for the quantization and packing outputs. Do not recompute gradients or Hessians solely because the accumulator representation changed.

---

# 19. Acceptance criteria

The implementation is complete only when all of the following are true:

- [ ] runtime no longer contains `self.c`;
- [ ] runtime contains `self.group_accumulator` with shape `[T, G]`;
- [ ] `R / D.shape[-1]` is removed;
- [ ] previous-module contributions are divided by `sqrt(previous_group_size)` when inserted;
- [ ] current-module rows are divided by `sqrt(current_group_size)` when \(R\) is computed;
- [ ] `X`, `D`, and Hessian use the same token ordering and final `num_groups`;
- [ ] solver functions are unchanged;
- [ ] zero accumulator produces exactly zero \(R\);
- [ ] explicit cross-term and `R * error` agree numerically;
- [ ] different module widths pass the symmetric-scaling test;
- [ ] focused tests pass;
- [ ] compilation succeeds;
- [ ] the first experiment logs finite accumulators, finite codebooks, and finite quantized weights.

---

# 20. Final algorithm summary

For each current module \(l\):

1. Load full-precision activation \(X_l\) and signed output gradient \(D_l\).
2. Let:

   \[
   G=\texttt{num\_groups},
   \qquad
   m_l=d_{\mathrm{out}}^{(l)}/G.
   \]

3. Compute:

   \[
   \boxed{
   R_l[J_{l,g},:]
   =
   \frac1{\sqrt{m_l}}
   \left(
   D_l[:,J_{l,g}]
   \odot
   \mathcal C[:,g,None]
   \right)^\top X_l.
   }
   \]

4. Solve the unchanged local objective:

   \[
   \boxed{
   J_l(e_l)
   =
   \frac12e_l^\top H_{ll}e_l
   +\langle R_l,E_l\rangle_F.
   }
   \]

5. Form quantization error:

   \[
   E_l=\widehat W_l-W_l.
   \]

6. Compute:

   \[
   \Delta Z_l=X_lE_l^\top.
   \]

7. Update each group:

   \[
   \boxed{
   \mathcal C[:,g]
   \leftarrow
   \mathcal C[:,g]
   +
   \frac1{\sqrt{m_l}}
   \operatorname{rowsum}
   \left(
   D_l[:,J_{l,g}]
   \odot
   \Delta Z_l[:,J_{l,g}]
   \right).
   }
   \]

8. Continue to the next module in the existing sequential order.

This is the only algorithmic change specified by this document.
