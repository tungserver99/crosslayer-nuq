from typing import Dict

import torch


def flatten_calibration_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Flatten calibration examples and sequence tokens into one token axis."""
    if tensor.dim() == 2:
        return tensor
    if tensor.dim() != 3:
        raise ValueError(f"Expected a 2D or 3D calibration tensor, got shape {tuple(tensor.shape)}")
    return tensor.reshape(-1, tensor.shape[-1])


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
        raise ValueError(f"X and D token counts differ: {X_flat.shape[0]} vs {D_flat.shape[0]}")
    if group_accumulator.shape != (X_flat.shape[0], num_groups):
        raise ValueError(
            "group_accumulator must have shape "
            f"({X_flat.shape[0]}, {num_groups}), got {tuple(group_accumulator.shape)}"
        )
    output_dim = D_flat.shape[1]
    if output_dim % num_groups != 0:
        raise ValueError(f"output_dim {output_dim} must be divisible by num_groups {num_groups}")
    return output_dim // num_groups


def compute_grouped_propagated_R(
    X: torch.Tensor,
    D: torch.Tensor,
    group_accumulator: torch.Tensor,
    num_groups: int,
    normalize_by_tokens: bool = False,
) -> torch.Tensor:
    """Return R with shape [d_out, d_in] using GuideQuant-consistent groups."""
    X_flat = flatten_calibration_tensor(X).float()
    D_flat = flatten_calibration_tensor(D).float().to(device=X_flat.device)
    accumulator = group_accumulator.to(dtype=X_flat.dtype, device=X_flat.device)

    group_size = _validate_grouped_shapes(X_flat, D_flat, accumulator, num_groups)

    token_count, output_dim = D_flat.shape
    D_grouped = D_flat.reshape(token_count, num_groups, group_size)
    weighted_grouped = D_grouped * accumulator[:, :, None] / float(group_size) ** 0.5
    weighted_rows = weighted_grouped.reshape(token_count, output_dim)
    result = weighted_rows.T @ X_flat

    if normalize_by_tokens:
        result = result / token_count

    return result


def update_grouped_error_accumulator(
    group_accumulator: torch.Tensor,
    X: torch.Tensor,
    D: torch.Tensor,
    error: torch.Tensor,
    num_groups: int,
) -> torch.Tensor:
    """Return updated propagation accumulator with shape [T, num_groups]."""
    X_flat = flatten_calibration_tensor(X).float()
    D_flat = flatten_calibration_tensor(D).float().to(device=X_flat.device)
    error = error.float().to(device=X_flat.device)
    accumulator = group_accumulator.to(dtype=X_flat.dtype, device=X_flat.device)

    group_size = _validate_grouped_shapes(X_flat, D_flat, accumulator, num_groups)

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
    contribution = (D_grouped * delta_grouped).sum(dim=-1) / float(group_size) ** 0.5

    return accumulator + contribution


def compute_propagated_R(
    X: torch.Tensor,
    D: torch.Tensor,
    c: torch.Tensor,
    normalize_by_tokens: bool = True,
) -> torch.Tensor:
    X_flat = flatten_calibration_tensor(X).float()
    D_flat = flatten_calibration_tensor(D).float()
    c_flat = c.reshape(-1).to(dtype=X_flat.dtype, device=X_flat.device)

    if X_flat.shape[0] != D_flat.shape[0]:
        raise ValueError(f"X and D token counts differ: {X_flat.shape[0]} vs {D_flat.shape[0]}")
    if c_flat.shape[0] != X_flat.shape[0]:
        raise ValueError(f"Accumulator token count differs: {c_flat.shape[0]} vs {X_flat.shape[0]}")

    D_flat = D_flat.to(device=X_flat.device)
    result = (D_flat * c_flat[:, None]).T @ X_flat
    if normalize_by_tokens:
        result = result / X_flat.shape[0]
    return result


def update_error_accumulator(
    c: torch.Tensor,
    X: torch.Tensor,
    D: torch.Tensor,
    error: torch.Tensor,
) -> torch.Tensor:
    X_flat = flatten_calibration_tensor(X).float()
    D_flat = flatten_calibration_tensor(D).float().to(device=X_flat.device)
    error = error.float().to(device=X_flat.device)
    c_flat = c.reshape(-1).to(dtype=X_flat.dtype, device=X_flat.device)

    if X_flat.shape[0] != D_flat.shape[0]:
        raise ValueError(f"X and D token counts differ: {X_flat.shape[0]} vs {D_flat.shape[0]}")
    if c_flat.shape[0] != X_flat.shape[0]:
        raise ValueError(f"Accumulator token count differs: {c_flat.shape[0]} vs {X_flat.shape[0]}")
    if error.shape[1] != X_flat.shape[1]:
        raise ValueError(f"Error input dimension differs: {error.shape[1]} vs {X_flat.shape[1]}")
    if error.shape[0] != D_flat.shape[1]:
        raise ValueError(f"Error output dimension differs: {error.shape[0]} vs {D_flat.shape[1]}")

    delta_z = X_flat @ error.T
    return c_flat + (D_flat * delta_z).sum(dim=1)


def load_layer_signed_gradients(path: str) -> Dict[str, torch.Tensor]:
    return torch.load(path, map_location="cpu")
