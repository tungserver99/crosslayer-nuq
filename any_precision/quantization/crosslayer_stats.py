from typing import Dict

import torch


def flatten_calibration_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Flatten calibration examples and sequence tokens into one token axis."""
    if tensor.dim() == 2:
        return tensor
    if tensor.dim() != 3:
        raise ValueError(f"Expected a 2D or 3D calibration tensor, got shape {tuple(tensor.shape)}")
    return tensor.reshape(-1, tensor.shape[-1])


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
