import importlib.util
from pathlib import Path

import pytest
import torch


def load_crosslayer_stats():
    module_path = Path(__file__).resolve().parents[1] / "any_precision" / "quantization" / "crosslayer_stats.py"
    spec = importlib.util.spec_from_file_location("crosslayer_stats", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_flatten_calibration_tensor_preserves_feature_dimension():
    crosslayer_stats = load_crosslayer_stats()
    tensor = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)

    flattened = crosslayer_stats.flatten_calibration_tensor(tensor)

    assert flattened.shape == (6, 4)
    assert torch.equal(flattened[0], tensor[0, 0])
    assert torch.equal(flattened[-1], tensor[-1, -1])


def test_grouped_accumulator_matches_explicit_group_formula():
    stats = load_crosslayer_stats()
    X = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)
    D = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 4.0, 3.0]]], dtype=torch.float32)
    error = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 1.0]], dtype=torch.float32)
    accumulator = torch.zeros(2, 2, dtype=torch.float32)

    actual = stats.update_grouped_error_accumulator(accumulator, X, D, error, num_groups=2)

    X_flat = X.reshape(-1, 2)
    D_flat = D.reshape(-1, 4)
    delta_z = X_flat @ error.T
    expected = torch.stack([
        (D_flat[:, 0:2] * delta_z[:, 0:2]).sum(dim=1) / (2.0 ** 0.5),
        (D_flat[:, 2:4] * delta_z[:, 2:4]).sum(dim=1) / (2.0 ** 0.5),
    ], dim=1)

    assert actual.shape == (2, 2)
    assert torch.allclose(actual, expected)


def test_grouped_propagated_r_matches_explicit_row_formula():
    stats = load_crosslayer_stats()
    X = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32)
    D = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 4.0, 3.0]]], dtype=torch.float32)
    accumulator = torch.tensor([[5.0, 7.0], [11.0, 13.0]], dtype=torch.float32)

    actual = stats.compute_grouped_propagated_R(X, D, accumulator, num_groups=2, normalize_by_tokens=False)

    X_flat = X.reshape(-1, 2)
    D_flat = D.reshape(-1, 4)
    expected_rows = []
    for row_idx in range(4):
        group_idx = row_idx // 2
        weighted = D_flat[:, row_idx] * accumulator[:, group_idx] / (2.0 ** 0.5)
        expected_rows.append(weighted @ X_flat)
    expected = torch.stack(expected_rows, dim=0)

    assert actual.shape == (4, 2)
    assert torch.allclose(actual, expected)


def test_zero_group_accumulator_gives_zero_r():
    stats = load_crosslayer_stats()
    X = torch.randn(2, 3, 5)
    D = torch.randn(2, 3, 8)
    accumulator = torch.zeros(6, 4)

    R = stats.compute_grouped_propagated_R(X, D, accumulator, num_groups=4, normalize_by_tokens=False)

    assert torch.equal(R, torch.zeros_like(R))


def test_r_dot_error_equals_explicit_group_cross_term():
    stats = load_crosslayer_stats()
    torch.manual_seed(0)
    T, d_in, d_out, groups = 5, 3, 8, 4
    X = torch.randn(T, d_in)
    D = torch.randn(T, d_out)
    accumulator = torch.randn(T, groups)
    error = torch.randn(d_out, d_in)

    R = stats.compute_grouped_propagated_R(X, D, accumulator, num_groups=groups, normalize_by_tokens=False)
    actual = (R * error).sum()

    group_size = d_out // groups
    delta_z = X @ error.T
    D_grouped = D.reshape(T, groups, group_size)
    delta_grouped = delta_z.reshape(T, groups, group_size)
    current_group_score = (D_grouped * delta_grouped).sum(dim=-1) / (float(group_size) ** 0.5)
    expected = (accumulator * current_group_score).sum()

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_group_score_is_bounded_by_guidequant_group_energy():
    torch.manual_seed(1)
    T, groups, group_size = 7, 4, 6
    D_grouped = torch.randn(T, groups, group_size)
    delta_grouped = torch.randn(T, groups, group_size)

    group_score = (D_grouped * delta_grouped).sum(dim=-1) / (float(group_size) ** 0.5)
    saliency = D_grouped.pow(2).mean(dim=-1)
    guidequant_energy = saliency * delta_grouped.pow(2).sum(dim=-1)

    assert torch.all(group_score.pow(2) <= guidequant_energy + 1e-5)


def test_different_module_group_sizes_use_symmetric_scaling():
    stats = load_crosslayer_stats()
    torch.manual_seed(2)
    T, d_in, groups = 4, 3, 2
    X_prev = torch.randn(T, d_in)
    D_prev = torch.randn(T, 4)
    E_prev = torch.randn(4, d_in)

    accumulator = stats.update_grouped_error_accumulator(torch.zeros(T, groups), X_prev, D_prev, E_prev, num_groups=groups)

    X_cur = torch.randn(T, d_in)
    D_cur = torch.randn(T, 6)
    E_cur = torch.randn(6, d_in)

    R_cur = stats.compute_grouped_propagated_R(X_cur, D_cur, accumulator, num_groups=groups, normalize_by_tokens=False)
    actual = (R_cur * E_cur).sum()

    prev_delta = (X_prev @ E_prev.T).reshape(T, groups, 2)
    cur_delta = (X_cur @ E_cur.T).reshape(T, groups, 3)
    prev_score = (D_prev.reshape(T, groups, 2) * prev_delta).sum(dim=-1) / (2.0 ** 0.5)
    cur_score = (D_cur.reshape(T, groups, 3) * cur_delta).sum(dim=-1) / (3.0 ** 0.5)
    expected = (prev_score * cur_score).sum()

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_grouped_stats_reject_output_dim_not_divisible_by_groups():
    stats = load_crosslayer_stats()
    X = torch.randn(4, 3)
    D = torch.randn(4, 5)
    accumulator = torch.zeros(4, 2)

    with pytest.raises(ValueError, match="must be divisible"):
        stats.compute_grouped_propagated_R(X, D, accumulator, num_groups=2, normalize_by_tokens=False)
