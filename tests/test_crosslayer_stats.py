import importlib.util
from pathlib import Path

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


def test_compute_propagated_R_matches_explicit_sum():
    crosslayer_stats = load_crosslayer_stats()
    X = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    D = torch.tensor([[[0.5, -1.0], [2.0, 0.25]]])
    c = torch.tensor([1.5, -0.5])

    expected = (D.reshape(-1, 2) * c[:, None]).T @ X.reshape(-1, 2) / 2

    assert torch.allclose(crosslayer_stats.compute_propagated_R(X, D, c), expected)


def test_update_error_accumulator_matches_linear_gradient_identity():
    crosslayer_stats = load_crosslayer_stats()
    X = torch.tensor([[[1.0, 0.0], [0.0, 2.0]]])
    D = torch.tensor([[[1.0, 3.0], [-2.0, 0.5]]])
    error = torch.tensor([[0.25, -0.5], [1.0, 0.75]])
    c = torch.zeros(2)

    delta_z = X.reshape(-1, 2) @ error.T
    expected = (D.reshape(-1, 2) * delta_z).sum(dim=1)

    assert torch.allclose(crosslayer_stats.update_error_accumulator(c, X, D, error), expected)
