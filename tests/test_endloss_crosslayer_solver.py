import importlib.util
from pathlib import Path

import torch


def load_solver():
    module_path = Path(__file__).resolve().parents[1] / "any_precision" / "quantization" / "endloss_crosslayer_quantize.py"
    spec = importlib.util.spec_from_file_location("endloss_crosslayer_quantize", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gather_rows(C, labels):
    return torch.gather(C.unsqueeze(1).expand(-1, labels.shape[1], -1), 2, labels.long().unsqueeze(-1)).squeeze(-1)


def row_objective(w, h, r, q):
    e = q - w
    return 0.5 * e @ h @ e + r @ e


def test_cd_candidate_cost_matches_bruteforce_coordinate_argmin():
    solver = load_solver()
    w = torch.tensor([[0.2, -0.6, 1.1]], dtype=torch.float32)
    h = torch.tensor([[[2.0, 0.25, 0.0], [0.25, 1.5, -0.1], [0.0, -0.1, 1.0]]], dtype=torch.float32)
    r = torch.tensor([[0.4, -0.3, 0.2]], dtype=torch.float32)
    labels = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    c = torch.tensor([[-1.0, 0.0, 1.0]], dtype=torch.float32)
    q = gather_rows(c, labels)
    idx = 1
    e = q[0] - w[0]
    residual = h[0] @ e + r[0]

    expected_costs = []
    for value in c[0]:
        q_candidate = q[0].clone()
        q_candidate[idx] = value
        expected_costs.append(row_objective(w[0], h[0], r[0], q_candidate))
    expected = torch.stack(expected_costs).argmin()

    actual = solver.choose_codeword_by_delta_cost(q[0, idx], residual[idx], h[0, idx, idx], c[0])

    assert actual.item() == expected.item()


def test_codebook_update_with_r_makes_assignment_gradient_near_zero():
    solver = load_solver()
    w = torch.tensor([[0.2, -0.6, 1.1, 0.4]], dtype=torch.float32)
    h = torch.tensor([[[2.0, 0.1, 0.0, 0.2], [0.1, 1.7, -0.3, 0.0], [0.0, -0.3, 1.4, 0.1], [0.2, 0.0, 0.1, 1.2]]], dtype=torch.float32)
    r = torch.tensor([[0.3, -0.5, 0.2, 0.1]], dtype=torch.float32)
    labels = torch.tensor([[0, 1, 0, 1]], dtype=torch.int64)
    c = torch.tensor([[-1.0, 1.0]], dtype=torch.float32)

    updated = solver.update_C_with_r(w, h, labels, c, r, iteration=0)
    P = torch.nn.functional.one_hot(labels[0].long(), num_classes=2).float()
    q = P @ updated[0]
    gradient = P.T @ (h[0] @ (q - w[0]) + r[0])

    assert torch.allclose(gradient, torch.zeros_like(gradient), atol=1e-5)


def test_codebook_update_with_r_uses_batched_linear_solve(monkeypatch):
    solver = load_solver()
    w = torch.tensor(
        [[0.2, -0.6, 1.1, 0.4], [1.2, 0.7, -0.3, -0.9], [0.5, 0.1, -0.8, 0.6]],
        dtype=torch.float32,
    )
    h = torch.eye(4, dtype=torch.float32).unsqueeze(0)
    r = torch.tensor(
        [[0.3, -0.5, 0.2, 0.1], [-0.2, 0.4, 0.1, -0.3], [0.05, -0.1, 0.2, 0.3]],
        dtype=torch.float32,
    )
    labels = torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0], [0, 0, 1, 1]], dtype=torch.int64)
    c = torch.tensor([[-1.0, 1.0], [-0.5, 0.5], [-0.25, 0.25]], dtype=torch.float32)
    original_solve = torch.linalg.solve
    call_count = {"solve": 0}

    def counting_solve(A, b):
        call_count["solve"] += 1
        return original_solve(A, b)

    monkeypatch.setattr(torch.linalg, "solve", counting_solve)

    updated = solver.update_C_with_r(w, h, labels, c, r, iteration=0)

    assert call_count["solve"] == 1
    for row_idx in range(w.shape[0]):
        P = torch.nn.functional.one_hot(labels[row_idx].long(), num_classes=2).float()
        q = P @ updated[row_idx]
        gradient = P.T @ (h[0] @ (q - w[row_idx]) + r[row_idx])
        assert torch.allclose(gradient, torch.zeros_like(gradient), atol=1e-5)
