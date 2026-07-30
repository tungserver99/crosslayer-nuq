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

    assert torch.allclose(gradient, torch.zeros_like(gradient), atol=1e-4)


def test_codebook_update_with_r_uses_guidequant_lstsq_path(monkeypatch):
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
    original_lstsq = torch.linalg.lstsq
    call_count = {"lstsq": 0}

    def counting_lstsq(A, b):
        call_count["lstsq"] += 1
        return original_lstsq(A, b)

    monkeypatch.setattr(torch.linalg, "lstsq", counting_lstsq)

    updated = solver.update_C_with_r(w, h, labels, c, r, iteration=0)

    assert call_count["lstsq"] == 1
    assert torch.isfinite(updated).all()


def test_codebook_update_with_r_builds_reduced_target_and_guidequant_regularization(monkeypatch):
    solver = load_solver()
    w = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
    h = torch.diag(torch.tensor([4.0, 9.0, 16.0, 25.0], dtype=torch.float32)).unsqueeze(0)
    r = torch.tensor([[2.0, 3.0, 4.0, 5.0]], dtype=torch.float32)
    labels = torch.tensor([[0, 1, 0, 1]], dtype=torch.int64)
    c = torch.tensor([[-1.0, 1.0]], dtype=torch.float32)
    captured = {}

    class LstsqResult:
        pass

    def fake_lstsq(A, b):
        captured["A"] = A.detach().cpu().clone()
        captured["b"] = b.detach().cpu().clone()
        result = LstsqResult()
        result.solution = torch.zeros((A.shape[0], A.shape[2], 1), dtype=A.dtype, device=A.device)
        return result

    monkeypatch.setattr(torch.linalg, "lstsq", fake_lstsq)

    solver.update_C_with_r(w, h, labels, c, r, iteration=0)

    expected_A = torch.tensor(
        [[[2.0, 0.0], [0.0, 3.0], [4.0, 0.0], [0.0, 5.0], [1e-7 ** 0.5, 0.0], [0.0, 1e-7 ** 0.5]]],
        dtype=torch.float32,
    )
    expected_b = torch.tensor([[[1.0], [5.0], [11.0], [19.0], [0.0], [0.0]]], dtype=torch.float32)
    assert torch.allclose(captured["A"], expected_A)
    assert torch.allclose(captured["b"], expected_b)


def test_train_least_squares_with_r_updates_p_before_first_c(monkeypatch):
    solver = load_solver()
    calls = []

    def fake_update_p(W, H, labels, C, R, cd_cycles, verbose=True):
        calls.append("P")
        return labels

    def fake_update_c(W, H, labels, C, R, iteration):
        calls.append("C")
        return C

    objective_values = iter([3.0, 2.0, 1.0])

    def fake_objective(*args, **kwargs):
        return torch.tensor(next(objective_values), dtype=torch.float32)

    monkeypatch.setattr(solver, "update_P_with_r", fake_update_p)
    monkeypatch.setattr(solver, "update_C_with_r", fake_update_c)
    monkeypatch.setattr(solver, "objective_function_with_r", fake_objective)

    solver.train_least_squares_with_r(
        torch.zeros((1, 2), dtype=torch.float32).numpy(),
        torch.zeros((1, 2), dtype=torch.int64).numpy(),
        torch.zeros((1, 1), dtype=torch.float32).numpy(),
        torch.eye(2, dtype=torch.float32).unsqueeze(0).numpy(),
        torch.zeros((1, 2), dtype=torch.float32).numpy(),
        num_iterations=1,
        cd_cycles=1,
    )

    assert calls == ["P", "C"]
