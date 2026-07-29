import importlib.util
from pathlib import Path

import numpy as np


def load_solver():
    module_path = Path(__file__).resolve().parents[1] / "any_precision" / "quantization" / "endloss_crosslayer_quantize.py"
    spec = importlib.util.spec_from_file_location("endloss_crosslayer_quantize", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeAnalyzer:
    num_layers = 1


def test_seed_quantizes_and_updates_accumulator_per_linear_module(monkeypatch, tmp_path):
    solver = load_solver()
    module_names = ["a", "b"]
    order = []

    monkeypatch.setattr(solver, "_load_progress", lambda *args: ([0], []))
    monkeypatch.setattr(
        solver,
        "_load_layer",
        lambda *args: (
            module_names,
            [np.array([[1.0, 2.0]], dtype=np.float32), np.array([[3.0, 4.0]], dtype=np.float32)],
            [np.array([[0, 0]], dtype=np.uint8), np.array([[0, 0]], dtype=np.uint8)],
            [np.array([[1.0]], dtype=np.float32), np.array([[3.0]], dtype=np.float32)],
            [np.eye(2, dtype=np.float32)[None, :, :], np.eye(2, dtype=np.float32)[None, :, :]],
        ),
    )

    def fake_seed_layer(l, names, weights, labels, centroids, hessians, layer_R, seed_bit, group_count, num_iterations, cd_cycles):
        order.append(("seed", tuple(names)))
        quantized = [weights[0] + 0.25]
        return [[centroids[0].reshape(1, 1, 1)]], [labels[0].reshape(1, 1, 2)], [{"module": names[0]}], quantized

    def fake_provider(layer_idx, names):
        order.append(("provider", tuple(names)))
        return [np.zeros((1, 2), dtype=np.float32)]

    def fake_callback(layer_idx, names, fp_weights, quantized_weights, is_last_module=True):
        order.append(("callback", tuple(names), is_last_module))

    saved = []
    monkeypatch.setattr(solver, "seed_layer", fake_seed_layer)
    monkeypatch.setattr(solver, "_save_results", lambda *args: saved.append(args))

    solver.seed(
        analyzer=FakeAnalyzer(),
        module_names=module_names,
        initialization_path="init",
        hessians_path="hess",
        output_folder=str(tmp_path),
        seed_precision=1,
        layer_R_provider=fake_provider,
        layer_error_callback=fake_callback,
        cpu_count=1,
        num_iterations=1,
        cd_cycles=1,
    )

    assert order == [
        ("provider", ("a",)),
        ("seed", ("a",)),
        ("callback", ("a",), False),
        ("provider", ("b",)),
        ("seed", ("b",)),
        ("callback", ("b",), True),
    ]
    assert len(saved) == 1
