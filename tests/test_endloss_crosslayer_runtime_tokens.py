import importlib.util
import sys
import types
from pathlib import Path

import torch


def load_runtime_module():
    repo_root = Path(__file__).resolve().parents[1]
    package = types.ModuleType("any_precision")
    package.__path__ = [str(repo_root / "any_precision")]
    quant_package = types.ModuleType("any_precision.quantization")
    quant_package.__path__ = [str(repo_root / "any_precision" / "quantization")]

    modules = {
        "any_precision": package,
        "any_precision.quantization": quant_package,
        "any_precision.analyzer": types.ModuleType("any_precision.analyzer"),
        "any_precision.quantization.activations": types.ModuleType("any_precision.quantization.activations"),
        "any_precision.quantization.config": types.ModuleType("any_precision.quantization.config"),
        "any_precision.quantization.crosslayer_stats": types.ModuleType("any_precision.quantization.crosslayer_stats"),
        "any_precision.quantization.datautils": types.ModuleType("any_precision.quantization.datautils"),
        "any_precision.quantization.endloss_crosslayer_quantize": types.ModuleType("any_precision.quantization.endloss_crosslayer_quantize"),
        "any_precision.quantization.pack": types.ModuleType("any_precision.quantization.pack"),
        "any_precision.quantization.gradients": types.ModuleType("any_precision.quantization.gradients"),
    }
    modules["any_precision.analyzer"].get_analyzer = lambda *args, **kwargs: None
    modules["any_precision.quantization.activations"].accumulate_saliency_weighted_hessians = lambda *args, **kwargs: True
    modules["any_precision.quantization.activations"].get_inps = lambda *args, **kwargs: None
    modules["any_precision.quantization.config"].DEFAULT_SEED_PRECISION = 3
    modules["any_precision.quantization.config"].DEFAULT_CACHE_DIR = "cache"
    modules["any_precision.quantization.config"].DEFAULT_DATASET = "c4"
    modules["any_precision.quantization.config"].DEFAULT_SEQ_LEN = 2048
    modules["any_precision.quantization.config"].DEFAULT_NUM_EXAMPLES = 128
    modules["any_precision.quantization.crosslayer_stats"].compute_grouped_propagated_R = lambda *args, **kwargs: None
    modules["any_precision.quantization.crosslayer_stats"].flatten_calibration_tensor = lambda tensor: tensor.reshape(-1, tensor.shape[-1])
    modules["any_precision.quantization.crosslayer_stats"].update_grouped_error_accumulator = lambda *args, **kwargs: None
    modules["any_precision.quantization.datautils"].get_tokens = lambda *args, **kwargs: None
    modules["any_precision.quantization.endloss_crosslayer_quantize"].seed = lambda *args, **kwargs: None
    modules["any_precision.quantization.pack"].pack = lambda *args, **kwargs: None
    modules["any_precision.quantization.gradients"].load_signed_gradient_layer = lambda *args, **kwargs: None
    modules["any_precision.quantization.gradients"].signed_gradient_cache_complete = lambda *args, **kwargs: True

    old_modules = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        module_path = repo_root / "any_precision" / "quantization" / "endloss_crosslayer_main.py"
        spec = importlib.util.spec_from_file_location("any_precision.quantization.endloss_crosslayer_main", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, old_module in old_modules.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


def test_prepare_calibration_batches_unsqueezes_1d_tokens():
    runtime = load_runtime_module()
    tokens = [torch.arange(4), torch.arange(4, 8)]

    batches = runtime._prepare_calibration_batches(tokens)

    assert [batch.shape for batch in batches] == [torch.Size([1, 4]), torch.Size([1, 4])]
    assert torch.equal(batches[0][0], tokens[0])


def test_prepare_calibration_batches_keeps_2d_tokens():
    runtime = load_runtime_module()
    tokens = [torch.arange(4).reshape(1, 4)]

    batches = runtime._prepare_calibration_batches(tokens)

    assert batches[0] is tokens[0]


def test_layer_r_provider_uses_grouped_statistics_without_extra_row_scaling(monkeypatch):
    runtime_module = load_runtime_module()
    runtime = runtime_module.CrossLayerPropagationRuntime.__new__(runtime_module.CrossLayerPropagationRuntime)
    runtime.current_layer_idx = 0
    runtime.current_inputs = {"proj": torch.ones(1, 2, 3)}
    runtime.current_signed = {"proj": torch.ones(1, 2, 4)}
    runtime.num_groups = 2
    runtime.group_accumulator = torch.ones(2, 2)

    calls = []

    def fake_compute_grouped_r(X, D, group_accumulator, num_groups, normalize_by_tokens=False):
        calls.append((num_groups, normalize_by_tokens))
        return torch.full((D.shape[-1], X.shape[-1]), 8.0)

    monkeypatch.setattr(runtime_module, "compute_grouped_propagated_R", fake_compute_grouped_r)

    (R,) = runtime.layer_R_provider(0, ["proj"])

    assert calls == [(2, False)]
    assert torch.equal(torch.from_numpy(R), torch.full((4, 3), 8.0))
