import importlib.util
import sys
import types
from pathlib import Path

import torch


def load_gradients_module():
    repo_root = Path(__file__).resolve().parents[1]
    package = types.ModuleType("any_precision")
    package.__path__ = [str(repo_root / "any_precision")]
    quant_package = types.ModuleType("any_precision.quantization")
    quant_package.__path__ = [str(repo_root / "any_precision" / "quantization")]
    analyzer = types.ModuleType("any_precision.analyzer")
    analyzer.dispatch_model = lambda model: model
    config = types.ModuleType("any_precision.quantization.config")

    old_modules = {name: sys.modules.get(name) for name in [
        "any_precision",
        "any_precision.quantization",
        "any_precision.analyzer",
        "any_precision.quantization.config",
    ]}
    sys.modules["any_precision"] = package
    sys.modules["any_precision.quantization"] = quant_package
    sys.modules["any_precision.analyzer"] = analyzer
    sys.modules["any_precision.quantization.config"] = config

    try:
        module_path = repo_root / "any_precision" / "quantization" / "gradients.py"
        spec = importlib.util.spec_from_file_location("any_precision.quantization.gradients", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, old_module in old_modules.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


def test_signed_gradient_chunk_cache_loads_layer_in_sample_order(tmp_path):
    gradients = load_gradients_module()
    cache = tmp_path / "signed"
    cache.mkdir()
    Path(gradients._signed_chunk_dir(str(cache), 0)).mkdir(parents=True)
    torch.save({"format": "signed_gradient_chunks_v1", "num_samples": 2}, cache / gradients.SIGNED_META_FILENAME)
    torch.save({"a": torch.ones(1, 2, 3), "b": torch.full((1, 2, 1), 5.0)}, gradients._signed_chunk_file(str(cache), 0, 0))
    torch.save({"a": torch.full((1, 2, 3), 2.0), "b": torch.full((1, 2, 1), 6.0)}, gradients._signed_chunk_file(str(cache), 0, 1))

    assert gradients.signed_gradient_cache_complete(str(cache), [0], expected_samples=2)

    loaded = gradients.load_signed_gradient_layer(str(cache), 0, module_names=["a", "b"])

    assert loaded["a"].shape == (2, 2, 3)
    assert loaded["b"].shape == (2, 2, 1)
    assert torch.equal(loaded["a"][0], torch.ones(2, 3))
    assert torch.equal(loaded["a"][1], torch.full((2, 3), 2.0))
    assert torch.equal(loaded["b"].flatten(), torch.tensor([5.0, 5.0, 6.0, 6.0]))


def test_signed_gradient_chunk_cache_accepts_multi_sample_chunk_files(tmp_path):
    gradients = load_gradients_module()
    cache = tmp_path / "signed"
    cache.mkdir()
    Path(gradients._signed_chunk_dir(str(cache), 0)).mkdir(parents=True)
    torch.save(
        {"format": "signed_gradient_chunks_v2", "num_samples": 3, "chunks": {0: ["c00000.pt", "c00001.pt"]}},
        cache / gradients.SIGNED_META_FILENAME,
    )
    torch.save({"a": torch.arange(4, dtype=torch.float32).reshape(2, 1, 2)}, Path(gradients._signed_chunk_dir(str(cache), 0)) / "c00000.pt")
    torch.save({"a": torch.full((1, 1, 2), 9.0)}, Path(gradients._signed_chunk_dir(str(cache), 0)) / "c00001.pt")

    assert gradients.signed_gradient_cache_complete(str(cache), [0], expected_samples=3)

    loaded = gradients.load_signed_gradient_layer(str(cache), 0, module_names=["a"])

    assert loaded["a"].shape == (3, 1, 2)
    assert torch.equal(loaded["a"][0], torch.tensor([[0.0, 1.0]]))
    assert torch.equal(loaded["a"][2], torch.tensor([[9.0, 9.0]]))
