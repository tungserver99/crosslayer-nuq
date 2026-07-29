import os
import torch
from tqdm import tqdm
import logging
from typing import Optional, Tuple
from .config import *
from any_precision.analyzer import dispatch_model


SIGNED_CHUNKS_DIRNAME = "chunks"
SIGNED_META_FILENAME = "meta.pt"


def _signed_chunk_dir(signed_grad_path: str, layer_idx: int) -> str:
    return os.path.join(signed_grad_path, SIGNED_CHUNKS_DIRNAME, f"l{layer_idx}")


def _signed_chunk_file(signed_grad_path: str, layer_idx: int, sample_idx: int) -> str:
    return os.path.join(_signed_chunk_dir(signed_grad_path, layer_idx), f"s{sample_idx:05d}.pt")


def _signed_buffered_chunk_file(signed_grad_path: str, layer_idx: int, chunk_idx: int) -> str:
    return os.path.join(_signed_chunk_dir(signed_grad_path, layer_idx), f"c{chunk_idx:05d}.pt")


def signed_gradient_cache_complete(signed_grad_path: Optional[str], layers, expected_samples: Optional[int] = None) -> bool:
    if signed_grad_path is None:
        return True
    if not os.path.isdir(signed_grad_path):
        return False

    old_style_complete = all(
        os.path.exists(os.path.join(signed_grad_path, f"l{i}.pt"))
        for i in range(len(layers))
    )
    if old_style_complete:
        return True

    meta_path = os.path.join(signed_grad_path, SIGNED_META_FILENAME)
    if not os.path.exists(meta_path):
        return False
    meta = torch.load(meta_path, map_location="cpu")
    num_samples = int(meta.get("num_samples", -1))
    if expected_samples is not None and num_samples != expected_samples:
        return False
    if num_samples < 0:
        return False

    if meta.get("format") == "signed_gradient_chunks_v2":
        chunks_by_layer = meta.get("chunks", {})
        return all(
            all(os.path.exists(os.path.join(_signed_chunk_dir(signed_grad_path, layer_idx), chunk_name)) for chunk_name in chunks_by_layer.get(layer_idx, []))
            for layer_idx in range(len(layers))
        )

    return all(
        all(os.path.exists(_signed_chunk_file(signed_grad_path, layer_idx, sample_idx)) for sample_idx in range(num_samples))
        for layer_idx in range(len(layers))
    )


def _flush_signed_gradient_chunks(signed_grad_path: str, signed_grad_data, layers, chunk_idx: int, start_layer, end_layer):
    written = {}
    for layer_idx, layer in enumerate(layers):
        if (start_layer is not None) and (end_layer is not None):
            if not (start_layer <= layer_idx < end_layer):
                continue

        layer_dict = {}
        has_data = False
        for module_name, chunk_list in signed_grad_data[layer_idx].items():
            if len(chunk_list) > 0:
                layer_dict[module_name] = torch.cat(chunk_list, dim=0)
                chunk_list.clear()
                has_data = True
            else:
                layer_dict[module_name] = None

        if has_data:
            os.makedirs(_signed_chunk_dir(signed_grad_path, layer_idx), exist_ok=True)
            filename = f"c{chunk_idx:05d}.pt"
            torch.save(layer_dict, os.path.join(_signed_chunk_dir(signed_grad_path, layer_idx), filename))
            written.setdefault(layer_idx, []).append(filename)
    return written


def load_signed_gradient_layer(signed_grad_path: str, layer_idx: int, module_names=None):
    old_style_file = os.path.join(signed_grad_path, f"l{layer_idx}.pt")
    if os.path.exists(old_style_file):
        return torch.load(old_style_file, map_location="cpu")

    chunk_dir = _signed_chunk_dir(signed_grad_path, layer_idx)
    if not os.path.isdir(chunk_dir):
        raise FileNotFoundError(f"Signed gradient cache not found for layer {layer_idx}: {old_style_file} or {chunk_dir}")

    meta_path = os.path.join(signed_grad_path, SIGNED_META_FILENAME)
    chunk_names = None
    if os.path.exists(meta_path):
        meta = torch.load(meta_path, map_location="cpu")
        if meta.get("format") == "signed_gradient_chunks_v2":
            chunk_names = meta.get("chunks", {}).get(layer_idx, [])

    if chunk_names is None:
        chunk_names = sorted(
            name for name in os.listdir(chunk_dir)
            if name.startswith("s") and name.endswith(".pt")
        )
    chunk_files = [os.path.join(chunk_dir, name) for name in chunk_names]
    if len(chunk_files) == 0:
        raise FileNotFoundError(f"No signed gradient chunks found in {chunk_dir}")

    per_module = None
    for chunk_file in chunk_files:
        chunk = torch.load(chunk_file, map_location="cpu")
        if per_module is None:
            names = module_names if module_names is not None else chunk.keys()
            per_module = {name: [] for name in names}
        for name in per_module:
            value = chunk.get(name)
            if value is not None:
                per_module[name].append(value)

    return {
        name: torch.cat(chunks, dim=0) if len(chunks) > 0 else None
        for name, chunks in per_module.items()
    }


def get_gradients(
        analyzer,
        input_tokens,
        save_path: Optional[str] = None,
        saliency_path: Optional[str] = None,
        signed_grad_path: Optional[str] = None,
        num_groups: Optional[int] = None,
        sub_saliency: Optional[Tuple[int, int]] = None,
        skip_save_gradients: bool = False,
):
    """
    Calculates weight gradients for the given input tokens. Optionally also calculates
    GuideQuant grouped squared output-gradient saliency and signed output gradients.
    Signed output gradients are stored before squaring and are used by the EndLoss
    cross-layer propagation flow.
    """

    layers = analyzer.get_layers()
    signed_cache_complete = signed_gradient_cache_complete(signed_grad_path, layers, len(input_tokens))

    if save_path is not None and os.path.isfile(save_path) and signed_cache_complete:
        logging.info(f"Gradients already calculated and saved at {save_path}.")
        logging.info("Loading cached gradients...")
        return torch.load(save_path)

    logging.info(f"Calculating gradients on {len(input_tokens)} tokens...")

    model = analyzer.model
    if torch.cuda.device_count() > 1:
        model = dispatch_model(model)

    model = model.bfloat16()
    model.eval()

    if model.device.type != 'cuda' and torch.cuda.device_count() == 1:
        model.cuda()

    if sub_saliency is not None:
        start_layer, end_layer = sub_saliency
    else:
        start_layer, end_layer = (None, None)

    saliency_data = None
    signed_grad_data = None
    saliency_hooks = []

    if saliency_path is not None or signed_grad_path is not None:
        if saliency_path is not None:
            saliency_data = [
                {module_name: [] for module_name in analyzer.get_modules(layer).keys()}
                for layer in layers
            ]
        if signed_grad_path is not None:
            signed_grad_data = [
                {module_name: [] for module_name in analyzer.get_modules(layer).keys()}
                for layer in layers
            ]

        def make_forward_hook(layer_idx, module_name):
            def forward_hook(module, inp, out):
                out.retain_grad()

                def grad_hook(grad):
                    bsz, seq_len, hidden_dim = grad.shape
                    signed_grad = grad.float() * 1e3

                    if signed_grad_data is not None:
                        signed_grad_data[layer_idx][module_name].append(signed_grad.bfloat16().cpu())

                    if saliency_data is not None:
                        if num_groups is None:
                            raise ValueError("num_groups must be provided when saliency_path is set")
                        group_size = hidden_dim // num_groups
                        grad_squared = signed_grad.pow(2).view(bsz, seq_len, num_groups, group_size)
                        mean_squared_grad = grad_squared.mean(dim=-1)
                        saliency_data[layer_idx][module_name].append(mean_squared_grad.bfloat16().cpu())

                out.register_hook(grad_hook)
            return forward_hook

        for layer_idx, layer in enumerate(layers):
            if (start_layer is not None) and (end_layer is not None):
                if not (start_layer <= layer_idx < end_layer):
                    continue

            for module_name, module in analyzer.get_modules(layer).items():
                h = module.register_forward_hook(make_forward_hook(layer_idx, module_name))
                saliency_hooks.append(h)

    def square_grad_hook(grad):
        return grad.pow(2)

    weight_hooks = []
    for layer_idx in layers:
        for module in analyzer.get_modules(layer_idx).values():
            weight_hooks.append(module.weight.register_hook(square_grad_hook))

    signed_chunks_by_layer = {layer_idx: [] for layer_idx in range(len(layers))}
    signed_chunk_idx = 0
    signed_samples_in_buffer = 0
    signed_flush_every = int(os.environ.get("SIGNED_GRAD_FLUSH_EVERY", "8"))
    if signed_flush_every < 1:
        raise ValueError("SIGNED_GRAD_FLUSH_EVERY must be >= 1")
    if signed_grad_path is not None:
        os.makedirs(signed_grad_path, exist_ok=True)
        logging.info(f"Signed output gradients will be flushed every {signed_flush_every} samples.")

    for sample_idx, tokens in enumerate(tqdm(input_tokens, desc="Calculating gradients")):
        tokens = tokens.to(model.device).unsqueeze(0)
        outputs = model(input_ids=tokens, labels=tokens)
        loss = outputs.loss
        loss.backward()

        if signed_grad_data is not None:
            signed_samples_in_buffer += 1
            if signed_samples_in_buffer >= signed_flush_every:
                written = _flush_signed_gradient_chunks(signed_grad_path, signed_grad_data, layers, signed_chunk_idx, start_layer, end_layer)
                for layer_idx, filenames in written.items():
                    signed_chunks_by_layer[layer_idx].extend(filenames)
                signed_chunk_idx += 1
                signed_samples_in_buffer = 0

    if signed_grad_data is not None and signed_samples_in_buffer > 0:
        written = _flush_signed_gradient_chunks(signed_grad_path, signed_grad_data, layers, signed_chunk_idx, start_layer, end_layer)
        for layer_idx, filenames in written.items():
            signed_chunks_by_layer[layer_idx].extend(filenames)

    for h in weight_hooks:
        h.remove()

    for h in saliency_hooks:
        h.remove()

    model.cpu()

    gradients = []
    for layer_idx in layers:
        gradients_per_layer = {}
        for module_name, module in analyzer.get_modules(layer_idx).items():
            gradients_per_layer[module_name] = module.weight.grad
        gradients.append(gradients_per_layer)

    if saliency_path is not None:
        logging.info(f"Saving saliency files to {saliency_path}...")
        os.makedirs(saliency_path, exist_ok=True)

        for layer_idx, layer in enumerate(layers):
            if (start_layer is not None) and (end_layer is not None):
                if not (start_layer <= layer_idx < end_layer):
                    continue

            layer_dict = {}
            for module_name, chunk_list in saliency_data[layer_idx].items():
                layer_dict[module_name] = torch.cat(chunk_list, dim=0) if len(chunk_list) > 0 else None

            filename = os.path.join(saliency_path, f"l{layer_idx}.pt")
            if os.path.exists(filename):
                input(f"[WARNING] File {filename} already exists. Press Enter to overwrite or Ctrl+C to cancel.")
            torch.save(layer_dict, filename)

    if signed_grad_path is not None:
        logging.info(f"Saving signed output-gradient chunk metadata to {signed_grad_path}...")
        torch.save(
            {
                "format": "signed_gradient_chunks_v2",
                "num_samples": len(input_tokens),
                "flush_every": signed_flush_every,
                "chunks": signed_chunks_by_layer,
            },
            os.path.join(signed_grad_path, SIGNED_META_FILENAME),
        )

    if save_path is not None and not skip_save_gradients:
        logging.info(f"Saving gradients to {save_path}...")
        if not save_path.endswith('.pt'):
            save_path = save_path + '.pt'
        if os.path.exists(save_path):
            input(f"[WARNING] File {save_path} already exists. Press Enter to overwrite or Ctrl+C to cancel.")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(gradients, save_path)

    return gradients
