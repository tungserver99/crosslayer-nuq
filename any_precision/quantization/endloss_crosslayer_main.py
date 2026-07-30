import datetime
import logging
import os
import shutil
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch
from tqdm.auto import trange

from ..analyzer import get_analyzer
from .activations import accumulate_saliency_weighted_hessians, get_inps
from .config import *
from .crosslayer_stats import (
    compute_grouped_propagated_R,
    flatten_calibration_tensor,
    update_grouped_error_accumulator,
)
from .gradients import load_signed_gradient_layer, signed_gradient_cache_complete
from .datautils import get_tokens
from .endloss_crosslayer_quantize import seed as endloss_crosslayer_seed
from .pack import pack

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _prepare_calibration_batches(tokens):
    batches = []
    for token in tokens:
        if token.dim() == 1:
            batches.append(token.unsqueeze(0))
        elif token.dim() == 2:
            batches.append(token)
        else:
            raise ValueError(f"Expected calibration token tensor with 1 or 2 dims, got shape {tuple(token.shape)}")
    return batches


class CrossLayerPropagationRuntime:
    def __init__(
        self,
        analyzer,
        tokens,
        signed_gradients_path: str,
        num_groups: int,
        initial_activations_cache_path: Optional[str] = None,
    ):
        if num_groups is None or num_groups < 1:
            raise ValueError(f"num_groups must be a positive integer, got {num_groups}")
        self.analyzer = analyzer
        self.tokens = tokens
        self.calibration_batches = _prepare_calibration_batches(tokens)
        self.signed_gradients_path = signed_gradients_path
        self.num_groups = int(num_groups)
        self.layers = analyzer.get_layers()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.devices = [self.device]
        if initial_activations_cache_path is not None and os.path.exists(initial_activations_cache_path):
            logging.info(f"Using cached EndLoss initial activations from {initial_activations_cache_path}; skipping get_inps.")
            cached = torch.load(initial_activations_cache_path, map_location="cpu")
            self.inps = cached["inps"]
            self.forward_args = cached["forward_args"]
        else:
            logging.info("No cached EndLoss initial activations found; calling get_inps once for EndLoss runtime.")
            self.inps, self.forward_args = get_inps(
                analyzer=analyzer,
                data=self.calibration_batches,
                model_seqlen=self.calibration_batches[0].shape[-1],
                devices=self.devices,
                offload_activations=True,
            )
            if initial_activations_cache_path is not None:
                logging.info(f"Saving EndLoss initial activations to {initial_activations_cache_path}")
                os.makedirs(os.path.dirname(initial_activations_cache_path), exist_ok=True)
                torch.save({"inps": [inp.cpu() for inp in self.inps], "forward_args": self.forward_args}, initial_activations_cache_path)
        self.outs = [torch.zeros_like(inp) for inp in self.inps]
        self.group_accumulator = None
        self.current_inputs: Dict[str, torch.Tensor] = {}
        self.current_signed: Dict[str, torch.Tensor] = {}
        self.current_layer_idx = None

    def _process_forward_args(self):
        processed_args = {}
        for key, value in self.forward_args.items():
            if isinstance(value, torch.Tensor):
                processed_args[key] = value.to(self.device, non_blocking=True)
            elif isinstance(value, tuple) and all(isinstance(item, torch.Tensor) for item in value):
                processed_args[key] = tuple(item.to(self.device, non_blocking=True) for item in value)
            else:
                processed_args[key] = value
        return processed_args

    def _capture_layer_inputs_and_outputs(self, layer_idx: int, module_names: List[str]) -> Dict[str, torch.Tensor]:
        capture_start = time.perf_counter()
        layer = self.layers[layer_idx].to(self.device)
        captured = {name: [] for name in module_names}
        hooks = []

        for module_name in module_names:
            module = layer.get_submodule(module_name)

            def make_hook(name):
                def hook(module, inp):
                    captured[name].append(inp[0].detach().cpu().to(torch.bfloat16))
                return hook

            hooks.append(module.register_forward_pre_hook(make_hook(module_name)))

        forward_args = self._process_forward_args()
        with torch.no_grad():
            for sample_idx in trange(len(self.inps[0]), desc=f"Capturing EndLoss X for layer {layer_idx}", leave=False):
                local_inp = self.inps[0][sample_idx].to(self.device).unsqueeze(0)
                out_batch = layer(local_inp, **forward_args)[0]
                self.outs[0][sample_idx].copy_(out_batch.reshape_as(self.outs[0][sample_idx]), non_blocking=True)

        for hook in hooks:
            hook.remove()

        result = {name: torch.cat(chunks, dim=0) for name, chunks in captured.items()}
        logging.info(f"[Layer {layer_idx}] Captured EndLoss X/outs in {time.perf_counter() - capture_start:.2f}s")
        return result

    def layer_R_provider(self, layer_idx: int, module_names: List[str]) -> List[torch.Tensor]:
        if self.current_layer_idx != layer_idx or not self.current_inputs:
            all_module_names = self.analyzer.module_names
            self.current_inputs = self._capture_layer_inputs_and_outputs(layer_idx, all_module_names)
            load_start = time.perf_counter()
            self.current_signed = load_signed_gradient_layer(self.signed_gradients_path, layer_idx, all_module_names)
            logging.info(f"[Layer {layer_idx}] Loaded signed gradients in {time.perf_counter() - load_start:.2f}s")
            self.current_layer_idx = layer_idx

        first_module = module_names[0]
        token_count = flatten_calibration_tensor(self.current_signed[first_module]).shape[0]
        for signed_name, signed in self.current_signed.items():
            signed_token_count = flatten_calibration_tensor(signed).shape[0]
            if signed_token_count != token_count:
                raise ValueError(
                    f"Signed-gradient token count for {signed_name} differs: "
                    f"{signed_token_count} vs {token_count}"
                )
        if self.group_accumulator is None:
            self.group_accumulator = torch.zeros(token_count, self.num_groups, dtype=torch.float32)
        elif self.group_accumulator.shape != (token_count, self.num_groups):
            raise ValueError(
                "EndLoss group accumulator has shape "
                f"{tuple(self.group_accumulator.shape)}, expected ({token_count}, {self.num_groups})"
            )

        layer_R = []
        for module_name in module_names:
            r_start = time.perf_counter()
            X = self.current_inputs[module_name]
            D = self.current_signed[module_name]
            R = compute_grouped_propagated_R(
                X,
                D,
                self.group_accumulator,
                self.num_groups,
                normalize_by_tokens=False,
            )
            layer_R.append(R.cpu().float().numpy())
            logging.info(
                f"[Layer {layer_idx}][{module_name}] propagated R "
                f"mean={R.mean().item():.4e}, std={R.std().item():.4e}, "
                f"max_abs={R.abs().max().item():.4e}"
            )
            logging.info(f"[Layer {layer_idx}][{module_name}] Computed propagated R GEMM in {time.perf_counter() - r_start:.2f}s")
        return layer_R

    def layer_error_callback(self, layer_idx: int, module_names: List[str], fp_weights: List, quantized_weights: List, is_last_module: bool = True):
        for module_name, fp_weight, quantized_weight in zip(module_names, fp_weights, quantized_weights):
            update_start = time.perf_counter()
            X = self.current_inputs[module_name]
            D = self.current_signed[module_name]
            error = torch.from_numpy(quantized_weight - fp_weight).float()
            self.group_accumulator = update_grouped_error_accumulator(
                self.group_accumulator,
                X,
                D,
                error,
                self.num_groups,
            ).cpu()
            quantized_weight_max_abs = torch.as_tensor(quantized_weight).float().abs().max().item()
            logging.info(
                f"[Layer {layer_idx}][{module_name}] Updated EndLoss group accumulator "
                f"in {time.perf_counter() - update_start:.2f}s; "
                f"quantized_weight_max_abs={quantized_weight_max_abs:.4e}"
            )

        if torch.isfinite(self.group_accumulator).logical_not().any():
            raise ValueError(f"Non-finite EndLoss group accumulator after layer {layer_idx}")

        group_max_abs = self.group_accumulator.abs().amax(dim=0)
        logging.info(
            f"[Layer {layer_idx}] group accumulator "
            f"mean={self.group_accumulator.mean().item():.4e}, "
            f"std={self.group_accumulator.std().item():.4e}, "
            f"max_abs={self.group_accumulator.abs().max().item():.4e}, "
            f"per_group_max_abs={group_max_abs.tolist()}"
        )
        if is_last_module:
            transition_start = time.perf_counter()
            self.current_inputs = {}
            self.current_signed = {}
            self.current_layer_idx = None
            self.inps, self.outs = self.outs, [torch.zeros_like(out) for out in self.outs]
            self.layers[layer_idx] = self.layers[layer_idx].cpu()
            torch.cuda.empty_cache()
            logging.info(f"[Layer {layer_idx}] Prepared inputs for next layer in {time.perf_counter() - transition_start:.2f}s")


def endloss_crosslayer_nuq(
        model,
        seed_precision=DEFAULT_SEED_PRECISION,
        mode='pack',
        yaml_path=None, cache_dir=DEFAULT_CACHE_DIR,
        dataset=DEFAULT_DATASET, seq_len=DEFAULT_SEQ_LEN, num_examples=DEFAULT_NUM_EXAMPLES,
        cpu_count=None,
        overwrite_tokens=False,
        overwrite_quantize=False,
        overwrite_hessians=False,
        overwrite_pack=False,
        random_state=None,
        num_groups=None,
        num_iterations=3,
        cd_cycles=4,
        sub_hessian: Optional[Tuple[int, int]] = None,
        sub_qlayer: Optional[Tuple[int, int]] = None,
        is_nosal=False,
        signed_gradients_path=None,
):
    if num_groups is None or num_groups < 1:
        raise ValueError(f"num_groups must be a positive integer, got {num_groups}")
    num_groups = int(num_groups)

    model_string = model if isinstance(model, str) else model.name_or_path
    model_name = model_string.split("/")[-1]

    initialization_cache_path = (f"{cache_dir}/quantized/"
                                 f"{model_name}-w{seed_precision}_orig{seed_precision}"
                                 f"-{dataset}_s{num_examples}_blk{seq_len}")
    tokens_cache_path = (f"{cache_dir}/tokens/"
                         f"{model_name}-{dataset}_s{num_examples}_blk{seq_len}.pt")
    saliency_cache_path = (f"{cache_dir}/saliency/"
                           f"{model_name}-{dataset}_s{num_examples}_blk{seq_len}_g{num_groups}")
    if signed_gradients_path is None:
        signed_gradients_path = (f"{cache_dir}/signed_gradients/"
                                 f"{model_name}-{dataset}_s{num_examples}_blk{seq_len}_g{num_groups}")
    hessians_cache_path = (f"{cache_dir}/hessians/"
                           f"{model_name}-{dataset}_s{num_examples}_blk{seq_len}_g{num_groups}{'_nosal' if is_nosal else ''}")
    initial_activations_cache_path = (f"{cache_dir}/endloss_crosslayer_activations/"
                                      f"{model_name}-{dataset}_s{num_examples}_blk{seq_len}.pt")
    quantized_cache_path = (f"{cache_dir}/endloss_crosslayer_quantized/"
                            f"{model_name}-w{seed_precision}-{dataset}_s{num_examples}_blk{seq_len}"
                            f"_g{num_groups}_iter{num_iterations}_cd{cd_cycles}_endlossxl{'_nosal' if is_nosal else ''}")
    model_output_path = (f"{cache_dir}/layerwise_packed/"
                         f"endlossxl-layerwise-{model_name}-w{seed_precision}-{dataset}_s{num_examples}_blk{seq_len}"
                         f"_g{num_groups}_iter{num_iterations}_cd{cd_cycles}{'_nosal' if is_nosal else ''}")

    log_dir = "logs_endloss_crosslayer"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s | %(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(f"{log_dir}/{os.path.basename(quantized_cache_path)}_{timestamp}.txt")],
    )

    assert mode in ['tokens', 'hessians', 'quantize', 'pack'], \
        "mode must be one of 'tokens', 'hessians', 'quantize', or 'pack'. Use 'pack' to run the entire pipeline."

    logging.info(f"Initialization cache path: {initialization_cache_path}")
    logging.info(f"Tokens cache path: {tokens_cache_path}")
    logging.info(f"Saliency cache path: {saliency_cache_path}")
    logging.info(f"Signed gradients cache path: {signed_gradients_path}")
    logging.info(f"Hessians cache path: {hessians_cache_path}")
    logging.info(f"EndLoss initial activations cache path: {initial_activations_cache_path}")
    logging.info(f"Quantized cache path: {quantized_cache_path}")
    logging.info(f"Model output path: {model_output_path}")

    analyzer = get_analyzer(model, yaml_path=yaml_path, include_tokenizer=True)
    module_names = analyzer.module_names

    logging.info("------------------- Get tokens -------------------")
    tokens = get_tokens(dataset, "train", analyzer.tokenizer, seq_len, num_examples, tokens_cache_path, random_state)
    if mode == 'tokens':
        return

    logging.info("------------------- Get Hessians -------------------")
    if overwrite_hessians and os.path.exists(hessians_cache_path):
        logging.info(f"Detected cached Hessians at {hessians_cache_path}. Will delete and recalculate weighted XTX.")
        shutil.rmtree(hessians_cache_path)
    if overwrite_hessians and os.path.exists(initial_activations_cache_path):
        logging.info(f"Detected cached EndLoss initial activations at {initial_activations_cache_path}. Will delete and refresh with Hessians.")
        os.remove(initial_activations_cache_path)
    from_cache = accumulate_saliency_weighted_hessians(
        analyzer, tokens, saliency_cache_path, hessians_cache_path, num_groups,
        initial_activations_cache_path=initial_activations_cache_path,
    )
    if mode == 'hessians':
        return

    analyzer = get_analyzer(model, yaml_path=yaml_path, include_tokenizer=True)
    module_names = analyzer.module_names

    if not os.path.exists(initialization_cache_path):
        raise FileNotFoundError(f"Initialization cache path {initialization_cache_path} does not exist. Run quantize.py first.")
    if not signed_gradient_cache_complete(signed_gradients_path, analyzer.get_layers(), len(tokens)):
        raise FileNotFoundError(f"Signed gradients path {signed_gradients_path} does not exist or is incomplete. Run quantize.py first.")

    logging.info("------------------- Quantize -------------------")
    if overwrite_quantize and os.path.exists(quantized_cache_path):
        logging.info(f"Detected cached EndLoss cross-layer quantization at {quantized_cache_path}. Will delete and recalculate.")
        shutil.rmtree(quantized_cache_path)

    runtime = CrossLayerPropagationRuntime(
        analyzer,
        tokens,
        signed_gradients_path,
        num_groups,
        initial_activations_cache_path,
    )
    endloss_crosslayer_seed(
        analyzer=analyzer,
        module_names=module_names,
        initialization_path=initialization_cache_path,
        hessians_path=hessians_cache_path,
        output_folder=quantized_cache_path,
        seed_precision=seed_precision,
        layer_R_provider=runtime.layer_R_provider,
        layer_error_callback=runtime.layer_error_callback,
        cpu_count=cpu_count,
        num_iterations=num_iterations,
        cd_cycles=cd_cycles,
        sub_qlayer=sub_qlayer,
    )

    if mode == 'quantize':
        return

    analyzer.drop_original_weights()
    logging.info("------------------- Pack -------------------")
    if os.path.exists(model_output_path) and os.path.isdir(model_output_path) and os.listdir(model_output_path):
        if overwrite_pack:
            logging.info(f"Model output path {model_output_path} already exists and is not empty. Will delete and re-pack.")
            shutil.rmtree(model_output_path)
        else:
            logging.info(f"Model output path {model_output_path} already exists and is not empty. Will skip packing.")
            return

    pack(
        analyzer=analyzer,
        lut_path=quantized_cache_path,
        output_model_path=model_output_path,
        seed_precision=seed_precision,
        parent_precision=seed_precision,
        cpu_count=cpu_count,
    )
    logging.info("Packing complete.")
