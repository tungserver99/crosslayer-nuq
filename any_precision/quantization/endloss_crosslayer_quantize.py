import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

import numpy as np
import torch

try:
    from .utils import get_progress_bar
except ImportError:
    def get_progress_bar(total, desc):
        from tqdm import tqdm
        return tqdm(total=total, desc=desc)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _gather_weight(C: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return torch.gather(
        C.unsqueeze(1).expand(-1, labels.shape[1], -1),
        dim=2,
        index=labels.long().unsqueeze(-1),
    ).squeeze(-1)


def choose_codeword_by_delta_cost(
    q_i: torch.Tensor,
    residual_i: torch.Tensor,
    h_ii: torch.Tensor,
    codebook: torch.Tensor,
) -> torch.Tensor:
    delta = codebook - q_i
    cost = residual_i * delta + 0.5 * h_ii * delta.square()
    return cost.argmin()

@torch.no_grad()
def objective_function_with_r(
    W: torch.Tensor,
    H: torch.Tensor,
    labels: torch.Tensor,
    C: torch.Tensor,
    R: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    device = W.device
    labels = labels.to(device).long()
    C = C.to(device)
    H = H.to(device)
    if R is None:
        R = torch.zeros_like(W)
    else:
        R = R.to(device)

    W_hat = _gather_weight(C, labels)
    delta_w = W_hat - W
    num_groups = H.shape[0]
    group_size = W.shape[0] // num_groups
    delta_grp = delta_w.reshape(num_groups, group_size, delta_w.shape[-1])
    R_grp = R.reshape(num_groups, group_size, R.shape[-1])

    quadratic = torch.einsum("gri,gij,grj->gr", delta_grp, H, delta_grp)
    linear = (R_grp * delta_grp).sum(dim=-1)
    return (0.5 * quadratic + linear).mean()


def update_P_with_r(
    W: torch.Tensor,
    H: torch.Tensor,
    labels: torch.Tensor,
    C: torch.Tensor,
    R: Optional[torch.Tensor],
    cd_cycles: int,
    verbose: bool = True,
) -> torch.Tensor:
    device = _device()
    W = W.to(device)
    H = H.to(device)
    C = C.to(device)
    assignments_prev = labels.to(device).long()
    R = torch.zeros_like(W) if R is None else R.to(device)

    output_dim, input_dim = W.shape
    num_groups = H.shape[0]
    group_size = output_dim // num_groups
    assignments = assignments_prev.clone()
    W_hat = _gather_weight(C, assignments)

    W_grp = W.reshape(num_groups, group_size, input_dim)
    C_grp = C.reshape(num_groups, group_size, C.shape[-1])
    W_hat_grp = W_hat.reshape(num_groups, group_size, input_dim)
    R_grp = R.reshape(num_groups, group_size, input_dim)

    residual = torch.bmm(W_hat_grp - W_grp, H) + R_grp
    h_diag = torch.diagonal(H, dim1=1, dim2=2).reshape(num_groups, 1, input_dim)

    pb = get_progress_bar(cd_cycles * input_dim, "Updating P inside")
    for _ in range(cd_cycles):
        for update_idx in range(input_dim):
            q_i = W_hat_grp[:, :, update_idx].unsqueeze(-1)
            residual_i = residual[:, :, update_idx].unsqueeze(-1)
            h_ii = h_diag[:, :, update_idx].unsqueeze(-1)
            delta = C_grp - q_i
            cost = residual_i * delta + 0.5 * h_ii * delta.square()
            argmin = cost.argmin(dim=-1)
            new_q = torch.gather(C_grp, dim=-1, index=argmin.unsqueeze(-1)).squeeze(-1)
            old_q = W_hat_grp[:, :, update_idx]
            change = new_q - old_q

            assignments.reshape(num_groups, group_size, input_dim)[:, :, update_idx] = argmin
            W_hat_grp[:, :, update_idx] = new_q
            residual += change.unsqueeze(-1) * H[:, update_idx, :].unsqueeze(1)
            pb.update(1)
    pb.close()

    num_changed = (assignments_prev != assignments).sum().item()
    total_assignments = assignments_prev.numel()
    if verbose:
        logging.info(f"Percentage of assignments changed: {num_changed / total_assignments * 100:.2f}%")

    return assignments.detach().cpu()


def update_C_with_r(
    W: torch.Tensor,
    H: torch.Tensor,
    labels: torch.Tensor,
    C: torch.Tensor,
    R: Optional[torch.Tensor],
    iteration: int,
) -> torch.Tensor:
    del iteration
    device = _device()
    W = W.to(device)
    H = H.to(device)
    labels = labels.to(device).long()
    C = C.to(device)
    R = torch.zeros_like(W) if R is None else R.to(device)

    output_dim, input_dim = W.shape
    num_groups = H.shape[0]
    group_size = output_dim // num_groups
    n_cluster = C.shape[-1]
    sub_channel_size = 64
    sub_input_size = 2 ** 16
    lambda_reg = 1e-7
    result_chunks = []

    num_centroid_chunks = sum((group_size + sub_channel_size - 1) // sub_channel_size for _ in range(num_groups))
    pb = get_progress_bar(num_centroid_chunks, "Updating centroids")

    sqrt_lambda = torch.sqrt(torch.tensor(lambda_reg, dtype=W.dtype, device=device))
    reg_eye = sqrt_lambda * torch.eye(n_cluster, dtype=W.dtype, device=device)

    for group_idx in range(num_groups):
        group_start = group_idx * group_size
        group_end = group_start + group_size
        h = H[group_idx]
        L = torch.linalg.cholesky(h)
        reduced_X = L.transpose(-2, -1)
        W_group = W[group_start:group_end]
        R_group = R[group_start:group_end]
        linv_r = torch.linalg.solve_triangular(L, R_group.transpose(0, 1), upper=False).transpose(0, 1)
        reduced_target = W_group @ reduced_X.transpose(0, 1) - linv_r

        for start in range(group_start, group_end, sub_channel_size):
            end = min(start + sub_channel_size, group_end)
            local_start = start - group_start
            local_end = end - group_start
            labels_batch = labels[start:end]
            batch_size = end - start
            P_batch = torch.nn.functional.one_hot(labels_batch, num_classes=n_cluster).to(dtype=W.dtype)

            A_batch_list = []
            b_batch_list = []
            for st_idx_inp in range(0, input_dim, sub_input_size):
                end_idx_inp = min(input_dim, st_idx_inp + sub_input_size)
                X_batch = reduced_X[st_idx_inp:end_idx_inp]
                A_batch_list.append(torch.einsum("bj,ijc->ibc", X_batch, P_batch))
                b_batch_list.append(reduced_target[local_start:local_end, st_idx_inp:end_idx_inp].unsqueeze(-1))

            A_batch = torch.cat(A_batch_list, dim=1)
            b_batch = torch.cat(b_batch_list, dim=1)
            A_batch = torch.cat([A_batch, reg_eye.unsqueeze(0).expand(batch_size, -1, -1)], dim=1)
            b_batch = torch.cat(
                [b_batch, torch.zeros((batch_size, n_cluster, 1), dtype=W.dtype, device=device)],
                dim=1,
            )
            C_hat_batch = torch.linalg.lstsq(A_batch, b_batch).solution.squeeze(-1)
            result_chunks.append(C_hat_batch)
            pb.update(1)

    pb.close()
    return torch.cat(result_chunks, dim=0).detach().cpu()

def train_least_squares_with_r(
    W: np.ndarray,
    init_labels: np.ndarray,
    init_centroids: np.ndarray,
    H: np.ndarray,
    R: Optional[np.ndarray] = None,
    num_iterations: int = 3,
    cd_cycles: int = 4,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    device = _device()
    labels = torch.tensor(init_labels, dtype=torch.int64, device="cpu")
    C = torch.tensor(init_centroids, dtype=torch.float32, device="cpu")
    W_t = torch.tensor(W, dtype=torch.float32, device=device)
    H_t = torch.tensor(H, dtype=torch.float32, device=device)
    R_t = torch.zeros_like(W_t) if R is None else torch.tensor(R, dtype=torch.float32, device=device)

    diag_idx = torch.arange(H_t.shape[1], device=device)
    for group_idx in range(H_t.shape[0]):
        avg_diag = torch.mean(torch.diag(H_t[group_idx]))
        damp, prev_damp = 1e-5, 0.0
        while True:
            try:
                torch.linalg.cholesky(H_t[group_idx])
                logging.info(f"{group_idx + 1}-th H is PD, dampening factor={prev_damp:.2e}")
                break
            except RuntimeError as exc:
                logging.info(exc)
                logging.info(f"{group_idx + 1}-th H is not PD, try dampening with factor={damp:.2e}")
                H_t[group_idx, diag_idx, diag_idx] += (damp - prev_damp) * avg_diag
                prev_damp = damp
                damp *= 10
                if damp > 1e0:
                    raise

    best_obj = objective_function_with_r(W_t, H_t, labels, C, R_t).item()
    best_labels = labels.detach().cpu().clone()
    best_C = C.detach().cpu().clone()
    log_dict = {"objective": [best_obj], "iteration": [0]}
    logging.info(f"Initial objective: {best_obj:.6f}")

    start_time = time.time()
    for iteration in range(num_iterations):
        logging.info(f"Iteration {iteration + 1}: updating P assignments")
        labels = update_P_with_r(W_t, H_t, labels, C, R_t, cd_cycles=cd_cycles)

        obj_after_p = objective_function_with_r(W_t, H_t, labels, C, R_t).item()
        log_dict["objective"].append(obj_after_p)
        log_dict["iteration"].append(iteration + 1)
        logging.info(f"Iteration {iteration + 1} (P update): Objective: {obj_after_p:.4f}")

        logging.info(f"Iteration {iteration + 1}: updating C centroids")
        C = update_C_with_r(W_t, H_t, labels, C, R_t, iteration)
        current_obj = objective_function_with_r(W_t, H_t, labels, C, R_t).item()
        log_dict["objective"].append(current_obj)
        log_dict["iteration"].append(iteration + 1)
        if current_obj < best_obj:
            best_obj = current_obj
            best_labels = labels.detach().cpu().clone()
            best_C = C.detach().cpu().clone()
            logging.info(f"Iteration {iteration + 1} (C update): Objective: {current_obj:.4f} | Improved and using this one.")
        else:
            logging.info(f"Iteration {iteration + 1} (C update): Objective: {current_obj:.4f} | Not improved. Using previous best values.")
            labels, C = best_labels, best_C
            break

    logging.info(f"Least squares training time: {time.time() - start_time:.2f} seconds")
    return best_labels.numpy().astype(np.uint8), best_C.numpy().astype(np.float32), log_dict


def fix_hessian_shape(H: torch.Tensor) -> torch.Tensor:
    if H.shape[1] == H.shape[2]:
        return H
    if H.shape[0] == H.shape[1]:
        return H.permute(2, 0, 1)
    raise ValueError(f"Invalid Hessian shape: {H.shape}")


def seed_layer(
    l: int,
    module_names: List[str],
    layer_modules: List[np.ndarray],
    layer_init_labels: List[np.ndarray],
    layer_init_centroids: List[np.ndarray],
    layer_hessian: List[np.ndarray],
    layer_R: List[np.ndarray],
    seed_bit: int,
    group_count: int,
    num_iterations: int = 3,
    cd_cycles: int = 4,
) -> Tuple[List[List[np.ndarray]], List[np.ndarray], List[dict], List[np.ndarray]]:
    assert group_count == 1, "Group-wise quantization is not supported yet"
    n_cluster = 2**seed_bit
    lut_by_bit_by_module = []
    parent_weights_by_modules = []
    log_dict_by_module = []
    quantized_weights = []

    for m_idx, module_name in enumerate(module_names):
        logging.info(f"Quantizing Layer [{l}], Module [{module_name}] ({m_idx + 1}/{len(module_names)})")
        module_weight = layer_modules[m_idx]
        output_dim, input_dim = module_weight.shape
        init_labels = layer_init_labels[m_idx].reshape(output_dim, input_dim)
        init_centroids = layer_init_centroids[m_idx].reshape(output_dim, n_cluster)

        labels, C, log_dict = train_least_squares_with_r(
            module_weight.reshape(output_dim, input_dim),
            init_labels,
            init_centroids,
            layer_hessian[m_idx],
            layer_R[m_idx],
            num_iterations=num_iterations,
            cd_cycles=cd_cycles,
        )

        lut_by_bit_by_module.append([C.reshape(output_dim, 1, n_cluster)])
        parent_weights_by_modules.append(labels.reshape(output_dim, 1, input_dim).astype(np.uint8))
        log_dict_by_module.append(log_dict)
        quantized_weights.append(C[np.arange(output_dim)[:, None], labels].astype(np.float32))

    return lut_by_bit_by_module, parent_weights_by_modules, log_dict_by_module, quantized_weights


def _save_results(parent_parameters_path, seed_precision, parent_precision, module_names, luts_by_bit_by_module, parent_weights, log_dict, l):
    for i, bit in enumerate(range(seed_precision, parent_precision + 1)):
        output_lut_file_name = f"{parent_parameters_path}/lut_{bit}/l{l}.pt"
        output_log_dict_file_name = f"{parent_parameters_path}/lut_{bit}/log_dict{l}.pt"
        os.makedirs(os.path.dirname(output_lut_file_name), exist_ok=True)
        lut_dict = {}
        module_name_to_log_dict = {}
        for j in range(len(module_names)):
            lut_dict[module_names[j]] = luts_by_bit_by_module[j][i].astype(np.float16)
            module_name_to_log_dict[module_names[j]] = log_dict[j]
        torch.save(lut_dict, output_lut_file_name)
        torch.save(module_name_to_log_dict, output_log_dict_file_name)

    parent_weight_dict = {module_names[j]: parent_weights[j].astype(np.uint8) for j in range(len(module_names))}
    output_weights_layer_file_name = f"{parent_parameters_path}/weights/l{l}.pt"
    os.makedirs(os.path.dirname(output_weights_layer_file_name), exist_ok=True)
    torch.save(parent_weight_dict, output_weights_layer_file_name)


def _load_progress(parent_parameters_path, seed_precision, parent_precision, layer_count):
    todo_ran = []
    processed_ran = []
    for l in range(layer_count):
        has_luts = all(os.path.exists(f"{parent_parameters_path}/lut_{bit}/l{l}.pt") for bit in range(seed_precision, parent_precision + 1))
        has_weights = os.path.exists(f"{parent_parameters_path}/weights/l{l}.pt")
        if has_luts and has_weights:
            processed_ran.append(l)
        else:
            todo_ran.append(l)
    return todo_ran, processed_ran


def _load_layer(analyzer, module_names, initialization_path, hessians_path, seed_precision, l):
    init_labels = torch.load(os.path.join(initialization_path, "weights", f"l{l}.pt"))
    init_centroids = torch.load(os.path.join(initialization_path, f"lut_{seed_precision}", f"l{l}.pt"))
    hessian = torch.load(os.path.join(hessians_path, f"l{l}.pt"))
    return (
        module_names,
        [analyzer.get_layer_weights(l)[name].float().numpy() for name in module_names],
        [init_labels[name] for name in module_names],
        [init_centroids[name].astype(np.float32) for name in module_names],
        [fix_hessian_shape(hessian[name]).float().numpy() for name in module_names],
    )


def seed(
    analyzer,
    module_names: List[str],
    initialization_path: str,
    hessians_path: str,
    output_folder: str,
    seed_precision: int,
    layer_R_provider,
    layer_error_callback,
    cpu_count: Optional[int] = None,
    num_iterations: int = 3,
    cd_cycles: int = 4,
    sub_qlayer: Optional[Tuple[int, int]] = None,
):
    group_count = 1
    if cpu_count is None:
        cpu_count = os.cpu_count() or 1
    pipelined_io = cpu_count >= 8
    io_workers = 2 if cpu_count >= 64 else 1

    layers_to_process, completed_layers = _load_progress(output_folder, seed_precision, seed_precision, analyzer.num_layers)
    if sub_qlayer:
        layers_to_process = [i for i in layers_to_process if i in range(sub_qlayer[0], sub_qlayer[1])]
    if completed_layers and layers_to_process:
        raise RuntimeError(
            "EndLoss cross-layer quantization is sequential and cannot safely resume from a partial cache. "
            f"Found completed layers {completed_layers}; remove {output_folder} or rerun with overwrite_quantize."
        )
    if completed_layers:
        logging.info(f"The following layers will be skipped as they have already been processed:\n{completed_layers}")
    if not layers_to_process:
        logging.info("All layers have already been processed. Exiting...")
        return

    logging.info(f"Quantizing layers {layers_to_process}")

    def layer_loader(layer_idx):
        return _load_layer(analyzer, module_names, initialization_path, hessians_path, seed_precision, layer_idx)

    def run_layer(l, loaded):
        layer_start = time.perf_counter()
        logging.info(f"[Layer {l}] Starting EndLoss quantization")
        module_names_l, model_layer, init_labels_layer, init_centroids_layer, hessian_layer = loaded
        all_luts_by_bit_by_module = []
        all_parent_weights = []
        all_log_dict = []

        for m_idx, module_name in enumerate(module_names_l):
            is_last_module = m_idx == len(module_names_l) - 1
            one_name = [module_name]
            one_model = [model_layer[m_idx]]
            one_labels = [init_labels_layer[m_idx]]
            one_centroids = [init_centroids_layer[m_idx]]
            one_hessian = [hessian_layer[m_idx]]
            r_start = time.perf_counter()
            one_R = layer_R_provider(l, one_name)
            logging.info(f"[Layer {l}][{module_name}] Prepared propagated R in {time.perf_counter() - r_start:.2f}s")

            solve_start = time.perf_counter()
            luts_by_bit_by_module, parent_weights, log_dict, quantized_weights = seed_layer(
                l,
                one_name,
                one_model,
                one_labels,
                one_centroids,
                one_hessian,
                one_R,
                seed_precision,
                group_count,
                num_iterations=num_iterations,
                cd_cycles=cd_cycles,
            )
            logging.info(f"[Layer {l}][{module_name}] Finished CD/codebook solve in {time.perf_counter() - solve_start:.2f}s")
            all_luts_by_bit_by_module.extend(luts_by_bit_by_module)
            all_parent_weights.extend(parent_weights)
            all_log_dict.extend(log_dict)
            callback_start = time.perf_counter()
            layer_error_callback(l, one_name, one_model, quantized_weights, is_last_module=is_last_module)
            logging.info(f"[Layer {l}][{module_name}] Finished accumulator/callback in {time.perf_counter() - callback_start:.2f}s")

        logging.info(f"[Layer {l}] Finished EndLoss quantization in {time.perf_counter() - layer_start:.2f}s")
        return module_names_l, all_luts_by_bit_by_module, all_parent_weights, all_log_dict
    if pipelined_io:
        pending_saves = []
        with ThreadPoolExecutor(max_workers=io_workers) as io_executor:
            future_load = None
            for l in layers_to_process:
                if future_load is None:
                    future_load = io_executor.submit(layer_loader, l)
                loaded = future_load.result()
                next_idx = layers_to_process.index(l) + 1
                future_load = io_executor.submit(layer_loader, layers_to_process[next_idx]) if next_idx < len(layers_to_process) else None
                save_payload = run_layer(l, loaded)
                pending_saves.append(
                    io_executor.submit(_save_results, output_folder, seed_precision, seed_precision, *save_payload, l)
                )
            logging.info("Waiting for EndLoss quantization IO to finish...")
            for future_save in pending_saves:
                future_save.result()
    else:
        for l in layers_to_process:
            load_wait_start = time.perf_counter()
            loaded = layer_loader(l)
            logging.info(f"[Layer {l}] Loaded weights/init/Hessian cache in {time.perf_counter() - load_wait_start:.2f}s")
            save_payload = run_layer(l, loaded)
            _save_results(output_folder, seed_precision, seed_precision, *save_payload, l)
