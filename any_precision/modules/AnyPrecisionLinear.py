import torch
import torch.nn as nn
import numpy as np

try:
    import ap_gemv
except:
    ap_gemv = None


def _calculate_new_indices(byte_indices, threads_per_warp, offset=0):
    bytes_per_thread = 4
    bytes_per_warp = threads_per_warp * bytes_per_thread
    warp_idx, byte_offsets_within_warp = np.divmod(byte_indices, bytes_per_warp)
    warp_offsets = warp_idx * bytes_per_warp
    thread_indices = byte_indices % threads_per_warp
    byte_offsets_within_thread = byte_offsets_within_warp // threads_per_warp
    byte_offsets_within_thread ^= 3
    return warp_offsets + thread_indices * bytes_per_thread + byte_offsets_within_thread + offset


def _permute_bitmaps(bitmaps, inverse=False):
    _, _, total_bytes = bitmaps.shape
    full_warps_bytes = (total_bytes // 128) * 128
    remaining_bytes_start_idx = full_warps_bytes

    full_warp_byte_indices = np.arange(full_warps_bytes)
    new_full_warp_byte_indices = _calculate_new_indices(full_warp_byte_indices, 32)

    remaining_bytes = total_bytes - full_warps_bytes
    if remaining_bytes:
        remaining_byte_indices = np.arange(remaining_bytes)
        adjusted_threads_per_warp = remaining_byte_indices.size // 4
        new_remaining_byte_indices = _calculate_new_indices(
            remaining_byte_indices, adjusted_threads_per_warp,
            offset=remaining_bytes_start_idx)

        new_byte_indices = np.empty(total_bytes, dtype=np.int64)
        new_byte_indices[:full_warps_bytes] = new_full_warp_byte_indices
        new_byte_indices[full_warps_bytes:] = new_remaining_byte_indices
    else:
        new_byte_indices = new_full_warp_byte_indices

    if not inverse:
        return bitmaps[:, :, np.argsort(new_byte_indices)]
    return bitmaps[:, :, np.argsort(np.argsort(new_byte_indices))]


@torch.library.custom_op("plugin::anyprec_gemv", mutates_args={"output"})
def anyprec_gemv(x: torch.Tensor, q_weight: torch.Tensor, lut: torch.Tensor, output: torch.Tensor, bitwidth: int) -> None:
    ap_gemv.anyprec_gemv(x, output, q_weight, lut, bitwidth)


@anyprec_gemv.register_fake
def _(x, q_weight, lut, output, bitwidth):
    return None


class AnyPrecisionLinear(nn.Module):
    def __init__(self, in_features, out_features, supported_bits, bias=True, precisions=None, device=None,
                 dtype=None):
        super().__init__()
        if precisions is None:
            precisions = supported_bits
        if not isinstance(precisions, list):
            raise RuntimeError('supported_bits must be a list of integers.')
        if dtype is not None and dtype != torch.float16:
            raise RuntimeError('Only float16 is supported for now.')

        self.in_features = in_features
        self.out_features = out_features
        self.precisions = precisions
        self.precision = max(self.precisions)
        self.supported_bits = supported_bits

        self.register_buffer(
            'qweight',
            torch.empty((max(supported_bits), out_features, in_features // 32), dtype=torch.int32, device=device)
        )

        for bit in supported_bits:
            self.register_buffer(
                f'lut{bit}',
                torch.empty((out_features, 2 ** bit), dtype=dtype, device=device)
            )

        if bias:
            self.register_buffer(
                "bias",
                torch.empty((out_features,), dtype=dtype, device=device)
            )
        else:
            self.bias = None

        self.output = torch.zeros((1, 1, self.out_features), dtype=torch.float16, device='cuda') \
            if ap_gemv is not None else None
        self._dense_weight_cache = {}

    def prune_precisions(self):
        self.qweight = self.qweight[:max(self.precisions)]
        for bit in self.supported_bits:
            if bit not in self.precisions:
                delattr(self, f'lut{bit}')
        self._dense_weight_cache.clear()

    def forward(self, x, **kwargs):
        if 'precision' in kwargs:
            w_bits = kwargs['precision']
        else:
            w_bits = self.precision

        if ap_gemv is None:
            weight = self._torch_dequantize_weight(w_bits, x.dtype, x.device)
            x = torch.matmul(x, weight.T)
        elif x.numel() // x.shape[-1] > 1:
            weight = ap_gemv.anyprec_dequant(self.qweight, self._buffers[f'lut{w_bits}'].to(torch.float16), w_bits).to(x.dtype)
            x = torch.matmul(x, weight.T)
        else:
            anyprec_gemv(x.to(torch.float16), self.qweight, self._buffers[f'lut{w_bits}'].to(torch.float16), self.output, w_bits)
            x = self.output.to(x.dtype)

        if self.bias is not None:
            x += self.bias

        return x.clamp_(torch.finfo(x.dtype).min * (1.0 - 5e-3), torch.finfo(x.dtype).max * (1.0 - 5e-3))
        # return x

    def _torch_dequantize_weight(self, w_bits, dtype, device):
        cache_key = (w_bits, dtype, str(device))
        if cache_key in self._dense_weight_cache:
            return self._dense_weight_cache[cache_key]

        qweight = self.qweight[:w_bits].detach().cpu().contiguous().numpy()
        packed_bytes = qweight.view(np.uint8).reshape(
            w_bits, self.out_features, self.in_features // 8)
        bitmaps = _permute_bitmaps(packed_bytes, inverse=True)

        codes = np.zeros((self.out_features, self.in_features), dtype=np.int64)
        for bit_idx in range(w_bits):
            unpacked = np.unpackbits(
                bitmaps[bit_idx], axis=-1, count=self.in_features).astype(np.int64)
            codes = (codes << 1) | unpacked

        lut = self._buffers[f'lut{w_bits}'].detach().cpu()
        weight = torch.gather(lut, 1, torch.from_numpy(codes)).to(device=device, dtype=dtype)
        self._dense_weight_cache[cache_key] = weight
        return weight

    def set_precision(self, precision):
        if precision not in self.precisions:
            raise RuntimeError(f"{self.precisions}-bit precisions are supported but {precision}-bit was specified.")

        self.precision = precision

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'