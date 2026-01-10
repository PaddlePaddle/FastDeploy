# import sys
# sys.path.append("/data/dyf/code/Songguo2025/SQAttn_paddle")
import paddle
from loguru import logger
# from paddle_utils import *
# from qtorch.quant import float_quantize


class BaseQuantizer(object):
    def __init__(self, bit, symmetric, granularity, **kwargs):
        self.bit = bit
        self.sym = symmetric
        self.granularity = granularity
        self.kwargs = kwargs
        if self.granularity == "per_group":
            self.group_size = self.kwargs["group_size"]
        self.calib_algo = self.kwargs.get("calib_algo", "minmax")

    def get_tensor_range(self, tensor):
        if self.calib_algo == "minmax":
            return self.get_minmax_range(tensor)
        elif self.calib_algo == "mse":
            return self.get_mse_range(tensor)
        else:
            raise ValueError(f"Unsupported calibration algorithm: {self.calib_algo}")

    def get_minmax_range(self, tensor):
        if tensor.dtype == paddle.bfloat16 or tensor.dtype == paddle.float16:
            tensor = tensor.astype(paddle.float32)
            
        if self.granularity == "per_tensor":
            max_val = paddle.compat.max(tensor)
            min_val = paddle.compat.min(tensor)
        elif self.granularity == "per_channel":
            max_val = tensor.amax(dim=(0, 1), keepdim=True)
            min_val = tensor.amin(dim=(0, 1), keepdim=True)
        else:
            max_val = tensor.amax(dim=-1, keepdim=True)
            min_val = tensor.amin(dim=-1, keepdim=True)
        return min_val, max_val

    def get_mse_range(self, tensor):
        raise NotImplementedError

    def get_qparams(self, tensor_range, device):
        min_val, max_val = tensor_range[0], tensor_range[1]
        qmin = self.qmin.to(device)
        qmax = self.qmax.to(device)
        if self.sym:
            abs_max = paddle.compat.max(max_val.abs(), min_val.abs())
            # abs_max = abs_max.clamp(min=1e-05)
            abs_max = abs_max.clip(min=1e-05)
            scales = abs_max / qmax
            zeros = paddle.tensor(0.0)
        else:
            # scales = (max_val - min_val).clamp(min=1e-05) / (qmax - qmin)
            scales = (max_val - min_val).clip(min=1e-05) / (qmax - qmin)
            # zeros = (qmin - paddle.round(min_val / scales)).clamp(qmin, qmax)
            zeros = (qmin - paddle.round(min_val / scales)).clip(qmin, qmax)
        return scales, zeros, qmax, qmin

    def reshape_tensor(self, tensor, allow_padding=False):
        if self.granularity == "per_group":
            t = tensor.reshape(-1, self.group_size)
        else:
            t = tensor
        return t

    def restore_tensor(self, tensor, shape):
        if tensor.shape == shape:
            t = tensor
        else:
            t = tensor.reshape(shape)
        return t

    def get_tensor_qparams(self, tensor):
        tensor = self.reshape_tensor(tensor)
        tensor_range = self.get_tensor_range(tensor)
        scales, zeros, qmax, qmin = self.get_qparams(tensor_range, tensor.place)
        return tensor, scales, zeros, qmax, qmin

    def fake_quant_tensor(self, tensor):
        org_shape = tensor.shape
        org_dtype = tensor.dtype
        tensor, scales, zeros, qmax, qmin = self.get_tensor_qparams(tensor)
        tensor = self.quant_dequant(tensor, scales, zeros, qmax, qmin)
        tensor = self.restore_tensor(tensor, org_shape).to(org_dtype)
        return tensor

    def real_quant_tensor(self, tensor):
        org_shape = tensor.shape
        tensor, scales, zeros, qmax, qmin = self.get_tensor_qparams(tensor)
        tensor = self.quant(tensor, scales, zeros, qmax, qmin)
        tensor = self.restore_tensor(tensor, org_shape)
        if self.sym:
            zeros = None
        return tensor, scales, zeros


class IntegerQuantizer(BaseQuantizer):
    def __init__(self, bit, symmetric, granularity, **kwargs):
        super().__init__(bit, symmetric, granularity, **kwargs)
        if "int_range" in self.kwargs:
            self.qmin = self.kwargs["int_range"][0]
            self.qmax = self.kwargs["int_range"][1]
        elif self.sym:
            self.qmin = -(2 ** (self.bit - 1))
            self.qmax = 2 ** (self.bit - 1) - 1
        else:
            self.qmin = 0.0
            self.qmax = 2**self.bit - 1
        self.qmin = paddle.tensor(self.qmin)
        self.qmax = paddle.tensor(self.qmax)
        self.dst_nbins = 2**bit

    def quant(self, tensor, scales, zeros, qmax, qmin):
        tensor = paddle.clamp(paddle.round(tensor / scales) + zeros, qmin, qmax)
        return tensor

    def dequant(self, tensor, scales, zeros):
        tensor = (tensor - zeros) * scales
        return tensor

    def quant_dequant(self, tensor, scales, zeros, qmax, qmin):
        tensor = self.quant(tensor, scales, zeros, qmax, qmin)
        tensor = self.dequant(tensor, scales, zeros)
        return tensor

def safe_finfo(dtype):
    if dtype == paddle.float32:
        return paddle.finfo(paddle.float32)
    elif dtype == paddle.float16:
        return paddle.finfo(paddle.float16)
    elif dtype == paddle.bfloat16:
        return paddle.finfo(paddle.bfloat16)
    elif str(dtype) == "paddle.float8_e4m3fn":
        class _Finfo:
            min = -448.0
            max = 448.0
            eps = 0.015625  # 2^-6
        return _Finfo()
    elif str(dtype) == "paddle.float8_e5m2":
        class _Finfo:
            min = -57344.0
            max = 57344.0
            eps = 6.1035e-5  # 2^-14
        return _Finfo()
    else:
        raise TypeError(f"Unsupported dtype: {dtype}")



def float_quantize(x, e_bits=5, m_bits=10, rounding="nearest"):
    """
    模拟浮点量化 (类似 QTorch.float_quantize)
    Args:
        x: paddle.Tensor
        e_bits: 指数位数 (例如 FP8 E4M3 -> 4)
        m_bits: 尾数位数 (例如 FP8 E4M3 -> 3)
        rounding: 'nearest' 或 'stochastic'
    Returns:
        量化后的 tensor (paddle.Tensor)
    """
    assert rounding in ["nearest", "stochastic"]

    # ---- IEEE-like 参数计算 ----
    bias = 2 ** (e_bits - 1) - 1
    eps = 2.0 ** (-m_bits)              # 最小步长
    max_exp = 2 ** (e_bits - 1) - 1
    min_exp = 1 - bias

    # ---- 特殊情况 ----
    sign = paddle.sign(x)
    x = paddle.abs(x)

    # ---- 零处理 ----
    zero_mask = (x == 0)

    # ---- 获取指数与尾数 ----
    exp = paddle.floor(paddle.log2(x + 1e-30))
    frac = x / (2.0 ** exp) - 1.0

    # ---- 量化指数 ----
    exp_clamped = paddle.clip(exp, min_exp, max_exp)

    # ---- 量化尾数 ----
    if rounding == "nearest":
        frac_q = paddle.round(frac / eps) * eps
    else:  # stochastic rounding
        noise = paddle.rand(frac.shape, dtype=frac.dtype)
        frac_q = paddle.floor(frac / eps + noise) * eps

    frac_q = paddle.clip(frac_q, 0.0, 1.0 - eps)

    # ---- 重建量化值 ----
    x_q = (1.0 + frac_q) * (2.0 ** exp_clamped)
    x_q = x_q * sign

    # ---- 下溢/上溢处理 ----
    max_val = (2 - 2 ** (-m_bits)) * (2.0 ** max_exp)
    x_q = paddle.clip(x_q, -max_val, max_val)

    # ---- 零恢复 ----
    x_q = paddle.where(zero_mask, paddle.zeros_like(x_q), x_q)

    return x_q


class FloatQuantizer(BaseQuantizer):
    def __init__(self, bit, symmetric, granularity, **kwargs):
        super().__init__(bit, symmetric, granularity, **kwargs)
        assert self.bit in [
            "e4m3",
            "e5m2",
        ], f"Unsupported bit configuration: {self.bit}"
        assert self.sym
        if self.bit == "e4m3":
            self.e_bits = 4
            self.m_bits = 3
            self.fp_dtype = paddle.float8_e4m3fn
        elif self.bit == "e5m2":
            self.e_bits = 5
            self.m_bits = 2
            self.fp_dtype = paddle.float8_e5m2
        else:
            raise ValueError(f"Unsupported bit configuration: {self.bit}")
        # import pdb; pdb.set_trace()
        finfo = safe_finfo(self.fp_dtype)
        self.qmin, self.qmax = finfo.min, finfo.max
        self.qmax = paddle.tensor(self.qmax)
        self.qmin = paddle.tensor(self.qmin)

    def quant(self, tensor, scales, zeros, qmax, qmin):
        scaled_tensor = tensor / scales + zeros
        scaled_tensor = paddle.clamp(scaled_tensor, self.qmin.cuda(), self.qmax.cuda())
        org_dtype = scaled_tensor.dtype
        q_tensor = float_quantize(
            scaled_tensor.float(), self.e_bits, self.m_bits, rounding="nearest"
        )
        q_tensor.to(org_dtype)
        return q_tensor

    def dequant(self, tensor, scales, zeros):
        tensor = (tensor - zeros) * scales
        return tensor

    def quant_dequant(self, tensor, scales, zeros, qmax, qmin):
        tensor = self.quant(tensor, scales, zeros, qmax, qmin)
        tensor = self.dequant(tensor, scales, zeros)
        return tensor


if __name__ == "__main__":
    weight = paddle.randn(4096, 4096, dtype=paddle.bfloat16).cuda()
    quantizer = IntegerQuantizer(4, False, "per_channel")
    q_weight = quantizer.fake_quant_tensor(weight)
    logger.info(weight)
    logger.info(q_weight)
    logger.info(
        f"cosine = {paddle.nn.functional.cosine_similarity(x1=weight.view(1, -1).to(paddle.float64), x2=q_weight.view(1, -1).to(paddle.float64))}"
    )
