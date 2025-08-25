import paddle
import numpy as np

def npu_quant_weight(weight_np):
    if isinstance(weight_np, paddle.Tensor):
        if weight_np.dtype == paddle.bfloat16:
            weight_np = paddle.cast(weight_np, paddle.float16)
        weight_np = weight_np.numpy()
    weight = weight_np
    max_value = np.max(np.abs(weight), axis=0).reshape(1, -1)
    quanted_weight = clip_and_round(weight / max_value * 127.0)
    quanted_weight = paddle.to_tensor(quanted_weight)
    weight_scales = (max_value / 127.0).astype(weight_np.dtype).reshape(-1)
    weight_scales = paddle.to_tensor(weight_scales)
    weight_scales = paddle.cast(weight_scales, paddle.get_default_dtype())
    return quanted_weight, weight_scales

def clip_and_round(x):
    return np.clip(np.around(x), -127, 127).astype("int8")