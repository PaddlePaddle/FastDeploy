import numpy as np
import paddle

from fastdeploy.model_executor.ops.gpu import apply_rotary_pos_emb_vision


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.concat([-x2, x1], axis=-1)


# def apply_rotary_pos_emb_vision_py_v2(tensor: paddle.Tensor, freqs: paddle.Tensor, cos: paddle.Tensor, sin: paddle.Tensor) -> paddle.Tensor:
#     # 1.tensor shape = [seq, num_head, 80]
#     # 2.freqs shape = [seq, 1, 80]
#     tensor = tensor.astype(dtype="float32")
#     freqs = freqs.astype(dtype="float32")
#     # 3.sin_shape [seq, 1, 80]
#     # 4.cos_shape [seq, 1, 80]
#     output = tensor * cos + rotate_half(tensor) * sin
#     # 5.output_shape = [seq, num_head, 80]
#     return output


def apply_rotary_pos_emb_vision_v1(tensor: paddle.Tensor, freqs: paddle.Tensor) -> paddle.Tensor:
    orig_dtype = tensor.dtype

    with paddle.amp.auto_cast(False):
        tensor = tensor.astype(dtype="float32")
        cos = freqs.cos()
        sin = freqs.sin()
        cos = cos.unsqueeze(1).tile(repeat_times=[1, 1, 2]).astype(dtype="float32")
        sin = sin.unsqueeze(1).tile(repeat_times=[1, 1, 2]).astype(dtype="float32")
        output = tensor * cos + rotate_half(tensor) * sin
    output = paddle.cast(output, orig_dtype)
    return output


paddle.seed(111)
tensor = paddle.rand([100, 16, 80])
freqs = paddle.rand([100, 40])

out_py = apply_rotary_pos_emb_vision_v1(tensor.unsqueeze(axis=0), freqs).squeeze(axis=0).numpy()

# 先tile 再cos
# freqs = freqs.unsqueeze(axis=1)
freqs = freqs.tile(repeat_times=[1, 1, 2]).astype(dtype="float32")
cos = freqs.cos()
sin = freqs.sin()

print(cos)
print(sin)
# exit()

out2 = apply_rotary_pos_emb_vision(tensor, freqs, cos, sin).numpy()
print(out_py.shape)
print(out2.shape)
print(np.allclose(out_py, out2, 1e-3))
# print("tensor", tensor)
# print("freqs", freqs)
# print("out:", paddle.to_tensor(out))
# print("out_py:", paddle.to_tensor(out2))
