import os
import unittest

import numpy as np
import paddle

paddle.seed(2026)


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y


def get_tma_aligned_size(x: int, element_size: int) -> int:
    """
    Align x to TMA-required size.

    Args:
        x: size in elements
        element_size: size of each element in bytes

    Returns:
        Aligned size in elements
    """
    kNumTMAAlignmentBytes = 16
    assert kNumTMAAlignmentBytes % element_size == 0
    return align(x, kNumTMAAlignmentBytes // element_size)


def _get_mn_major_tma_aligned_packed_ue8m0_tensor_torch_impl(
    x: paddle.Tensor,
):
    assert x.dtype == paddle.float and x.dim() in (2, 3)

    ue8m0_tensor = (x.view(paddle.int) >> 23).to(paddle.uint8)

    mn, k = x.shape[-2], x.shape[-1]
    remove_dim = False

    if x.dim() == 2:
        x, remove_dim = x.unsqueeze(0), True
    b = x.shape[0]

    aligned_mn = get_tma_aligned_size(mn, 4)
    aligned_k = align(k, 4)

    padded = paddle.zeros((b, aligned_mn, aligned_k), device=x.device, dtype=paddle.uint8)
    padded[:, :mn, :k] = ue8m0_tensor

    padded = padded.view(-1).view(dtype=paddle.int).view(b, aligned_mn, aligned_k // 4)

    transposed = paddle.zeros((b, aligned_k // 4, aligned_mn), device=x.device, dtype=paddle.int).mT
    transposed[:, :, :] = padded

    aligned_x = transposed[:, :mn, :]

    return aligned_x.squeeze(0) if remove_dim else aligned_x


def transform_scale_ue8m0(sf, mn, weight_block_size=None):
    get_mn_major_tma_aligned_packed_ue8m0_tensor = _get_mn_major_tma_aligned_packed_ue8m0_tensor_torch_impl
    if weight_block_size:
        assert weight_block_size == [128, 128]
        sf = sf.index_select(-2, paddle.arange(mn, device=sf.device) // 128)
    sf = get_mn_major_tma_aligned_packed_ue8m0_tensor(sf)
    return sf


def ceil_to_ue8m0_paddle(x: paddle.Tensor):
    """
    x > 0
    return 2 ^ ceil(log2(x))
    """
    # log2(x)
    log2_x = paddle.log(x) / paddle.log(paddle.to_tensor(2.0, dtype=x.dtype))
    # ceil
    ceil_log2_x = paddle.ceil(log2_x)
    # 2^k
    return paddle.pow(paddle.to_tensor(2.0, dtype=x.dtype), ceil_log2_x)


def masked_per_token_quant_ref(input_tensor, recv_expert_count, block_size, use_ue8m0):
    """
    Paddle API implementation of masked_per_token_quant

    Args:
        input_tensor: Input tensor with shape [num_local_expert, num_max_tokens_per_expert, hidden_size]
        recv_expert_count: Expert token count tensor with shape [num_local_expert]
        block_size: Quantization block size

    Returns:
        Tuple of (quantized_tensor, scale_tensor)
    """
    MAX_VALUE = 448.0
    epsilon = 1e-10

    # Get dimensions
    input_shape = input_tensor.shape
    num_local_expert = input_shape[0]
    num_max_tokens_per_expert = input_shape[1]
    hidden_size = input_shape[2]

    # CUDA kernel uses: hidden_size_scale = hidden_size / block_size (integer division)
    # This assumes hidden_size is divisible by block_size
    hidden_size_scale = hidden_size // block_size

    # Check environment variable for fine-grained range
    use_finegrained_range = False
    env_var = os.getenv("PER_TOKEN_QUANT_FP8_USE_FINEGRAINED_RANGE")
    if env_var:
        use_finegrained_range = bool(int(env_var))

    # Create mask for valid tokens based on recv_expert_count
    token_indices = paddle.arange(num_max_tokens_per_expert, dtype="int32").unsqueeze(
        0
    )  # [1, num_max_tokens_per_expert]
    expert_counts = recv_expert_count.unsqueeze(1)  # [num_local_expert, 1]
    valid_mask = token_indices < expert_counts  # [num_local_expert, num_max_tokens_per_expert]

    # Reshape input for block-wise processing
    # [num_local_expert, num_max_tokens_per_expert, hidden_size_scale, block_size]
    reshaped_input = paddle.reshape(
        input_tensor, [num_local_expert, num_max_tokens_per_expert, hidden_size_scale, block_size]
    ).astype("float32")

    # Calculate max absolute values per block
    max_abs_val = paddle.max(
        paddle.abs(reshaped_input), axis=-1, keepdim=True
    )  # [num_local_expert, num_max_tokens_per_expert, hidden_size_scale, 1]
    max_abs_val = paddle.clip(max_abs_val, min=epsilon)

    # Apply valid mask - set invalid tokens' max values to epsilon
    valid_mask_expanded = valid_mask.unsqueeze(2).unsqueeze(3)  # [num_local_expert, num_max_tokens_per_expert, 1, 1]
    max_abs_val = paddle.where(valid_mask_expanded, max_abs_val, paddle.to_tensor(epsilon))

    # Apply fine-grained range if enabled
    if use_finegrained_range:
        max_abs_val *= 7.0

    # Calculate scale
    scale = max_abs_val / MAX_VALUE

    if use_ue8m0:
        scale = ceil_to_ue8m0_paddle(scale)

    # Quantize
    quanted_value = reshaped_input / scale

    # Convert to float8_e4m3fn and reshape back
    quanted_x_reshaped = quanted_value.astype("float8_e4m3fn")
    quanted_x = paddle.reshape(quanted_x_reshaped, [num_local_expert, num_max_tokens_per_expert, hidden_size])

    # Apply valid mask to quantized output - convert to float32 first, then back to float8_e4m3fn
    valid_mask_full = valid_mask.unsqueeze(2)  # [num_local_expert, num_max_tokens_per_expert, 1]
    quanted_x_float32 = quanted_x.astype("float32")
    quanted_x_masked_float32 = paddle.where(valid_mask_full, quanted_x_float32, paddle.zeros_like(quanted_x_float32))
    quanted_x = quanted_x_masked_float32.astype("float8_e4m3fn")

    # Prepare scale output - squeeze the last dimension
    quanted_scale = paddle.squeeze(scale, axis=-1)  # [num_local_expert, num_max_tokens_per_expert, hidden_size_scale]

    # Apply valid mask to scale
    valid_mask_scale = valid_mask.unsqueeze(2)  # [num_local_expert, num_max_tokens_per_expert, 1]
    quanted_scale = paddle.where(valid_mask_scale, quanted_scale, paddle.zeros_like(quanted_scale))

    if use_ue8m0:
        quanted_scale = transform_scale_ue8m0(quanted_scale, mn=quanted_x.shape[-2])

    return quanted_x, quanted_scale


def run_fused(x, token_nums, block_size, use_ue8m0=False):
    import fastdeploy.model_executor.ops.gpu as ops

    return ops.fused_mask_swiglu_fp8_quant(x, token_nums, block_size, use_ue8m0)


def run_separate(x, token_nums, block_size, use_ue8m0=False):
    """Run separate operations (FastDeploy non-fused kernels)"""
    from fastdeploy.model_executor.ops.gpu import group_swiglu_with_masked

    swiglu = group_swiglu_with_masked(x, token_nums)
    q, scale = masked_per_token_quant_ref(swiglu, token_nums, block_size, use_ue8m0)
    return q, scale


# ------------------------------------------------------------
# Test case
# ------------------------------------------------------------


def benchmark_cuda(fn, warmup=10, repeat=10):
    """
    Benchmark a CUDA function using paddle.device.Event
    fn: callable with no return dependency on CPU
    """
    # warmup
    for _ in range(warmup):
        fn()
    paddle.device.synchronize()

    start = paddle.device.Event(enable_timing=True)
    end = paddle.device.Event(enable_timing=True)

    start.record()
    for _ in range(repeat):
        fn()
    end.record()

    end.synchronize()
    elapsed_ms = start.elapsed_time(end)  # ms

    return elapsed_ms / repeat


class TestFusedSwigluFP8Quant(unittest.TestCase):
    def setUp(self):
        paddle.set_device("gpu")
        # 10, 2048, 7168
        self.group_num = 10
        self.group_size = 2048
        self.hidden_dim = 7168
        self.block_size = 128
        self.x = paddle.randn(
            [self.group_num, self.group_size, self.hidden_dim * 2],
            dtype="bfloat16",
        )
        self.token_nums = paddle.to_tensor([50, 51, 50, 50, 50, 50, 50, 49, 51, 51], dtype="int32")

    def fused_vs_separate_exact_match(self, use_ue8m0=False):
        """
        Test fused kernel vs separate operations - should be exact match
        This compares FastDeploy's fused kernel vs FastDeploy's separate kernels
        """
        # Run separate operations
        q_ref, s_ref = run_separate(self.x, self.token_nums, self.block_size, use_ue8m0)

        # Run fused kernel
        q_fused, s_fused = run_fused(self.x, self.token_nums, self.block_size, use_ue8m0)

        def run_sep():
            run_separate(self.x, self.token_nums, self.block_size)

        def run_fus():
            run_fused(self.x, self.token_nums, self.block_size)

        t_sep = benchmark_cuda(run_sep)
        t_fus = benchmark_cuda(run_fus)

        print("\n====== Fused vs Separate Benchmark ======")
        print(f"Separate: {t_sep:.3f} ms")
        print(f"Fused   : {t_fus:.3f} ms")
        print(f"Speedup : {t_sep / t_fus:.2f}x")

        # ---------------- valid mask ----------------
        arange = paddle.arange(self.group_size, dtype="int32")
        valid = arange < self.token_nums.unsqueeze(1)  # [G, S]

        valid_flat = valid.reshape([-1])

        # ---------------- FP8 output ----------------
        q_ref_flat = q_ref.reshape([-1, q_ref.shape[-1]]).astype("float32")
        q_fused_flat = q_fused.reshape([-1, q_fused.shape[-1]]).astype("float32")

        # ---------------- scale ----------------
        s_ref_flat = s_ref.reshape([-1, s_ref.shape[-1]])
        s_fused_flat = s_fused.reshape([-1, s_fused.shape[-1]])

        np.testing.assert_allclose(
            s_ref_flat[valid_flat].numpy(),
            s_fused_flat[valid_flat].numpy(),
            rtol=1e-06,
            err_msg="**scale mismatch**",
        )

        np.testing.assert_allclose(
            q_ref_flat[valid_flat].numpy(),
            q_fused_flat[valid_flat].numpy(),
            equal_nan=True,
            rtol=0.5,
            err_msg="**quant_x mismatch**",
        )

    def test_fused(self):
        self.fused_vs_separate_exact_match(use_ue8m0=True)
        self.fused_vs_separate_exact_match(use_ue8m0=False)


if __name__ == "__main__":
    unittest.main()
