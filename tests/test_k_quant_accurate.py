#!/usr/bin/env python3
"""
Accurate test for indexer_k_quant_and_cache operator.
This test uses proper understanding of operator behavior from kernel code.
"""

import os
import sys

import numpy as np
import paddle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastdeploy.model_executor.ops.gpu import indexer_k_quant_and_cache


def float_to_fp8_e4m3(x):
    """Convert float to FP8 E4M3 format (numpy implementation)."""
    # FP8 E4M3 format: 1 sign bit, 4 exponent bits, 3 mantissa bits
    # Range: [-448, 448]

    x = np.clip(x, -448.0, 448.0)

    # Handle zero case
    if x == 0:
        return 0

    # Extract sign
    sign = 1 if x >= 0 else -1
    abs_x = abs(x)

    # For E4M3 FP8 format:
    # Exponent bias = 7, exponent range = [0, 15] (4 bits)
    # Mantissa has 3 bits: m0 m1 m2 (no implied 1 like in IEEE754)

    # Find exponent
    exponent = np.floor(np.log2(abs_x))
    exponent = np.clip(exponent, -6, 8)  # E4M3 range: exponent in [-6, 8]
    if exponent < -6:
        # Subnormal would be handled here, but FP8 E4M3 has limited subnormal range
        exponent = -6

    # Normalize mantissa
    mantissa = abs_x / (2.0**exponent)
    # For E4M3, mantissa in [1.0, 1 + 7/8) = [1.0, 1.875)
    mantissa = mantissa - 1.0  # Remove leading 1
    mantissa_int = int(round(mantissa * 8))  # 3 bits = 8 values

    # Combine bits
    sign_bit = 0 if sign > 0 else 1
    exp_bias = exponent + 7  # Add bias 7
    exp_bits = int(exp_bias) & 0xF  # 4 bits

    return (sign_bit << 7) | (exp_bits << 3) | (mantissa_int & 0x7)


def fp8_e4m3_to_float(fp8_val):
    """Convert FP8 E4M3 value to float."""
    # Extract bits
    sign_bit = (fp8_val >> 7) & 0x1
    exp_bits = (fp8_val >> 3) & 0xF
    mantissa_bits = fp8_val & 0x7

    # Handle special cases
    if exp_bits == 0:
        # Subnormal number
        if mantissa_bits == 0:
            return 0.0
        significand = mantissa_bits / 8.0
        exponent = -6  # E_min - 1
    else:
        # Normal number
        significand = 1.0 + mantissa_bits / 8.0
        exponent = exp_bits - 7  # Remove bias

    value = significand * (2.0**exponent)
    if sign_bit:
        value = -value

    return value


def naive_k_quant_and_cache(
    k, slot_mapping, head_dim=128, quant_block_size=128, cache_block_size=8, scale_format="fp16"
):
    """Naive implementation of k quantization and caching."""
    num_tokens = k.shape[0]

    # Allocate cache (128 bytes per token + 4 bytes per block for scale, using 64-byte alignment)
    cache_size = cache_block_size * 136  # 136 bytes per entry = 128 + 8 (128 quant + 8 for scale align)
    cache = np.zeros(cache_size, dtype=np.uint8)

    scales = []

    for token_idx in range(num_tokens):
        slot = slot_mapping[token_idx]
        if slot < 0:
            continue

        block_idx = slot // cache_block_size
        block_offset = slot % cache_block_size

        # Calculate cache position
        cache_offset = block_idx * cache_block_size * 136 + block_offset * 136

        # Process each block of quant_block_size
        for block_start in range(0, head_dim, quant_block_size):
            block_end = min(block_start + quant_block_size, head_dim)
            block = k[token_idx, block_start:block_end]

            # Find max absolute value
            max_abs = np.max(np.abs(block))

            # Calculate scale (from kernel: scale = fmaxf(amax, 1e-4f) / kFp8ScaleDivisorDS)
            if max_abs < 1e-4:
                max_abs = 1e-4
            scale = max_abs / 224.0  # Using 224 as kFp8ScaleDivisorDS

            if scale_format == "ue8m0":
                # Power-of-2 scaling
                scale = 2 ** np.ceil(np.log2(scale))

            # Scale and quantize
            scaled = block / scale
            scaled = np.clip(scaled, -224.0, 224.0)  # Using half range as in kernel

            # Quantize to FP8 E4M3
            quantized = np.array([float_to_fp8_e4m3(float(x)) for x in scaled], dtype=np.uint8)

            # Write to cache
            cache_pos = cache_offset + block_start
            cache[cache_pos : cache_pos + len(quantized)] = quantized

            # Store scale (as float32, 4 bytes)
            scale_bytes = np.frombuffer(np.float32(scale).tobytes(), dtype=np.uint8)
            scale_pos = cache_offset + 128  # After 128 bytes of quantized data
            block_idx_in_token = block_start // quant_block_size
            cache[scale_pos + block_idx_in_token * 4 : scale_pos + (block_idx_in_token + 1) * 4] = scale_bytes

            scales.append(scale)

    return cache, scales


def decode_cache(cache, slot_mapping, head_dim=128, quant_block_size=128, cache_block_size=8):
    """Decode cache to get quantized values."""
    num_tokens = len(slot_mapping)
    decoded = []
    scale_list = []

    for token_idx in range(num_tokens):
        slot = slot_mapping[token_idx]
        if slot < 0:
            decoded.append(np.zeros(head_dim, dtype=np.float32))
            scale_list.append(None)
            continue

        block_idx = slot // cache_block_size
        block_offset = slot % cache_block_size
        cache_offset = block_idx * cache_block_size * 136 + block_offset * 136

        # Get quantized data
        quantized_data = cache[cache_offset : cache_offset + head_dim]

        # Get scales (each 4 bytes float32)
        scale_offset = cache_offset + 128
        num_blocks = head_dim // quant_block_size
        scales = []

        for block_idx_in_token in range(num_blocks):
            scale_bytes = cache[scale_offset + block_idx_in_token * 4 : scale_offset + (block_idx_in_token + 1) * 4]
            if len(scale_bytes) == 4:
                scale_value = np.frombuffer(scale_bytes.tobytes(), dtype=np.float32)[0]
                scales.append(scale_value)

        # Dequantize
        dequantized = np.zeros(head_dim, dtype=np.float32)
        for block_idx_in_token in range(num_blocks):
            block_start = block_idx_in_token * quant_block_size
            block_end = min(block_start + quant_block_size, head_dim)

            if block_idx_in_token < len(scales) and scales[block_idx_in_token] > 0:
                block_data = quantized_data[block_start:block_end]
                # Dequantize each value
                for i in range(len(block_data)):
                    dequantized_val = fp8_e4m3_to_float(block_data[i]) * scales[block_idx_in_token]
                    dequantized[block_start + i] = dequantized_val

        decoded.append(dequantized)
        scale_list.append(scales)

    return decoded, scale_list


def test_accuracy():
    """Test with controlled data to understand operator behavior."""
    print("=" * 80)
    print("Accurate Test for indexer_k_quant_and_cache")
    print("=" * 80)

    # Use simple test data
    num_tokens = 2
    head_dim = 128
    quant_block_size = 128  # Should match kernel default
    cache_block_size = 8

    # Create test data with controlled values
    np.random.seed(42)
    k = np.random.randn(num_tokens, head_dim).astype(np.float16)
    # Normalize to reasonable range for FP8
    k = k * 0.5

    # Simple slot mapping
    slot_mapping = np.array([0, 1], dtype=np.int64)

    print("Test Configuration:")
    print(f"  num_tokens: {num_tokens}")
    print(f"  head_dim: {head_dim}")
    print(f"  quant_block_size: {quant_block_size}")
    print(f"  cache_block_size: {cache_block_size}")
    print(f"  k data range: [{k.min():.6f}, {k.max():.6f}]")
    print()

    # Convert to Paddle tensors
    k_tensor = paddle.to_tensor(k)
    slot_mapping_tensor = paddle.to_tensor(slot_mapping)

    # Expected cache size: 136 bytes per entry (128 quant + 8 for scale alignment)
    expected_cache_size = cache_block_size * 136
    cache_tensor = paddle.zeros(expected_cache_size, dtype=paddle.uint8)

    print("Running GPU operator...")

    # Run GPU operator
    indexer_k_quant_and_cache(
        k_tensor,
        cache_tensor,
        slot_mapping_tensor,
        head_dim,
        quant_block_size,
        cache_block_size,
        cache_stride=head_dim + 8,  # 128 + 8 = 136
        use_ue8m0=False,
    )

    # Get results
    cache_np = cache_tensor.numpy()

    print("\nCache analysis:")

    # Analyze cache content
    for token_idx in range(num_tokens):
        slot = slot_mapping[token_idx]
        if slot < 0:
            print(f"Token {token_idx}: slot {slot} (skipped)")
            continue

        block_idx = slot // cache_block_size
        block_offset = slot % cache_block_size
        cache_offset = block_idx * cache_block_size * 136 + block_offset * 136

        print(f"\nToken {token_idx} (slot {slot}):")

        # Check quantized data
        quantized_data = cache_np[cache_offset : cache_offset + head_dim]
        non_zero = np.count_nonzero(quantized_data)
        print(f"  Quantized data non-zero bytes: {non_zero}/{head_dim}")

        # Check scales
        scale_offset = cache_offset + 128
        # For head_dim=128, quant_block_size=128, we have 1 block
        if head_dim // quant_block_size == 1:
            scale_bytes = cache_np[scale_offset : scale_offset + 4]
            if len(scale_bytes) == 4:
                scale = np.frombuffer(scale_bytes.tobytes(), dtype=np.float32)[0]
                print(f"  Scale: {scale:.6e}")

                # Analyze max value in original data
                max_abs = np.max(np.abs(k[token_idx]))
                expected_scale = max(max_abs, 1e-4) / 224.0
                print(f"  Expected scale (max_abs={max_abs:.6f}/224): {expected_scale:.6e}")
                print(f"  Scale ratio (actual/expected): {scale/expected_scale:.6f}")

        # Check dequantized values
        if head_dim // quant_block_size == 1 and "scale" in locals():
            dequantized = np.zeros(head_dim, dtype=np.float32)
            for i in range(head_dim):
                dequantized[i] = fp8_e4m3_to_float(quantized_data[i]) * scale

            # Compare with original (approximate due to FP8 quantization)
            mse = np.mean((dequantized - k[token_idx]) ** 2)
            max_err = np.max(np.abs(dequantized - k[token_idx]))
            print(f"  Dequantization MSE vs original: {mse:.6e}")
            print(f"  Max absolute error: {max_err:.6f}")

            # Show some sample comparisons
            print("  Sample values (first 5):")
            for i in range(min(5, head_dim)):
                orig = k[token_idx, i]
                deq = dequantized[i]
                qval = quantized_data[i]
                print(f"    [{i}] orig={orig:.4f}, quant=0x{qval:02x}, deq={deq:.4f}, err={(deq-orig):.4f}")

    # Also run naive implementation for comparison
    print("\n" + "-" * 80)
    print("Naive implementation comparison:")

    naive_cache, naive_scales = naive_k_quant_and_cache(
        k, slot_mapping, head_dim, quant_block_size, cache_block_size, scale_format="fp16"
    )

    # Compare with GPU cache
    if len(cache_np) == len(naive_cache):
        # Find differences
        diff_indices = np.where(cache_np != naive_cache)[0]
        print(f"  Cache differences at {len(diff_indices)} positions")

        if len(diff_indices) > 0:
            print("  First 5 differences:")
            for i in diff_indices[:5]:
                gpu_val = cache_np[i]
                naive_val = naive_cache[i]
                print(f"    [{i}] GPU={gpu_val:02x}, Naive={naive_val:02x}, diff={gpu_val-naive_val}")

        # Check scale values
        for token_idx in range(num_tokens):
            slot = slot_mapping[token_idx]
            if slot < 0:
                continue

            block_idx = slot // cache_block_size
            block_offset = slot % cache_block_size
            cache_offset = block_idx * cache_block_size * 136 + block_offset * 136
            scale_offset = cache_offset + 128

            # Get GPU scale
            gpu_scale_bytes = cache_np[scale_offset : scale_offset + 4]
            gpu_scale = np.frombuffer(gpu_scale_bytes.tobytes(), dtype=np.float32)[0]

            # Get naive scale
            if naive_scales and len(naive_scales) > token_idx:
                naive_scale = naive_scales[token_idx] if token_idx < len(naive_scales) else 0
                print(f"  Token {token_idx}: GPU scale={gpu_scale:.6e}, Naive scale={naive_scale:.6e}")

    print("\n" + "=" * 80)
    print("Test completed.")


if __name__ == "__main__":
    test_accuracy()
