# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import paddle

from fastdeploy.model_executor.ops.xpu import speculate_get_padding_offset

test_failed = False


def ref_speculate_get_padding_offset(cum_offsets, seq_lens, max_seq_len, token_num_data):
    bsz = seq_lens.shape[0]

    padding_offset = np.zeros([token_num_data], dtype=np.int32)
    cum_offsets_out = np.zeros([bsz], dtype=np.int32)
    cu_seqlens_q = np.zeros([bsz + 1], dtype=np.int32)
    cu_seqlens_k = np.zeros([bsz + 1], dtype=np.int32)

    modified_indices = {
        "padding_offset": [],
        "cum_offsets_out": [],
        "cu_seqlens_q": [],
        "cu_seqlens_k": [],
    }

    cu_seqlens_q[0] = 0
    cu_seqlens_k[0] = 0
    modified_indices["cu_seqlens_q"].append(0)
    modified_indices["cu_seqlens_k"].append(0)

    for bi in range(bsz):
        cum_offset = 0 if bi == 0 else cum_offsets[bi - 1]
        cum_offsets_out[bi] = cum_offset
        modified_indices["cum_offsets_out"].append(bi)

        for i in range(seq_lens[bi]):
            idx = bi * max_seq_len - cum_offset + i
            if idx >= 0 and idx < token_num_data:
                padding_offset[idx] = cum_offset
                modified_indices["padding_offset"].append(idx)

        cum_seq_len = (bi + 1) * max_seq_len - cum_offsets[bi]
        cu_seqlens_q[bi + 1] = cum_seq_len
        cu_seqlens_k[bi + 1] = cum_seq_len
        modified_indices["cu_seqlens_q"].append(bi + 1)
        modified_indices["cu_seqlens_k"].append(bi + 1)

    return (
        padding_offset,
        cum_offsets_out,
        cu_seqlens_q,
        cu_seqlens_k,
        modified_indices,
    )


def test_speculate_get_padding_offset():
    global test_failed
    print("Testing speculate_get_padding_offset...")

    test_cases = [
        {
            "name": "Basic test case",
            "bsz": 4,
            "max_seq_len": 10,
            "token_num_data": 32,
            "cum_offsets": np.array([2, 5, 8, 12], dtype=np.int32),
            "seq_lens": np.array([8, 5, 7, 6], dtype=np.int32),
            "seq_lens_encoder": np.array([1, 0, 1, 0], dtype=np.int32),
        },
        {
            "name": "Batch copy optimization",
            "bsz": 5,
            "max_seq_len": 12,
            "token_num_data": 50,
            "cum_offsets": np.array([1, 4, 8, 13, 19], dtype=np.int32),
            "seq_lens": np.array([10, 6, 8, 5, 7], dtype=np.int32),
            "seq_lens_encoder": np.array([1, 0, 1, 0, 1], dtype=np.int32),
        },
        {
            "name": "Boundary conditions",
            "bsz": 3,
            "max_seq_len": 8,
            "token_num_data": 20,
            "cum_offsets": np.array([3, 8, 14], dtype=np.int32),
            "seq_lens": np.array([4, 3, 2], dtype=np.int32),
            "seq_lens_encoder": np.array([1, 0, 1], dtype=np.int32),
        },
        {
            "name": "Large sequence length",
            "bsz": 2,
            "max_seq_len": 2000,
            "token_num_data": 3000,
            "cum_offsets": np.array([100, 500], dtype=np.int32),
            "seq_lens": np.array([1800, 1500], dtype=np.int32),
            "seq_lens_encoder": np.array([1, 0], dtype=np.int32),
        },
    ]

    max_draft_tokens = 4
    all_passed = True

    for i, case in enumerate(test_cases):
        print(f"  Test case {i+1}: {case['name']}")

        input_ids = np.random.randint(0, 1000, (case["bsz"], case["max_seq_len"]), dtype=np.int64)
        draft_tokens = np.random.randint(0, 1000, (case["bsz"], max_draft_tokens), dtype=np.int64)
        token_num = np.array([case["token_num_data"]], dtype=np.int64)

        input_ids_tensor = paddle.to_tensor(input_ids)
        draft_tokens_tensor = paddle.to_tensor(draft_tokens)
        cum_offsets_tensor = paddle.to_tensor(case["cum_offsets"])
        seq_lens_tensor = paddle.to_tensor(case["seq_lens"])
        seq_lens_encoder_tensor = paddle.to_tensor(case["seq_lens_encoder"])
        token_num_tensor = paddle.to_tensor(token_num)

        (
            x_remove_padding,
            cum_offsets_out,
            padding_offset,
            cu_seqlens_q,
            cu_seqlens_k,
        ) = speculate_get_padding_offset(
            input_ids_tensor,
            draft_tokens_tensor,
            cum_offsets_tensor,
            token_num_tensor,
            seq_lens_tensor,
            seq_lens_encoder_tensor,
        )

        (
            ref_padding_offset,
            ref_cum_offsets_out,
            ref_cu_seqlens_q,
            ref_cu_seqlens_k,
            modified_indices,
        ) = ref_speculate_get_padding_offset(
            case["cum_offsets"],
            case["seq_lens"],
            case["max_seq_len"],
            case["token_num_data"],
        )

        output_arrays = {
            "padding_offset": padding_offset.numpy(),
            "cum_offsets_out": cum_offsets_out.numpy(),
            "cu_seqlens_q": cu_seqlens_q.numpy(),
            "cu_seqlens_k": cu_seqlens_k.numpy(),
        }

        ref_arrays = {
            "padding_offset": ref_padding_offset,
            "cum_offsets_out": ref_cum_offsets_out,
            "cu_seqlens_q": ref_cu_seqlens_q,
            "cu_seqlens_k": ref_cu_seqlens_k,
        }

        case_passed = True
        for key in output_arrays:
            modified_pos = modified_indices[key]
            if case["name"] == "Large sequence length" and key == "padding_offset":
                match_count = sum(1 for pos in modified_pos if output_arrays[key][pos] == ref_arrays[key][pos])
                total_positions = len(modified_pos)
                if match_count != total_positions:
                    case_passed = False
                    print(f"    \033[91m✗ {key}: {match_count}/{total_positions} positions match\033[0m")
                else:
                    print(f"    \033[92m✓ {key}: All {total_positions} positions match\033[0m")
            else:
                match_count = sum(1 for pos in modified_pos if output_arrays[key][pos] == ref_arrays[key][pos])
                if match_count != len(modified_pos):
                    case_passed = False
                    print(f"    \033[91m✗ {key}: {match_count}/{len(modified_pos)} positions match\033[0m")
                else:
                    print(f"    \033[92m✓ {key}: {match_count}/{len(modified_pos)} positions match\033[0m")

        if not case_passed:
            all_passed = False
            test_failed = True

    if all_passed:
        print("\033[92m✓ All speculate_get_padding_offset tests passed\033[0m\n")
    else:
        print("\033[91m✗ Some speculate_get_padding_offset tests failed\033[0m\n")


def test_speculate_get_padding_offset_edge_cases():
    global test_failed
    print("Testing speculate_get_padding_offset edge cases...")

    print("Test case 1: Single batch")
    bsz = 1
    max_seq_len = 10
    token_num_data = 10
    max_draft_tokens = 3

    input_ids = np.random.randint(0, 1000, (bsz, max_seq_len), dtype=np.int64)
    draft_tokens = np.random.randint(0, 1000, (bsz, max_draft_tokens), dtype=np.int64)
    cum_offsets = np.array([3], dtype=np.int32)
    seq_lens = np.array([7], dtype=np.int32)
    seq_lens_encoder = np.array([1], dtype=np.int32)
    token_num = np.array([token_num_data], dtype=np.int64)

    input_ids_tensor = paddle.to_tensor(input_ids)
    draft_tokens_tensor = paddle.to_tensor(draft_tokens)
    cum_offsets_tensor = paddle.to_tensor(cum_offsets)
    seq_lens_tensor = paddle.to_tensor(seq_lens)
    seq_lens_encoder_tensor = paddle.to_tensor(seq_lens_encoder)
    token_num_tensor = paddle.to_tensor(token_num)

    try:
        (
            x_remove_padding,
            cum_offsets_out,
            padding_offset,
            cu_seqlens_q,
            cu_seqlens_k,
        ) = speculate_get_padding_offset(
            input_ids_tensor,
            draft_tokens_tensor,
            cum_offsets_tensor,
            token_num_tensor,
            seq_lens_tensor,
            seq_lens_encoder_tensor,
        )
        print(
            f"\033[92m✓ Test case 1 passed, shapes: {[x.shape for x in [x_remove_padding, padding_offset, cum_offsets_out, cu_seqlens_q, cu_seqlens_k]]}\033[0m"
        )
    except Exception as e:
        print(f"\033[91m✗ Test case 1 failed: {e}\033[0m")
        test_failed = True

    print("Test case 2: Large batch")
    bsz = 8
    max_seq_len = 16
    token_num_data = 100

    input_ids = np.random.randint(0, 1000, (bsz, max_seq_len), dtype=np.int64)
    draft_tokens = np.random.randint(0, 1000, (bsz, max_draft_tokens), dtype=np.int64)
    cum_offsets = np.array([1, 3, 6, 10, 15, 21, 28, 36], dtype=np.int32)
    seq_lens = np.random.randint(1, max_seq_len, bsz).astype(np.int32)
    seq_lens_encoder = np.random.randint(0, 2, bsz).astype(np.int32)
    token_num = np.array([token_num_data], dtype=np.int64)

    input_ids_tensor = paddle.to_tensor(input_ids)
    draft_tokens_tensor = paddle.to_tensor(draft_tokens)
    cum_offsets_tensor = paddle.to_tensor(cum_offsets)
    seq_lens_tensor = paddle.to_tensor(seq_lens)
    seq_lens_encoder_tensor = paddle.to_tensor(seq_lens_encoder)
    token_num_tensor = paddle.to_tensor(token_num)

    try:
        (
            x_remove_padding,
            cum_offsets_out,
            padding_offset,
            cu_seqlens_q,
            cu_seqlens_k,
        ) = speculate_get_padding_offset(
            input_ids_tensor,
            draft_tokens_tensor,
            cum_offsets_tensor,
            token_num_tensor,
            seq_lens_tensor,
            seq_lens_encoder_tensor,
        )
        print(
            f"\033[92m✓ Test case 2 passed, shapes: {[x.shape for x in [x_remove_padding, padding_offset, cum_offsets_out, cu_seqlens_q, cu_seqlens_k]]}\033[0m"
        )
    except Exception as e:
        print(f"\033[91m✗ Test case 2 failed: {e}\033[0m")
        test_failed = True

    print("Test case 3: Small sequences")
    bsz = 3
    max_seq_len = 5
    token_num_data = 12

    input_ids = np.random.randint(0, 1000, (bsz, max_seq_len), dtype=np.int64)
    draft_tokens = np.random.randint(0, 1000, (bsz, max_draft_tokens), dtype=np.int64)
    cum_offsets = np.array([1, 2, 4], dtype=np.int32)
    seq_lens = np.array([2, 3, 1], dtype=np.int32)
    seq_lens_encoder = np.array([1, 0, 1], dtype=np.int32)
    token_num = np.array([token_num_data], dtype=np.int64)

    input_ids_tensor = paddle.to_tensor(input_ids)
    draft_tokens_tensor = paddle.to_tensor(draft_tokens)
    cum_offsets_tensor = paddle.to_tensor(cum_offsets)
    seq_lens_tensor = paddle.to_tensor(seq_lens)
    seq_lens_encoder_tensor = paddle.to_tensor(seq_lens_encoder)
    token_num_tensor = paddle.to_tensor(token_num)

    try:
        (
            x_remove_padding,
            cum_offsets_out,
            padding_offset,
            cu_seqlens_q,
            cu_seqlens_k,
        ) = speculate_get_padding_offset(
            input_ids_tensor,
            draft_tokens_tensor,
            cum_offsets_tensor,
            token_num_tensor,
            seq_lens_tensor,
            seq_lens_encoder_tensor,
        )
        print(
            f"\033[92m✓ Test case 3 passed, shapes: {[x.shape for x in [x_remove_padding, padding_offset, cum_offsets_out, cu_seqlens_q, cu_seqlens_k]]}\033[0m\n"
        )
    except Exception as e:
        print(f"\033[91m✗ Test case 3 failed: {e}\033[0m\n")
        test_failed = True


def test_large_scale():
    global test_failed
    print("Testing large scale data...")

    bsz = 32
    max_seq_len = 128
    token_num_data = 2048
    max_draft_tokens = 16

    input_ids = np.random.randint(0, 1000, (bsz, max_seq_len), dtype=np.int64)
    draft_tokens = np.random.randint(0, 1000, (bsz, max_draft_tokens), dtype=np.int64)
    cum_offsets = np.cumsum(np.random.randint(1, 20, bsz)).astype(np.int32)
    seq_lens = np.random.randint(1, max_seq_len, bsz).astype(np.int32)
    seq_lens_encoder = np.random.randint(0, 2, bsz).astype(np.int32)
    token_num = np.array([token_num_data], dtype=np.int64)

    input_ids_tensor = paddle.to_tensor(input_ids)
    draft_tokens_tensor = paddle.to_tensor(draft_tokens)
    cum_offsets_tensor = paddle.to_tensor(cum_offsets)
    seq_lens_tensor = paddle.to_tensor(seq_lens)
    seq_lens_encoder_tensor = paddle.to_tensor(seq_lens_encoder)
    token_num_tensor = paddle.to_tensor(token_num)

    try:
        (
            x_remove_padding,
            cum_offsets_out,
            padding_offset,
            cu_seqlens_q,
            cu_seqlens_k,
        ) = speculate_get_padding_offset(
            input_ids_tensor,
            draft_tokens_tensor,
            cum_offsets_tensor,
            token_num_tensor,
            seq_lens_tensor,
            seq_lens_encoder_tensor,
        )
        print("\033[92m✓ Large scale speculate_get_padding_offset test passed\033[0m")
        print(
            f"\033[92m  Shapes: {[x.shape for x in [x_remove_padding, padding_offset, cum_offsets_out, cu_seqlens_q, cu_seqlens_k]]}\033[0m\n"
        )
    except Exception as e:
        print(f"\033[91m✗ Large scale speculate_get_padding_offset test failed: {e}\033[0m\n")
        test_failed = True


def get_modified_indices_for_consistency_test(cum_offsets, seq_lens, max_seq_len, token_num_data):
    bsz = seq_lens.shape[0]

    modified_indices = {
        "x_remove_padding": [],
        "padding_offset": [],
        "cum_offsets_out": [],
        "cu_seqlens_q": [],
        "cu_seqlens_k": [],
    }

    for bi in range(bsz):
        modified_indices["cum_offsets_out"].append(bi)

    for i in range(bsz + 1):
        modified_indices["cu_seqlens_q"].append(i)
        modified_indices["cu_seqlens_k"].append(i)

    for bi in range(bsz):
        cum_offset = 0 if bi == 0 else cum_offsets[bi - 1]
        for i in range(seq_lens[bi]):
            padding_idx = bi * max_seq_len - cum_offset + i
            if padding_idx >= 0 and padding_idx < token_num_data:
                modified_indices["padding_offset"].append(padding_idx)

            remove_padding_idx = bi * max_seq_len - cum_offsets[bi] + i
            if remove_padding_idx >= 0 and remove_padding_idx < token_num_data:
                modified_indices["x_remove_padding"].append(remove_padding_idx)

    return modified_indices


def test_consistency():
    global test_failed
    print("Testing consistency...")

    np.random.seed(42)

    bsz = 4
    max_seq_len = 8
    token_num_data = 24
    max_draft_tokens = 3

    input_ids = np.random.randint(0, 1000, (bsz, max_seq_len), dtype=np.int64)
    draft_tokens = np.random.randint(0, 1000, (bsz, max_draft_tokens), dtype=np.int64)
    cum_offsets = np.array([1, 3, 6, 10], dtype=np.int32)
    seq_lens = np.array([6, 4, 5, 3], dtype=np.int32)
    seq_lens_encoder = np.array([1, 0, 1, 0], dtype=np.int32)
    token_num = np.array([token_num_data], dtype=np.int64)

    input_ids_tensor = paddle.to_tensor(input_ids)
    draft_tokens_tensor = paddle.to_tensor(draft_tokens)
    cum_offsets_tensor = paddle.to_tensor(cum_offsets)
    seq_lens_tensor = paddle.to_tensor(seq_lens)
    seq_lens_encoder_tensor = paddle.to_tensor(seq_lens_encoder)
    token_num_tensor = paddle.to_tensor(token_num)

    modified_indices = get_modified_indices_for_consistency_test(cum_offsets, seq_lens, max_seq_len, token_num_data)

    print("Checking consistency for modified positions only:")
    for key, indices in modified_indices.items():
        print(f"  {key}: {len(indices)} positions")

    results = []
    for run in range(3):
        (
            x_remove_padding,
            cum_offsets_out,
            padding_offset,
            cu_seqlens_q,
            cu_seqlens_k,
        ) = speculate_get_padding_offset(
            input_ids_tensor,
            draft_tokens_tensor,
            cum_offsets_tensor,
            token_num_tensor,
            seq_lens_tensor,
            seq_lens_encoder_tensor,
        )
        results.append(
            [
                x_remove_padding.numpy(),
                cum_offsets_out.numpy(),
                padding_offset.numpy(),
                cu_seqlens_q.numpy(),
                cu_seqlens_k.numpy(),
            ]
        )

    output_names = [
        "x_remove_padding",
        "cum_offsets_out",
        "padding_offset",
        "cu_seqlens_q",
        "cu_seqlens_k",
    ]
    consistent = True
    for j, name in enumerate(output_names):
        indices = modified_indices[name] if name in modified_indices else []

        if not indices:
            print(f"\033[93m    ~ {name}: No modified indices to check\033[0m")
            continue

        positions_consistent = True

        for i in range(1, len(results)):
            for idx in indices:
                if results[0][j][idx] != results[i][j][idx]:
                    consistent = False
                    positions_consistent = False
                    print(
                        f"\033[91m    ✗ {name}[{idx}]: Run 1 = {results[0][j][idx]}, Run {i+1} = {results[i][j][idx]}\033[0m"
                    )
                    break
            if not positions_consistent:
                break

        if positions_consistent:
            print(f"\033[92m    ✓ {name}: All {len(indices)} modified positions are consistent\033[0m")

    if consistent:
        print(
            "\033[92m✓ Consistency test passed - results are identical across runs (modified positions only)\033[0m\n"
        )
    else:
        print("\033[91m✗ Consistency test failed - some modified positions are inconsistent\033[0m\n")
        print("Note: This test now only compares positions that the kernel actually modifies,")
        print("      ignoring uninitialized values in other positions.\n")
        test_failed = True


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Speculate Get Padding Offset Kernels")
    print("=" * 60)

    test_speculate_get_padding_offset()
    test_speculate_get_padding_offset_edge_cases()
    test_large_scale()
    test_consistency()

    print("=" * 60)
    if test_failed:
        print("\033[91mSOME TESTS FAILED! \033[0m")
        print("\033[91mPlease check the output above for failed test details.\033[0m")
    else:
        print("\033[92mALL TESTS PASSED! \033[0m")
        print("\033[92mAll speculate_get_padding_offset kernels are working correctly.\033[0m")
    print("=" * 60)
