# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from dataclasses import asdict
from types import SimpleNamespace

from fastdeploy.cache_manager.prefix_cache_manager import PrefixCacheManager
from fastdeploy.config import CacheConfig, FDConfig, ParallelConfig
from fastdeploy.engine.args_utils import EngineArgs
from fastdeploy.engine.request import ImagePosition, Request
from fastdeploy.scheduler import SchedulerConfig


def make_prefix_cache_manager(max_num_seqs, enable_mm=False, num_gpu_blocks_override=100, max_num_batched_tokens=3200):
    engine_args = EngineArgs(
        max_num_seqs=max_num_seqs,
        num_gpu_blocks_override=num_gpu_blocks_override,
        max_num_batched_tokens=max_num_batched_tokens,
    )
    args = asdict(engine_args)
    cache_cfg = CacheConfig(args)
    model_cfg = SimpleNamespace(enable_mm=enable_mm, max_model_len=4196)
    speculative_cfg = SimpleNamespace(method=None)
    model_cfg.print = print
    cache_cfg.bytes_per_layer_per_block = 1
    parallel_cfg = ParallelConfig(args)
    scheduler_cfg = SchedulerConfig(args)
    graph_opt_cfg = engine_args.create_graph_optimization_config()
    fd_config = FDConfig(
        model_config=model_cfg,
        cache_config=cache_cfg,
        parallel_config=parallel_cfg,
        graph_opt_config=graph_opt_cfg,
        speculative_config=speculative_cfg,
        scheduler_config=scheduler_cfg,
    )
    return PrefixCacheManager(config=fd_config, tensor_parallel_size=8, splitwise_role="mixed")


def test_block_num_limit():
    import pytest

    with pytest.raises(AssertionError):
        make_prefix_cache_manager(max_num_seqs=3, enable_mm=False, num_gpu_blocks_override=20)


def test_normal_case():
    block_size = 64
    cache_manager = make_prefix_cache_manager(max_num_seqs=3, enable_mm=False, num_gpu_blocks_override=128)
    req1 = Request.from_dict({"request_id": "req1", "prompt_token_ids": [1] * 3200, "prompt_token_ids_len": 3200})
    req2 = Request.from_dict(
        {"request_id": "req2", "prompt_token_ids": [1] * 1600 + [2] * 1600, "prompt_token_ids_len": 3200}
    )
    req3 = Request.from_dict(
        {"request_id": "req3", "prompt_token_ids": [1] * 1600 + [3] * 1600, "prompt_token_ids_len": 3200}
    )
    (common_block_ids, matched_token_num, hit_info) = cache_manager.request_match_blocks(req1, block_size)
    assert len(common_block_ids) == 0
    assert matched_token_num == 0
    assert len(cache_manager.gpu_free_block_list) == 128
    req1.block_tables.extend(common_block_ids)
    # allocate for req1 inputs
    num_new_block = 50
    req1.block_tables.extend(cache_manager.allocate_gpu_blocks(num_new_block))
    req1.num_computed_tokens += 50 * block_size
    cache_manager.update_cache_blocks(req1, block_size, req1.num_computed_tokens)
    assert len(cache_manager.gpu_free_block_list) == 78
    # allocate for req2 inputs
    (common_block_ids, matched_token_num, hit_info) = cache_manager.request_match_blocks(req2, block_size)
    assert len(common_block_ids) == 25
    assert matched_token_num == 25 * block_size
    req2.num_cached_tokens = matched_token_num
    req2.num_computed_tokens = 25 * block_size
    num_new_block = 25
    req2.block_tables.extend(common_block_ids)
    req2.block_tables.extend(cache_manager.allocate_gpu_blocks(num_new_block))
    cache_manager.update_cache_blocks(req2, block_size, req2.num_computed_tokens)
    # allocate for req3 input
    (common_block_ids, matched_token_num, hit_info) = cache_manager.request_match_blocks(req3, block_size)
    assert len(common_block_ids) == 25
    assert matched_token_num == 25 * block_size
    req3.num_cached_tokens = matched_token_num
    req3.num_computed_tokens = 25 * block_size
    assert len(cache_manager.gpu_free_block_list) == 53
    req3.block_tables.extend(common_block_ids)
    num_new_block = 25
    assert cache_manager.can_allocate_gpu_blocks(num_new_block)
    req3.block_tables.extend(cache_manager.allocate_gpu_blocks(num_new_block))
    cache_manager.update_cache_blocks(req3, block_size, req3.num_computed_tokens)
    assert len(cache_manager.gpu_free_block_list) == 28


def test_mm_extra_keys():
    block_size = 64
    cache_manager = make_prefix_cache_manager(max_num_seqs=3, enable_mm=True)

    prompt_token_ids = [1] * 100 + [2] * 100
    req1 = {
        "request_id": "req1",
        "prompt_token_ids": prompt_token_ids,
        "prompt_token_ids_len": len(prompt_token_ids),
    }
    for idx in range(0, len(prompt_token_ids), block_size):
        token_ids_lens = min(block_size, len(prompt_token_ids[idx:]))
        mm_idx, extra_keys = cache_manager.get_block_hash_extra_keys(
            request=Request.from_dict(req1),
            start_idx=idx,
            end_idx=idx + token_ids_lens,
            mm_idx=0,
        )
        assert extra_keys == [], f"extra_keys {extra_keys} != [], start_idx {idx}, end_idx {idx + token_ids_lens}"
        assert mm_idx == 0, f"mm_idx {mm_idx} != 0, start_idx {idx}, end_idx {idx + token_ids_lens}"

    # block 1
    prompt_token_ids = [1] * 30 + [-1] * 34
    mm_positions = [ImagePosition(offset=30, length=80)]
    mm_hashes = ["image1"]
    extra_keys_list = [(0, ["image1"])]

    # block 2
    prompt_token_ids += [-1] * 46 + [2] * 18
    extra_keys_list.append((1, ["image1"]))

    # block 3
    prompt_token_ids += [-1] * 100
    mm_positions.append(ImagePosition(offset=128, length=100))
    mm_hashes.append("image2")
    extra_keys_list.append((1, ["image2"]))

    # block 4、5
    prompt_token_ids += [3] * 40
    extra_keys_list.append((1, ["image2"]))
    extra_keys_list.append((1, []))

    req2 = {
        "request_id": "req2",
        "prompt_token_ids": prompt_token_ids,
        "prompt_token_ids_len": len(prompt_token_ids),
        "multimodal_inputs": {
            "mm_positions": mm_positions,
            "mm_hashes": mm_hashes,
        },
    }

    mm_idx, key_idx = 0, 0
    for idx in range(0, len(prompt_token_ids), block_size):
        token_ids_lens = min(block_size, len(prompt_token_ids[idx:]))
        mm_idx, extra_keys = cache_manager.get_block_hash_extra_keys(
            request=Request.from_dict(req2),
            start_idx=idx,
            end_idx=idx + token_ids_lens,
            mm_idx=mm_idx,
        )

        target_idx, target_keys = extra_keys_list[key_idx]
        assert (
            mm_idx == target_idx
        ), f"mm_idx {mm_idx} != target_idx {target_idx}, start_idx {idx}, end_idx {idx + token_ids_lens}"
        assert (
            extra_keys == target_keys
        ), f"extra_keys {extra_keys} != target_keys {target_keys}, start_idx {idx}, end_idx {idx + token_ids_lens}"
        key_idx += 1


def test_mm_prefix_cache():
    block_size = 64
    cache_manager = make_prefix_cache_manager(max_num_seqs=3, enable_mm=True, num_gpu_blocks_override=100)
    multimodal_inputs = {
        "mm_positions": [ImagePosition(offset=120, length=1200)],
        "mm_hashes": ["image1"],
    }
    req1_dict = {
        "request_id": "req1",
        "prompt_token_ids": [1] * 120 + [-1] * 1200 + [2] * 120,
        "prompt_token_ids_len": 1440,
        "multimodal_inputs": multimodal_inputs,
    }
    req1 = Request.from_dict(req1_dict)

    multimodal_inputs = dict(multimodal_inputs)
    multimodal_inputs["mm_positions"].append(ImagePosition(offset=1836, length=587))
    multimodal_inputs["mm_hashes"].append("image2")
    req2_dict = {
        "request_id": "req2",
        "prompt_token_ids": [1] * 120 + [-1] * 1200 + [2] * 120 + [3] * 396 + [-1] * 587,
        "prompt_token_ids_len": 2423,
        "multimodal_inputs": multimodal_inputs,
    }
    req2 = Request.from_dict(req2_dict)

    multimodal_inputs = dict(multimodal_inputs)
    multimodal_inputs["mm_hashes"] = ["image3", "image4"]
    req3_dict = {
        "request_id": "req3",
        "prompt_token_ids": [1] * 120 + [-1] * 1200 + [2] * 120 + [3] * 396 + [-1] * 587,
        "prompt_token_ids_len": 2423,
        "multimodal_inputs": multimodal_inputs,
    }
    req3 = Request.from_dict(req3_dict)

    multimodal_inputs = dict(multimodal_inputs)
    multimodal_inputs["mm_positions"] = [ImagePosition(offset=120, length=1200)]
    multimodal_inputs["mm_hashes"] = ["image3"]
    req4_dict = {
        "request_id": "req4",
        "prompt_token_ids": [1] * 120 + [-1] * 1200 + [2] * 120 + [3] * 352,
        "prompt_token_ids_len": 1792,
        "multimodal_inputs": multimodal_inputs,
    }
    req4 = Request.from_dict(req4_dict)

    (common_block_ids, matched_token_num, hit_info) = cache_manager.request_match_blocks(req1, block_size)
    assert len(common_block_ids) == 0
    assert matched_token_num == 0
    assert len(cache_manager.gpu_free_block_list) == 100
    req1.block_tables.extend(common_block_ids)

    # allocate for req1 inputs
    num_new_block = 22
    req1.block_tables.extend(cache_manager.allocate_gpu_blocks(num_new_block))
    req1.num_computed_tokens += 22 * block_size
    cache_manager.update_cache_blocks(req1, block_size, req1.num_computed_tokens)
    assert len(cache_manager.gpu_free_block_list) == 78

    # allocate for req2 inputs
    (common_block_ids, matched_token_num, hit_info) = cache_manager.request_match_blocks(req2, block_size)
    assert len(common_block_ids) == 22
    assert matched_token_num == 22 * block_size
    req2.num_cached_tokens = matched_token_num
    req2.num_computed_tokens = matched_token_num
    num_new_block = 15
    req2.block_tables.extend(common_block_ids)
    req2.block_tables.extend(cache_manager.allocate_gpu_blocks(num_new_block))
    req2.num_computed_tokens += 15 * block_size
    cache_manager.update_cache_blocks(req2, block_size, req2.num_computed_tokens)

    # allocate for req3 input
    (common_block_ids, matched_token_num, hit_info) = cache_manager.request_match_blocks(req3, block_size)
    assert len(common_block_ids) == 1
    assert matched_token_num == 1 * block_size
    req3.num_cached_tokens = matched_token_num
    req3.num_computed_tokens = matched_token_num
    assert len(cache_manager.gpu_free_block_list) == 63
    req3.block_tables.extend(common_block_ids)
    num_new_block = 36
    assert cache_manager.can_allocate_gpu_blocks(num_new_block)
    req3.block_tables.extend(cache_manager.allocate_gpu_blocks(num_new_block))
    req3.num_computed_tokens += 36 * block_size
    cache_manager.update_cache_blocks(req3, block_size, req3.num_computed_tokens)
    assert len(cache_manager.gpu_free_block_list) == 27

    # allocate for req4 input
    (common_block_ids, matched_token_num, hit_info) = cache_manager.request_match_blocks(req4, block_size)
    assert len(common_block_ids) == 28
    assert matched_token_num == 28 * block_size
