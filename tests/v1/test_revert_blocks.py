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

from fastdeploy.cache_manager.cache_data import BlockNode
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
    model_cfg = SimpleNamespace(enable_mm=enable_mm, max_model_len=8192)
    speculative_cfg = SimpleNamespace(method=None)
    model_cfg.print = print
    cache_cfg.bytes_per_layer_per_block = 1
    parallel_cfg = ParallelConfig(args)
    scheduler_cfg = SchedulerConfig()
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


def test_revert_match_blocks():
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
    request_1 = Request.from_dict(req1_dict)
    matched_token_num = 20 * 64
    match_node_ids = []
    matche_nodes = []
    match_gpu_block_ids = []
    match_cpu_block_ids = []
    for idx in range(20):
        node_id = idx + 10
        block = BlockNode(node_id, [], 0, 0, idx, 0, None, None, None)
        match_node_ids.append(node_id)
        matche_nodes.append(block)
        match_gpu_block_ids.append(idx)
    match_cpu_block_ids.append(match_gpu_block_ids.pop())
    match_cpu_block_ids.append(match_gpu_block_ids.pop())
    gpu_match_token_num = len(match_gpu_block_ids) * block_size
    cpu_match_token_num = len(match_cpu_block_ids) * block_size

    (
        gpu_match_token_num,
        cpu_match_token_num,
        current_match_node,
    ) = cache_manager._revert_match_blocks(
        request=request_1,
        matched_token_num=matched_token_num,
        block_size=block_size,
        chunk_idx=0,
        match_node_ids=match_node_ids,
        matche_nodes=matche_nodes,
        match_gpu_block_ids=match_gpu_block_ids,
        match_cpu_block_ids=match_cpu_block_ids,
        gpu_match_token_num=gpu_match_token_num,
        cpu_match_token_num=cpu_match_token_num,
        swap_node_ids=[],
    )

    assert match_gpu_block_ids == [0, 1]
    assert match_cpu_block_ids == []
    assert gpu_match_token_num == 120
    assert cpu_match_token_num == 0
    assert match_node_ids == [10, 11]
