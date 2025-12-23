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

import unittest
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


class TestIsChunkedMMInput(unittest.TestCase):
    def setUp(self):
        self.cache_manager = make_prefix_cache_manager(max_num_seqs=3, enable_mm=True, num_gpu_blocks_override=100)

    def test_is_chunked_mm_input_none_input(self):
        result, idx = self.cache_manager.is_chunked_mm_input(None, 10)
        self.assertFalse(result)
        self.assertEqual(idx, 0)

    def test_is_chunked_mm_input_no_mm_positions(self):
        mm_inputs = {"other_field": "value"}
        result, idx = self.cache_manager.is_chunked_mm_input(mm_inputs, 10)
        self.assertFalse(result)
        self.assertEqual(idx, 0)

    def test_is_chunked_mm_input_empty_positions(self):
        mm_inputs = {"mm_positions": []}
        result, idx = self.cache_manager.is_chunked_mm_input(mm_inputs, 10)
        self.assertFalse(result)
        self.assertEqual(idx, 0)

    def test_is_chunked_mm_input_matched_in_chunk(self):
        mm_inputs = {
            "mm_positions": [
                ImagePosition(offset=5, length=10),
                ImagePosition(offset=20, length=10),
            ]
        }
        result, idx = self.cache_manager.is_chunked_mm_input(mm_inputs, 8)
        self.assertTrue(result)
        self.assertEqual(idx, 0)

    def test_is_chunked_mm_input_matched_in_second_chunk(self):
        mm_inputs = {
            "mm_positions": [
                ImagePosition(offset=5, length=10),
                ImagePosition(offset=20, length=10),
            ]
        }
        result, idx = self.cache_manager.is_chunked_mm_input(mm_inputs, 25)
        self.assertTrue(result)
        self.assertEqual(idx, 1)

    def test_is_chunked_mm_input_before_first_chunk(self):
        mm_inputs = {
            "mm_positions": [
                ImagePosition(offset=5, length=10),
                ImagePosition(offset=20, length=10),
            ]
        }
        result, idx = self.cache_manager.is_chunked_mm_input(mm_inputs, 3)
        self.assertFalse(result)
        self.assertEqual(idx, 0)

    def test_is_chunked_mm_input_after_last_chunk(self):
        mm_inputs = {
            "mm_positions": [
                ImagePosition(offset=5, length=10),
                ImagePosition(offset=20, length=10),
            ]
        }
        result, idx = self.cache_manager.is_chunked_mm_input(mm_inputs, 35)
        self.assertFalse(result)
        self.assertEqual(idx, 0)


class TestRevertMatchBlocks(unittest.TestCase):
    def setUp(self):
        self.block_size = 64
        self.cache_manager = make_prefix_cache_manager(max_num_seqs=3, enable_mm=True, num_gpu_blocks_override=100)

    def make_match_blocks(self, gpu_block_num, cpu_block_num):
        block_num = gpu_block_num + cpu_block_num
        matched_token_num = block_num * self.block_size
        match_node_ids = []
        matche_nodes = []
        match_gpu_block_ids = []
        match_cpu_block_ids = []
        for idx in range(block_num):
            node_id = idx + 10
            block = BlockNode(node_id, [], 0, 0, idx, 0, None, None, None)
            match_node_ids.append(node_id)
            matche_nodes.append(block)
            match_gpu_block_ids.append(idx)

        for _ in range(cpu_block_num):
            match_cpu_block_ids.append(match_gpu_block_ids.pop())

        gpu_match_token_num = len(match_gpu_block_ids) * self.block_size
        cpu_match_token_num = len(match_cpu_block_ids) * self.block_size
        return (
            matched_token_num,
            match_node_ids,
            matche_nodes,
            match_gpu_block_ids,
            match_cpu_block_ids,
            gpu_match_token_num,
            cpu_match_token_num,
        )

    def test_revert_full_blocks(self):
        # Setup test data
        multimodal_inputs = {
            "mm_positions": [ImagePosition(offset=0, length=1200)],
            "mm_hashes": ["image1"],
        }
        req_dict = {
            "request_id": "req1",
            "prompt_token_ids": [-1] * 1200 + [2] * 120,
            "prompt_token_ids_len": 1320,
            "multimodal_inputs": multimodal_inputs,
        }

        (
            matched_token_num,
            match_node_ids,
            matche_nodes,
            match_gpu_block_ids,
            match_cpu_block_ids,
            gpu_match_token_num,
            cpu_match_token_num,
        ) = self.make_match_blocks(gpu_block_num=2, cpu_block_num=0)

        # Call method
        (
            gpu_match_token_num,
            cpu_match_token_num,
            current_match_node,
        ) = self.cache_manager._revert_match_blocks(
            request=Request.from_dict(req_dict),
            matched_token_num=matched_token_num,
            block_size=self.block_size,
            chunk_idx=0,
            match_node_ids=match_node_ids,
            matche_nodes=matche_nodes,
            match_gpu_block_ids=match_gpu_block_ids,
            match_cpu_block_ids=match_cpu_block_ids,
            gpu_match_token_num=gpu_match_token_num,
            cpu_match_token_num=cpu_match_token_num,
            swap_node_ids=[],
        )

        # Assertions
        self.assertEqual(gpu_match_token_num, 0)
        self.assertEqual(cpu_match_token_num, 0)
        self.assertEqual(len(match_node_ids), 0)
        self.assertEqual(len(match_gpu_block_ids), 0)

    def test_revert_partial_block(self):
        # Setup test data
        multimodal_inputs = {
            "mm_positions": [ImagePosition(offset=120, length=1200)],
            "mm_hashes": ["image1"],
        }
        req_dict = {
            "request_id": "req1",
            "prompt_token_ids": [1] * 120 + [-1] * 1200 + [2] * 120,
            "prompt_token_ids_len": 1440,
            "multimodal_inputs": multimodal_inputs,
        }

        (
            matched_token_num,
            match_node_ids,
            matche_nodes,
            match_gpu_block_ids,
            match_cpu_block_ids,
            gpu_match_token_num,
            cpu_match_token_num,
        ) = self.make_match_blocks(gpu_block_num=20, cpu_block_num=0)

        # Call method
        (
            gpu_match_token_num,
            cpu_match_token_num,
            current_match_node,
        ) = self.cache_manager._revert_match_blocks(
            request=Request.from_dict(req_dict),
            matched_token_num=matched_token_num,
            block_size=self.block_size,
            chunk_idx=0,
            match_node_ids=match_node_ids,
            matche_nodes=matche_nodes,
            match_gpu_block_ids=match_gpu_block_ids,
            match_cpu_block_ids=match_cpu_block_ids,
            gpu_match_token_num=gpu_match_token_num,
            cpu_match_token_num=cpu_match_token_num,
            swap_node_ids=[],
        )

        # Assertions
        self.assertEqual(gpu_match_token_num, 120)
        self.assertEqual(cpu_match_token_num, 0)
        self.assertEqual(len(match_node_ids), 2)
        self.assertEqual(len(match_gpu_block_ids), 2)

    def test_revert_with_cpu_blocks(self):
        # Setup test data
        multimodal_inputs = {
            "mm_positions": [ImagePosition(offset=120, length=1200), ImagePosition(offset=1440, length=420)],
            "mm_hashes": ["image1", "image2"],
        }
        req_dict = {
            "request_id": "req1",
            "prompt_token_ids": [1] * 120 + [-1] * 1200 + [2] * 120 + [-1] * 420,
            "prompt_token_ids_len": 1860,
            "multimodal_inputs": multimodal_inputs,
        }

        (
            matched_token_num,
            match_node_ids,
            matche_nodes,
            match_gpu_block_ids,
            match_cpu_block_ids,
            gpu_match_token_num,
            cpu_match_token_num,
        ) = self.make_match_blocks(gpu_block_num=22, cpu_block_num=6)

        # Call method
        (
            gpu_match_token_num,
            cpu_match_token_num,
            current_match_node,
        ) = self.cache_manager._revert_match_blocks(
            request=Request.from_dict(req_dict),
            matched_token_num=matched_token_num,
            block_size=self.block_size,
            chunk_idx=1,
            match_node_ids=match_node_ids,
            matche_nodes=matche_nodes,
            match_gpu_block_ids=match_gpu_block_ids,
            match_cpu_block_ids=match_cpu_block_ids,
            gpu_match_token_num=gpu_match_token_num,
            cpu_match_token_num=cpu_match_token_num,
            swap_node_ids=[],
        )

        # Assertions
        self.assertEqual(gpu_match_token_num, 22 * self.block_size)
        self.assertEqual(cpu_match_token_num, 32)
        self.assertEqual(len(match_node_ids), 23)
        self.assertEqual(len(match_gpu_block_ids), 22)
        self.assertEqual(len(match_cpu_block_ids), 1)


if __name__ == "__main__":
    unittest.main()
