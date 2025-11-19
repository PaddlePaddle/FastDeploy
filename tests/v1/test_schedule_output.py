"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
"""

from dataclasses import asdict
from types import SimpleNamespace

from fastdeploy.config import CacheConfig, FDConfig, ParallelConfig, SchedulerConfig
from fastdeploy.engine.args_utils import EngineArgs
from fastdeploy.engine.request import Request
from fastdeploy.engine.sched.resource_manager_v1 import ResourceManagerV1


def test_normal_schedule():
    max_num_seqs = 3
    engine_args = EngineArgs(max_num_seqs=max_num_seqs, num_gpu_blocks_override=160, max_num_batched_tokens=3200)
    args = asdict(engine_args)
    cache_cfg = CacheConfig(args)
    model_cfg = SimpleNamespace(enable_mm=False)
    speculative_cfg = SimpleNamespace(method=None)
    model_cfg.print = print
    model_cfg.max_model_len = 5120
    cache_cfg.bytes_per_layer_per_block = 1
    parallel_cfg = ParallelConfig(args)
    scheduler_cfg = SchedulerConfig(args)
    graph_opt_cfg = engine_args.create_graph_optimization_config()
    fd_config = FDConfig(
        model_config=model_cfg,
        cache_config=cache_cfg,
        parallel_config=parallel_cfg,
        speculative_config=speculative_cfg,
        graph_opt_config=graph_opt_cfg,
        scheduler_config=scheduler_cfg,
    )
    resource_manager_v1 = ResourceManagerV1(
        max_num_seqs=max_num_seqs, config=fd_config, tensor_parallel_size=8, splitwise_role="mixed"
    )
    req1 = Request.from_dict({"request_id": "req1", "prompt_token_ids": [1] * 3199, "prompt_token_ids_len": 3199})
    req2 = Request.from_dict({"request_id": "req2", "prompt_token_ids": [2] * 3201, "prompt_token_ids_len": 3201})
    req3 = Request.from_dict({"request_id": "req3", "prompt_token_ids": [3] * 3200, "prompt_token_ids_len": 3200})
    resource_manager_v1.add_request(req1)
    resource_manager_v1.add_request(req2)
    resource_manager_v1.add_request(req3)
    # step 1
    assert len(resource_manager_v1.waiting) == 3
    scheduler_reqs, _ = resource_manager_v1.schedule()
    assert len(scheduler_reqs) == 2
    assert scheduler_reqs[0].request_id == "req1"
    assert scheduler_reqs[1].request_id == "req2"
    assert scheduler_reqs[0].prefill_start_index == 0
    assert scheduler_reqs[1].prefill_start_index == 0
    assert scheduler_reqs[0].prefill_end_index == 3199
    assert scheduler_reqs[1].prefill_end_index == 1
    assert len(resource_manager_v1.running) == 2
    assert len(resource_manager_v1.waiting) == 1
    # step 2
    scheduler_reqs, _ = resource_manager_v1.schedule()
    assert len(scheduler_reqs) == 2
    assert scheduler_reqs[0].request_id == "req1"
    assert len(scheduler_reqs[0].block_tables) == 52
    assert scheduler_reqs[1].request_id == "req2"
    assert scheduler_reqs[1].prefill_start_index == 1
    assert scheduler_reqs[1].prefill_end_index == 3200
    assert len(resource_manager_v1.running) == 2
    assert len(resource_manager_v1.waiting) == 1
    # step 3
    scheduler_reqs, _ = resource_manager_v1.schedule()
    assert len(scheduler_reqs) == 2
    assert scheduler_reqs[0].request_id == "req2"
    assert scheduler_reqs[0].prefill_start_index == 3200
    assert scheduler_reqs[0].prefill_end_index == 3201
    assert scheduler_reqs[1].request_id == "req3"
    assert scheduler_reqs[1].prefill_start_index == 0
    assert scheduler_reqs[1].prefill_end_index == 3198
    assert len(resource_manager_v1.running) == 3
    assert len(resource_manager_v1.waiting) == 0


def test_preempted_request():
    max_num_seqs = 2
    engine_args = EngineArgs(max_num_seqs=max_num_seqs, num_gpu_blocks_override=102, max_num_batched_tokens=3200)
    args = asdict(engine_args)
    cache_cfg = CacheConfig(args)
    model_cfg = SimpleNamespace(enable_mm=False)
    speculative_cfg = SimpleNamespace(method=None)
    model_cfg.print = print
    model_cfg.max_model_len = 5120
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
    resource_manager_v1 = ResourceManagerV1(
        max_num_seqs=max_num_seqs, config=fd_config, tensor_parallel_size=8, splitwise_role="mixed"
    )
    req1 = Request.from_dict({"request_id": "req1", "prompt_token_ids": [1] * 3200, "prompt_token_ids_len": 3200})
    req2 = Request.from_dict({"request_id": "req2", "prompt_token_ids": [2] * 3200, "prompt_token_ids_len": 3200})
    resource_manager_v1.add_request(req1)
    resource_manager_v1.add_request(req2)
    # step 1
    assert len(resource_manager_v1.waiting) == 2
    scheduler_reqs, _ = resource_manager_v1.schedule()
    assert len(scheduler_reqs) == 1
    assert scheduler_reqs[0].request_id == "req1"
    assert scheduler_reqs[0].prefill_start_index == 0
    assert scheduler_reqs[0].prefill_end_index == 3200
    assert len(resource_manager_v1.running) == 1
    assert len(resource_manager_v1.waiting) == 1
    # step 2
    scheduler_reqs, _ = resource_manager_v1.schedule()
    assert len(scheduler_reqs) == 2
    assert scheduler_reqs[0].request_id == "req1"
    assert len(scheduler_reqs[0].block_tables) == 52
    # step 3
    req1.output_token_ids.extend([1] * 128)
    scheduler_reqs, _ = resource_manager_v1.schedule()
    assert len(scheduler_reqs) == 2
    assert scheduler_reqs[0].request_id == "req2"
    assert len(resource_manager_v1.running) == 1
    # to be added into waiting queue
    assert len(resource_manager_v1.waiting) == 0
    assert "req2" in resource_manager_v1.to_be_rescheduled_request_id_set
    # mock token_processor to add into waiting
    resource_manager_v1.waiting.appendleft(req2)
    # step 4
    scheduler_reqs, _ = resource_manager_v1.schedule()
    assert len(scheduler_reqs) == 0
    assert len(resource_manager_v1.running) == 1
    assert len(resource_manager_v1.waiting) == 1
