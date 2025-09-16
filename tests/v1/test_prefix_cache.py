from dataclasses import asdict
from types import SimpleNamespace

from fastdeploy.cache_manager.prefix_cache_manager import PrefixCacheManager
from fastdeploy.config import CacheConfig, FDConfig, ParallelConfig, SchedulerConfig
from fastdeploy.engine.args_utils import EngineArgs
from fastdeploy.engine.request import Request


def test_normal_case():
    max_num_seqs = 3
    block_size = 64
    engine_args = EngineArgs(max_num_seqs=max_num_seqs, num_gpu_blocks_override=100, max_num_batched_tokens=3200)
    args = asdict(engine_args)
    cache_cfg = CacheConfig(args)
    model_cfg = SimpleNamespace(enable_mm=False)
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
        scheduler_cfg=scheduler_cfg,
    )
    cache_manager = PrefixCacheManager(config=fd_config, tensor_parallel_size=8, splitwise_role="mixed")
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
    assert len(cache_manager.gpu_free_block_list) == 100
    req1.block_tables.extend(common_block_ids)
    # allocate for req1 inputs
    num_new_block = 50
    req1.block_tables.extend(cache_manager.allocate_gpu_blocks(num_new_block))
    req1.num_computed_tokens += 50 * block_size
    cache_manager.update_cache_blocks(req1, block_size, req1.num_computed_tokens)
    assert len(cache_manager.gpu_free_block_list) == 50
    # allocate for req2 inputs
    (common_block_ids, matched_token_num, hit_info) = cache_manager.request_match_blocks(req2, block_size)
    assert len(common_block_ids) == 25
    assert matched_token_num == 25 * block_size
    req2.num_cached_tokens = matched_token_num
    req2.num_computed_tokens == 25 * block_size
    num_new_block = 25
    req2.block_tables.extend(common_block_ids)
    req2.block_tables.extend(cache_manager.allocate_gpu_blocks(num_new_block))
    cache_manager.update_cache_blocks(req2, block_size, req2.num_computed_tokens)
    # allocate for req3 input
    (common_block_ids, matched_token_num, hit_info) = cache_manager.request_match_blocks(req3, block_size)
    assert len(common_block_ids) == 25
    assert matched_token_num == 25 * block_size
    req3.num_cached_tokens = matched_token_num
    req3.num_computed_tokens == 25 * block_size
    assert len(cache_manager.gpu_free_block_list) == 25
    req3.block_tables.extend(common_block_ids)
    num_new_block = 25
    assert cache_manager.can_allocate_gpu_blocks(num_new_block)
    req3.block_tables.extend(cache_manager.allocate_gpu_blocks(num_new_block))
    cache_manager.update_cache_blocks(req3, block_size, req3.num_computed_tokens)
    assert len(cache_manager.gpu_free_block_list) == 0
