import os
import time
from typing import Optional

import paddle
import paddle.distributed.fleet.base.topology as tp
from paddle.distributed import fleet
from paddleformers.utils.log import logger


class MoeBalanceAnalyser:
    def __init__(
        self,
        num_layers,
        num_moe_expert,
        save_dir="/root/paddlejob/workspace/env_run/output/gaoziyuan/moe_analyser",
        interval=120,
    ):

        self.expert_topk_select = paddle.zeros([num_layers, num_moe_expert], dtype="int64")
        self.save_dir = save_dir
        self.interval = interval
        # hcg = tp._HYBRID_PARALLEL_GROUP
        self.tensor_parallel_rank = self._get_tp_rank()
        logger.info(
            f"MoeBalanceAnalyser tensor_parallel_rank: {self.tensor_parallel_rank}. save_dir: {self.save_dir}. save_interval: {self.interval}"
        )

        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        self._last_save_time = time.time()

    def _get_tp_rank(self):
        tensor_parallel_degree = paddle.distributed.get_world_size()
        hcg = tp._HYBRID_PARALLEL_GROUP
        if hcg is None:
            strategy = fleet.DistributedStrategy()
            strategy.hybrid_configs = {
                "dp_degree": 1,
                "mp_degree": tensor_parallel_degree,
                "pp_degree": 1,
                "sharding_degree": 1,
            }
            fleet.init(is_collective=True, strategy=strategy)
            hcg = fleet.get_hybrid_communicate_group()

        tensor_parallel_rank = hcg.get_model_parallel_rank()
        return tensor_parallel_rank

    def update(self, layer_id, topk_ids):
        """
        更新某一层的专家选择频次统计。
        :param layer_id: 当前层索引
        :param topk_ids: 形状为 [batch_size, top_k] 的张量，表示每个 token 选中的 top-k 专家 ID
        """
        # 获取专家总数
        num_experts = self.expert_topk_select.shape[1]
        # 将 topk_ids 展平成一维
        flat_topk_ids = topk_ids.reshape([-1])
        # 使用 paddle.bincount 统计每个专家 ID 出现的次数
        counts = paddle.bincount(flat_topk_ids, minlength=num_experts)
        print("counts", counts)
        # 累加到对应层的统计中
        self.expert_topk_select[layer_id, :] += counts.cast(self.expert_topk_select.dtype)
        print("expert_topk_select")
        print(self.expert_topk_select)
        # 检查是否需要保存
        self._check_and_save()

    def _check_and_save(self):
        """检查时间间隔是否超过阈值"""
        now = time.time()
        if now - self._last_save_time >= self.interval:
            self.save()
            self._last_save_time = now

    def save(self, force=True):
        """保存一次结果
        :param force: 是否无视间隔强制保存
        """
        now = time.time()
        if force or (now - self._last_save_time >= self.interval):
            # --- 1. 保存本 rank 的局部数据 ---
            local_name = f"moe_expert_rank{self.tensor_parallel_rank}.pdparams"
            local_path = os.path.join(self.save_dir, local_name)
            paddle.save(self.expert_topk_select, local_path)
            logger.info(f"save moe_info to {local_path}")
            # --- 2. 做 all_reduce(sum)，得到 global_sum ---
            # TODO: 目前这里由于 moe 空转会冲突，会 hang 住
            # global_sum = self.expert_topk_select.clone()
            # dist.all_reduce(global_sum, op=dist.ReduceOp.SUM)
            # if dist.get_rank() == 0:
            #     global_path = os.path.join(self.save_dir, "moe_expert_global_sum.pdparams")
            #     paddle.save(global_sum, global_path)
            #     logger.info(f"[INFO] rank0 保存 global_sum: {global_path}")

            self._last_save_time = now

    def __del__(self):
        """对象销毁前强制保存一次"""
        try:
            self.save(force=True)
        except Exception:
            pass


# 模块级全局变量，每个 rank 一个实例
_GLOBAL_MOE_ANALYSER: Optional[MoeBalanceAnalyser] = None


def get_moe_analyser(
    num_layers: int = 37,
    num_moe_expert: int = 128,
    save_dir: str = "/root/paddlejob/workspace/env_run/output/gaoziyuan/moe_analyser",
    interval: int = 120,
) -> MoeBalanceAnalyser:
    """
    获取全局唯一的 MoE 分析器实例（每个 rank 独立）
    第一次调用时初始化，后续调用返回同一实例
    """
    global _GLOBAL_MOE_ANALYSER
    if _GLOBAL_MOE_ANALYSER is None:
        _GLOBAL_MOE_ANALYSER = MoeBalanceAnalyser(
            num_layers=num_layers, num_moe_expert=num_moe_expert, save_dir=save_dir, interval=interval
        )
    return _GLOBAL_MOE_ANALYSER
