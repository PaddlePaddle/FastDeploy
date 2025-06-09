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

from abc import ABC, abstractmethod
import paddle.distributed as dist
import paddle.distributed.fleet as fleet
import paddle
from fastdeploy.utils import get_logger
from fastdeploy.model_executor.layers.attention import get_attention_backend
from fastdeploy.worker.model_runner.forward_meta import ForwardMeta

logger = get_logger("worker", "worker.log")


class ModelRunnerBase(ABC):

    """
        Initializes the model and sets up necessary parameters.

    Args:
        config (Config): The configuration object for the model.
        args (Namespace): The arguments passed to the script.

    Returns:
        None.

    Raises:
        None.
    """

    def __init__(self, config, args):
        self.share_inputs = {}
        self.model_cfg = config
        self.args = args

        self.init_dist_env()

        self._init_share_inputs(args.max_num_seqs)
        self.init_rotary_position_embedding(args.max_model_len)
        self.num_gpu_blocks = args.total_block_num


        self._load_model(config.model_name_or_path, args.dynamic_load_weight)
        self._init_attn_backend()
        self._init_kvcache()
        self.forward_meta = None
        self.attn_backend = None

    def _init_attn_backend(self):
        self.attn_backend_cls = get_attention_backend(
            self.args.attention_backend)

    def _init_forward_meta(self):
        self.forward_meta = ForwardMeta.init_forward_mata(self)
        self.share_inputs["forward_meta"] = self.forward_meta

    def _log_memory_usage(self, context: str = "") -> None:
        """Log current GPU memory usage."""
        max_alloc = paddle.device.cuda.max_memory_allocated() / (1024 ** 3)
        max_reserved = paddle.device.cuda.max_memory_reserved() / (1024 ** 3)
        curr_alloc = paddle.device.cuda.memory_allocated() / (1024 ** 3)
        curr_reserved = paddle.device.cuda.memory_reserved() / (1024 ** 3)

        logger.info(f"GPU memory usage {context}:")
        logger.warning(
            f"max_allocated: {max_alloc:.2f}GB\n"
            f"max_reserved: {max_reserved:.2f}GB\n"
            f"current_allocated: {curr_alloc:.2f}GB\n"
            f"current_reserved: {curr_reserved:.2f}GB"
        )

    def init_dist_env(self, seed=20):
        """
        init distributed env
        """
        self.nranks = dist.get_world_size()
        strategy = fleet.DistributedStrategy()

        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": self.nranks,
            "pp_degree": 1,
            "sharding_degree": 1,
        }

        # Set control in tensor parallel
        strategy.tensor_parallel_configs = {"tensor_init_seed": seed}
        fleet.init(is_collective=True, strategy=strategy)
        self.rank = fleet.worker_index()

    def _load_model_init_val(self):
        """
        initialize model config from config file
        """
        def _get_attr(key, default=None):
            if hasattr(self.model_cfg, key):
                return getattr(self.model_cfg, key)
            return default
        self.top_p = _get_attr("top_p", 0.0)
        self.temperature = _get_attr("temperature", 1.0)
        self.rope_theta = _get_attr("rope_theta", 10000.0)
        self.rope_scaling = _get_attr("rope_scaling", None)
        self.penalty_score = _get_attr("penalty_score", 1.0)
        self.frequency_score = _get_attr("frequency_score", 0.0)
        self.presence_score = _get_attr("presence_score", 0.0)
        self.min_length = _get_attr("min_length", 1)
        self.max_length = self.args.max_model_len

    def _init_share_inputs(self, max_num_seqs):
        """
        初始化共享的输入，包括预测和训练。
            将所有需要的张量都初始化为零或者特定值。

            Args:
                max_num_seqs (int): 最大批次大小，用于初始化张量。

            Returns:
                None.
        """
        # 统一使用paddle.full创建张量
        self._load_model_init_val()

        int64_config = {"dtype": "int64"}
        int32_config = {"dtype": "int32"}
        float32_config = {"dtype": "float32"}
        bool_config = {"dtype": "bool"}

        # 批量初始化张量
        self.share_inputs.update({
            "pre_ids": paddle.full([max_num_seqs, self.max_length], -1, **int64_config),
            "input_ids": paddle.full([max_num_seqs, self.args.max_model_len], self.args.pad_token_id, **int64_config),
            "eos_token_id": paddle.full([self.args.eos_tokens_lens, 1], 0, **int64_config),
            "top_p": paddle.full([max_num_seqs, 1], self.top_p, **float32_config),
            "temperature": paddle.full([max_num_seqs, 1], self.temperature, **float32_config),
            "penalty_score": paddle.full([max_num_seqs, 1], self.penalty_score, **float32_config),
            "frequency_score": paddle.full([max_num_seqs, 1], self.frequency_score, **float32_config),
            "presence_score": paddle.full([max_num_seqs, 1], self.presence_score, **float32_config),
            # TODO 名称统一
            "min_dec_len": paddle.full([max_num_seqs, 1], self.min_length, **int64_config),
            "max_dec_len": paddle.full([max_num_seqs, 1], self.max_length, **int64_config),
            "min_length": paddle.full([max_num_seqs, 1], self.min_length, **int64_config),
            "max_length": paddle.full([max_num_seqs, 1], self.max_length, **int64_config),
            "seq_lens_this_time": paddle.full(max_num_seqs, 0, **int32_config),
            "seq_lens_encoder": paddle.full([max_num_seqs, 1], 0, **int32_config),
            "step_seq_lens_encoder": paddle.full([max_num_seqs, 1], 0, **int32_config),
            "seq_lens_decoder": paddle.full([max_num_seqs, 1], 0, **int32_config),
            "step_idx": paddle.full([max_num_seqs, 1], 0, **int64_config),
            "not_need_stop": paddle.full([1], False, **bool_config).cpu(),
            "stop_flags": paddle.full([max_num_seqs, 1], True, **bool_config),
            "stop_nums": paddle.full([1], max_num_seqs, **int64_config),
            "bad_tokens": paddle.full([1], -1, **int64_config),
            "next_tokens": paddle.full([max_num_seqs, 1], -1, **int64_config),
            "is_block_step": paddle.full([max_num_seqs], False, **bool_config),
            "encoder_block_lens": paddle.full([max_num_seqs], 0, **int32_config),
            "step_block_list": paddle.full([max_num_seqs], -1, **int32_config),
            "step_lens": paddle.full([1], 0, **int32_config),
            "recover_block_list": paddle.full([max_num_seqs], -1, **int32_config),
            "recover_lens": paddle.full([1], 0, **int32_config),
            "need_block_list": paddle.full([max_num_seqs], -1, **int32_config),
            "need_block_len": paddle.full([1], 0, **int32_config),
            "used_list_len": paddle.full([max_num_seqs], 0, **int32_config),
            "infer_seed": paddle.full([max_num_seqs, 1], 0, **int64_config),
            "first_token_ids": paddle.full([max_num_seqs, 1], -1, **int64_config),
            "ori_seq_lens_encoder": paddle.full([max_num_seqs, 1], 0, **int32_config),
            "system_lens": paddle.full([max_num_seqs, 1], 0, **int32_config),
            "system_ids": paddle.full([max_num_seqs, 1], -1, **int32_config),
        })

        # 计算block tables相关参数
        pre_max_block_num = (
            self.args.max_model_len + self.args.block_size - 1
        ) // self.args.block_size + self.args.enc_dec_block_num
        self.share_inputs["block_tables"] = paddle.full(
            [max_num_seqs, pre_max_block_num], -1, **int32_config
        )

        # 初始化free list
        free_list = list(
            range(self.args.total_block_num - 1, int(self.args.total_block_num * self.args.kv_cache_ratio) - 1, -1)
        )
        self.free_list_len = len(free_list)
        self.share_inputs.update({
            "free_list": paddle.to_tensor(free_list, dtype="int32"),
            "free_list_len": paddle.full([1], self.free_list_len, **int32_config),
        })

        # 初始化stop seqs
        self.share_inputs.update({
            "stop_seqs_len": paddle.full([self.model_cfg.max_stop_seqs_num], 0, **int32_config),
            "stop_seqs": paddle.full(
                [self.model_cfg.max_stop_seqs_num, self.model_cfg.stop_seqs_max_len], -1, **int64_config
            ),
        })
    
    def update_chunked_prefill(self, token_chunk_size=384):
        """
        更新chunked prefill相关参数
        """
        if not self.args.enable_chunked_prefill:
            return
        
        raise NotImplementedError("currently chunked_prefill is not supported.")
    
    def prefill_finished(self):
        """
        判断是否已经完成了prefill操作
        """
        return True

    @abstractmethod
    def init_rotary_position_embedding(self, max_model_len):
        """
            初始化旋转位置编码，需要重写该方法。
            参数max_model_len（int）：序列的最大长度。
            返回值（None）：无返回值，需要在方法内完成初始化操作。
        """
        raise NotImplementedError

    @abstractmethod
    def _load_model(self, model_dir, dynamic_load_weight):
        """
            加载模型，包括模型参数和优化器等。
        需要子类实现该方法。

        Args:
            model_dir (str): 模型保存的目录路径。

        Raises:
            NotImplementedError: 当前方法未被实现。

        Returns:
            None.
        """
        raise NotImplementedError

    @abstractmethod
    def _init_kvcache(self):
        """
            初始化kv缓存，用于快速查找数据块。
        该方法需要被子类实现。

        Args:
            max_block_num (int): 最大的数据块数量。

        Raises:
            NotImplementedError: 当该方法未被子类实现时会引发此异常。
        """
        raise NotImplementedError

    @abstractmethod
    def dy_input_preprocess(self):
        """
            预处理输入数据，用于计算dy。
            该函数需要在每次forward之前调用，并且只能调用一次。
            默认实现抛出NotImplementedError。子类可以根据具体的模型实现此功能。

            Raises:
                NotImplementedError: 如果没有实现该方法。
        """
        raise NotImplementedError
