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

import os
import threading
import uuid
from typing import Dict, List, Optional, Tuple

import paddle

from fastdeploy import envs
from fastdeploy.engine.request import Request, RequestStatus
from fastdeploy.utils import offload_logger

# 导入 share_external_data 用于从共享内存获取 KV cache
try:
    from fastdeploy.cache_manager.ops import share_external_data_
except ImportError:
    share_external_data_ = None


class OffloadManager:
    """
    KV Cache Offload管理器

    职责:
    1. 管理被offload请求的KV Cache
    2. 提供Decode阶段批量offload和resume接口
    3. 维护offloaded请求队列
    4. 支持多级卸载策略 (L1 GPU -> L2 CPU -> L3 SSD)
    """

    # 存储层级常量
    STORAGE_LEVEL_CPU = "L2"
    STORAGE_LEVEL_SSD = "L3"

    def __init__(self, config=None, cache_manager=None, model_runner=None):
        """
        初始化OffloadManager

        Args:
            config: FastDeploy配置对象
            cache_manager: PrefixCacheManager实例
            model_runner: ModelRunner实例 (用于访问 KV cache tensors)
        """
        self.config = config
        self.cache_manager = cache_manager
        self.model_runner = model_runner  # 用于访问 KV cache

        # offload开关
        self.enable_offload = getattr(config, "enable_decode_offload", False) if config else False

        # TODO：offload策略参数，需要兼容性处理
        self.min_steps = 20
        # cpu block大小为8KB
        self.cpu_offloading_chunk_size = getattr(envs, "FD_CPU_OFFLOAD_CHUNK_SIZE", 8192)
        # 默认cpu memory限制为50GB
        self.cpu_memory_limit = getattr(envs, "FD_CPU_MEMORY_LIMIT", 50 * 1024 * 1024 * 1024)
        # ssd存储路径
        self.storage_path = getattr(envs, "FD_OFFLOAD_STORAGE_PATH", "/tmp/fd_offload")

        # 保存offloaded请求的相关cache信息
        self._offloaded_requests: Dict[str, dict] = {}
        self._lock = threading.Lock()

        # 缓存配置信息（延迟初始化）
        self._cache_config = None
        self._key_cache_shape = None
        self._value_cache_shape = None
        self._num_layers = None
        self._tensor_parallel_size = None
        self._local_rank = 0
        self._device_id = 0
        self._cache_dtype = None

        # 确保存储目录存在
        if self.enable_offload and not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path, exist_ok=True)

        offload_logger.info(
            f"[DEBUG: offload] OffloadManager initialized: enable_offload={self.enable_offload}, "
            f"min_steps={self.min_steps}, storage_path={self.storage_path}"
        )

    def _init_cache_info(self):
        """初始化cache配置信息（延迟初始化）"""
        if self._cache_config is not None:
            return

        if self.cache_manager is None:
            return

        self._cache_config = self.cache_manager.cache_config
        self._num_layers = self.config.model_config.num_hidden_layers
        self._tensor_parallel_size = getattr(self.config.parallel_config, "tensor_parallel_size", 1)
        self._local_rank = getattr(self.config.parallel_config, "tensor_parallel_rank", 0)

        # 从 device_ids 获取实际的设备号（考虑 CUDA_VISIBLE_DEVICES）
        device_ids_str = getattr(self.config.parallel_config, "device_ids", "0")
        device_ids = device_ids_str.split(",")
        local_dp_id = getattr(self.config.parallel_config, "local_data_parallel_id", 0)
        # device_id 是当前 rank 对应的实际 GPU 设备号
        rank_in_node = self._local_rank % len(device_ids)
        self._device_id = int(device_ids[rank_in_node]) if rank_in_node < len(device_ids) else 0

        self._cache_dtype = self._cache_config.cache_dtype

        offload_logger.info(
            f"[DEBUG: offload] _init_cache_info: local_rank={self._local_rank}, "
            f"device_ids={device_ids}, rank_in_node={rank_in_node}, device_id={self._device_id}"
        )

        # 计算KV cache shape - 使用实际的GPU block数量
        # 从cache_config获取总的GPU block数
        total_gpu_blocks = getattr(self._cache_config, "total_block_num", None)
        if total_gpu_blocks is None:
            # 尝试从cache_manager获取
            total_gpu_blocks = getattr(self.cache_manager, "num_gpu_blocks", None)
        if total_gpu_blocks is None:
            # 最后尝试从gpu_free_block_list推断
            total_gpu_blocks = len(getattr(self.cache_manager, "gpu_free_block_list", []))

        if not total_gpu_blocks:
            offload_logger.error("[DEBUG: offload] Failed to get total_gpu_blocks, using default 100")
            total_gpu_blocks = 100  # 默认值，防止错误

        key_shape, val_shape = self._compute_kv_cache_shape(total_gpu_blocks)
        self._key_cache_shape = key_shape
        self._value_cache_shape = val_shape

        offload_logger.info(
            f"[DEBUG: offload] Cache info initialized: num_layers={self._num_layers}, "
            f"total_gpu_blocks={total_gpu_blocks}, key_shape={self._key_cache_shape}, "
            f"value_shape={self._value_cache_shape}"
        )

    def _get_cache_tensor_name(self, layer_id: int, is_key: bool) -> str:
        """获取共享内存中cache tensor的名称"""
        prefix = "key_caches" if is_key else "value_caches"
        return f"{prefix}_{layer_id}_rank{self._local_rank}.device{self._device_id}"

    def _get_cache_scale_tensor_name(self, layer_id: int, is_key: bool) -> str:
        """获取共享内存中cache scale tensor的名称（用于量化）"""
        prefix = "key_cache_scales" if is_key else "value_cache_scales"
        return f"{prefix}_{layer_id}_rank{self._local_rank}.device{self._device_id}"

    def _get_gpu_cache_tensor(self, layer_id: int, is_key: bool) -> Optional[paddle.Tensor]:
        """
        从共享内存获取GPU cache tensor的引用

        Args:
            layer_id: 层ID
            is_key: 是否为key cache

        Returns:
            paddle.Tensor: GPU cache tensor引用，失败返回None
        """
        if share_external_data_ is None:
            offload_logger.error("[DEBUG: offload] share_external_data_ is not available")
            return None

        try:
            tensor_name = self._get_cache_tensor_name(layer_id, is_key)
            cache_shape = self._key_cache_shape if is_key else self._value_cache_shape

            if cache_shape is None or len(cache_shape) == 0:
                offload_logger.error(f"[DEBUG: offload] cache_shape is None or empty for layer {layer_id}")
                return None

            # 创建空tensor并通过share_external_data_绑定到共享内存
            dtype = self._cache_dtype if self._cache_dtype else "bfloat16"
            if dtype == "block_wise_fp8":
                dtype = paddle.float8_e4m3fn
            elif dtype == "int4_zp":
                dtype = paddle.int8
            elif dtype == "bfloat16":
                dtype = paddle.bfloat16
            else:
                dtype = paddle.float16

            empty_tensor = paddle.empty(shape=[], dtype=dtype)
            try:
                cache_tensor = share_external_data_(empty_tensor, tensor_name, cache_shape, True)
            except Exception as e:
                offload_logger.error(f"[DEBUG: offload] share_external_data_ FAILED for {tensor_name}: {e}")
                raise

            return cache_tensor

        except Exception as e:
            offload_logger.error(f"[DEBUG: offload] Failed to get GPU cache tensor for layer {layer_id}: {e}")
            return None

    def _get_gpu_cache_scale_tensor(self, layer_id: int, is_key: bool) -> Optional[paddle.Tensor]:
        """
        从共享内存获取GPU cache scale tensor的引用（用于量化）
        """
        if share_external_data_ is None:
            return None

        try:
            tensor_name = self._get_cache_scale_tensor_name(layer_id, is_key)
            # scale shape: [num_blocks, num_heads, block_size]
            scale_shape = [
                self._key_cache_shape[0],
                self._key_cache_shape[1],
                self._key_cache_shape[2],
            ]

            empty_tensor = paddle.empty(shape=[], dtype=paddle.float32)
            scale_tensor = share_external_data_(empty_tensor, tensor_name, scale_shape, True)

            return scale_tensor

        except Exception as e:
            offload_logger.error(f"[DEBUG: offload] Failed to get GPU scale tensor for layer {layer_id}: {e}")
            return None

    def _compute_kv_cache_shape(self, max_block_num):
        """
        计算 KV Cache 的 shape

        Args:
            max_block_num: 最大 block 数量

        Returns:
            tuple: (key_cache_shape, val_cache_shape)
        """
        try:
            from fastdeploy.model_executor.layers.attention import get_attention_backend

            config = self.cache_manager.config
            cache_config = self.cache_manager.cache_config

            attn_cls = get_attention_backend()
            tp_size = getattr(config.parallel_config, "tensor_parallel_size", 1)
            num_heads = config.model_config.num_attention_heads // tp_size
            kv_num_heads = max(
                1,
                int(config.model_config.num_key_value_heads) // tp_size,
            )
            head_dim = config.model_config.head_dim

            kv_cache_quant_type = None
            if (
                config.quant_config
                and hasattr(config.quant_config, "kv_cache_quant_type")
                and config.quant_config.kv_cache_quant_type is not None
            ):
                kv_cache_quant_type = config.quant_config.kv_cache_quant_type

            encoder_block_shape_q = 64
            decoder_block_shape_q = 16
            key_cache_shape, value_cache_shape = attn_cls(
                config,
                kv_num_heads=kv_num_heads,
                num_heads=num_heads,
                head_dim=head_dim,
                encoder_block_shape_q=encoder_block_shape_q,
                decoder_block_shape_q=decoder_block_shape_q,
            ).get_kv_cache_shape(max_num_blocks=max_block_num, kv_cache_quant_type=kv_cache_quant_type)

            offload_logger.info(
                f"[DEBUG: offload] Computed key_cache_shape: {key_cache_shape}, value_cache_shape: {value_cache_shape}"
            )
            return key_cache_shape, value_cache_shape

        except Exception as e:
            offload_logger.error(f"[DEBUG: offload] Failed to compute kv_cache_shape: {e}")
            return None, None

    # ==================== 判断接口 ====================

    def can_offload(self, request: Request) -> bool:
        """
        检查请求是否可以被offload

        条件:
        1. offload功能已启用
        2. 请求未被offload过
        3. 请求有block_tables可被offload
        4. 请求处于decode阶段（num_computed_tokens >= need_prefill_tokens）
        5. need_prefill_tokens已初始化
        6. CPU内存充足

        注意: decode阶段判定和存储空间检查由调用者或offload_req负责
        """
        # DEBUG: can_offload入口检查
        offload_logger.debug(
            f"[DEBUG: can_offload] Checking request {request.request_id}, "
            f"enable_offload={self.enable_offload}, is_offloaded={request.is_offloaded}"
        )

        if not self.enable_offload:
            offload_logger.debug(f"[DEBUG: can_offload] {request.request_id}: offload disabled")
            return False

        if request.is_offloaded:
            offload_logger.debug(f"[DEBUG: can_offload] {request.request_id}: already offloaded")
            return False

        # 检查是否有可被offload的blocks
        if not request.block_tables:
            offload_logger.debug(f"[DEBUG: can_offload] {request.request_id}: no block_tables")
            return False

        # 新增：检查need_prefill_tokens是否已初始化
        if request.need_prefill_tokens is None:
            offload_logger.warning(
                f"[DEBUG: can_offload] {request.request_id}: need_prefill_tokens is None, cannot offload"
            )
            return False

        # 新增：检查请求是否处于decode阶段
        if request.num_computed_tokens < request.need_prefill_tokens:
            offload_logger.warning(
                f"[DEBUG: can_offload] {request.request_id} is not in decode phase, "
                f"num_computed_tokens={request.num_computed_tokens}, "
                f"need_prefill_tokens={request.need_prefill_tokens}, cannot offload"
            )
            return False

        # 注意：新方法 .to("cpu") 不需要预先分配的 CPU blocks
        # 因此移除对 cpu_free_block_list 的检查

        offload_logger.debug(f"[DEBUG: can_offload] {request.request_id}: can offload = True")
        return True

    def can_resume(self, request: Request) -> bool:
        """
        检查请求是否可以被恢复

        条件:
        1. offload功能已启用
        2. 请求存在offloaded信息
        3. 存在有效的KV Cache副本(cpu_copy或ssd_copy)
        4. GPU内存充足
        """
        if not self.enable_offload:
            return False

        if request.request_id not in self._offloaded_requests:
            return False

        offloaded_info = self._offloaded_requests.get(request.request_id)
        if offloaded_info is None:
            return False

        # 检查是否存在有效的KV Cache副本
        storage_level = offloaded_info.get("storage_level")
        if storage_level == self.STORAGE_LEVEL_CPU:
            if offloaded_info.get("kv_cache_cpu") is None:
                return False
        elif storage_level == self.STORAGE_LEVEL_SSD:
            storage_path = offloaded_info.get("storage_path")
            if not storage_path or not os.path.exists(storage_path):
                return False
        else:
            return False

        # 检查GPU内存是否充足
        num_blocks_needed = offloaded_info.get("num_blocks_needed", 0)
        if self.cache_manager is None:
            return False

        return self.cache_manager.can_allocate_gpu_blocks(num_blocks_needed)

    def offload_decode(
        self, running_requests: List[Request], min_steps: int = 20
    ) -> Tuple[List[Request], List[Request]]:
        """
        批量offload decode请求,直到当前batch能运行min_steps个step为止

        注意: 调度策略由外层ResourceManager决定,此函数仅负责:
        1. 检测请求是否处于decode阶段
        2. 执行offload操作
        3. 保存KV Cache相关信息
        4. 更新请求状态

        Args:
            running_requests: 待offload的请求列表(已按调度策略排序)
            min_steps: 最小运行step数(默认20)

        Returns:
            Tuple[offloaded_reqs, abort_reqs]:
            - offloaded_reqs: 成功offload的请求
            - abort_reqs: offload失败的请求(需要调用者abort处理)
        """
        # DEBUG: offload - 入口调试
        offload_logger.info(
            f"[DEBUG: offload_decode] offload_decode called, enable_offload={self.enable_offload}, "
            f"num_requests={len(running_requests)}"
        )

        if not self.enable_offload:
            return [], []

        offloaded_reqs = []
        abort_reqs = []
        remaining_count = len(running_requests)

        for req in running_requests:
            # 非decode阶段的请求不应出现在这里,记录warning但不处理
            # 请求状态未修改,调用者可重新调度
            if req.num_computed_tokens < req.need_prefill_tokens:
                offload_logger.warning(
                    f"[DEBUG: offload_decode] Request {req.request_id} is not in decode phase, "
                    f"num_computed_tokens={req.num_computed_tokens}, "
                    f"need_prefill_tokens={req.need_prefill_tokens}"
                )
                continue

            # DEBUG: offload - can_offload 检查
            can_offload_result = self.can_offload(req)
            offload_logger.info(
                f"[DEBUG: offload_decode] can_offload({req.request_id})={can_offload_result}, "
                f"is_offloaded={req.is_offloaded}, block_tables={len(req.block_tables) if req.block_tables else 0}"
            )

            if not can_offload_result:
                continue

            # 执行offload
            if self.offload_req(req):
                offloaded_reqs.append(req)
                remaining_count -= 1
                offload_logger.info(f"[DEBUG: offload_decode] Successfully offloaded request {req.request_id}")
            else:
                # offload失败,返回给调用者处理(需要abort)
                abort_reqs.append(req)
                offload_logger.warning(f"[DEBUG: offload_decode] Failed to offload request {req.request_id}")

            if self.cache_manager is not None and remaining_count > 0:
                block_size = self.cache_manager.cache_config.block_size
                blocks_needed_per_request = (min_steps + block_size - 1) // block_size
                total_blocks_needed = remaining_count * blocks_needed_per_request
                current_free_blocks = len(getattr(self.cache_manager, "gpu_free_block_list", []))
                if current_free_blocks >= total_blocks_needed:
                    offload_logger.info(
                        f"[DEBUG: offload_decode] Memory sufficient after offloading "
                        f"{len(offloaded_reqs)} requests, remaining={remaining_count}, "
                        f"free_blocks={current_free_blocks}, needed={total_blocks_needed}"
                    )
                    break

        return offloaded_reqs, abort_reqs

    # ==================== 单请求多级Offload接口 ====================

    def offload_req(self, request: Request) -> bool:
        """
        指定请求触发多级卸载(L1→L2→L3)

        执行步骤:
        1. 检查请求是否处于decode阶段
        2. 尝试L2 offload(CPU内存)
        3. 如果L2内存不足,触发L3 offload(SSD存储)
        4. 释放对应GPU blocks
        5. 更新请求状态
        """
        if not self.enable_offload:
            return False

        # 检查是否已经被offload
        if request.is_offloaded:
            offload_logger.warning(f"[DEBUG: offload_req] Request {request.request_id} already offloaded")
            return False

        # 初始化cache信息
        self._init_cache_info()

        # 尝试L2 offload (CPU)
        storage_level = self.STORAGE_LEVEL_CPU
        kv_cache_cpu = None

        try:
            kv_cache_cpu = self.get_cpu_copy(request)
            if kv_cache_cpu is None:
                # CPU offload失败,尝试SSD offload
                # 注意: SSD offload同样需要先获取数据,这里直接返回失败
                # 如果未来需要SSD offload,需要实现直接GPU->SSD的传输
                offload_logger.error(
                    f"[DEBUG: offload_req] CPU offload failed for {request.request_id}, " f"no available fallback"
                )
                return False
        except Exception as e:
            offload_logger.error(f"[DEBUG: offload_req] CPU offload failed: {e}")
            return False

        # 如果需要L3,保存到SSD (当前kv_cache_cpu已包含数据)
        storage_path = None
        if storage_level == self.STORAGE_LEVEL_SSD:
            try:
                storage_path = self.save_to_storage(kv_cache_cpu)
                if storage_path is None:
                    offload_logger.error(f"[DEBUG: offload_req] SSD offload failed for {request.request_id}")
                    # 清理已分配的CPU blocks
                    if kv_cache_cpu and self.cache_manager:
                        self.cache_manager.recycle_cpu_blocks(kv_cache_cpu.get("cpu_block_ids", []))
                    return False
                if kv_cache_cpu is not None:
                    del kv_cache_cpu
                    kv_cache_cpu = None
            except Exception as e:
                offload_logger.error(f"[DEBUG: offload_req] SSD offload failed: {e}")
                return False

        # 保存offload信息 - 在释放GPU blocks之前保存
        with self._lock:
            original_block_tables = list(request.block_tables) if request.block_tables else []

            # 新增：确保need_prefill_tokens不为None，提供默认值
            need_prefill_tokens_value = request.need_prefill_tokens
            if need_prefill_tokens_value is None:
                # 如果need_prefill_tokens未初始化，使用prompt_token_ids_len作为默认值
                need_prefill_tokens_value = request.prompt_token_ids_len if request.prompt_token_ids_len else 0
                offload_logger.warning(
                    f"[DEBUG: offload_req] Request {request.request_id} need_prefill_tokens is None during offload, "
                    f"using default value: {need_prefill_tokens_value}"
                )

            self._offloaded_requests[request.request_id] = {
                "kv_cache_cpu": kv_cache_cpu,
                "storage_path": storage_path,
                "storage_level": storage_level,
                "num_tokens": request.num_total_tokens,
                "num_blocks_needed": len(original_block_tables),
                "output_token_ids": list(request.output_token_ids),
                "num_computed_tokens": request.num_computed_tokens,
                "need_prefill_tokens": need_prefill_tokens_value,
                "prompt_token_ids": list(request.prompt_token_ids) if request.prompt_token_ids else None,
                "prompt_token_ids_len": request.prompt_token_ids_len,
                "sampling_params": request.sampling_params,
                "block_tables": original_block_tables,
            }

        # 释放GPU blocks
        self.release_gpu_blocks(request)

        # 更新请求状态
        request.status = RequestStatus.PREEMPTED
        request.is_offloaded = True

        offload_logger.info(
            f"[DEBUG: offload_req] Request {request.request_id} offloaded to {storage_level}, "
            f"num_tokens={request.num_total_tokens}, output_tokens={len(request.output_token_ids)}, "
            f"blocks_needed={len(original_block_tables)}"
        )

        return True

    def offload_kv_cache(self, request: Request, target_level: str = "L2") -> bool:
        """
        调用多级memory offload工具函数,并释放相应GPU blocks

        用于扩展性调用,可单独对指定请求进行KV Cache offload

        Args:
            request: 需要offload的请求
            target_level: 目标存储层级("L2"=CPU, "L3"=SSD)

        Returns:
            bool: offload是否成功
        """
        try:
            if target_level == self.STORAGE_LEVEL_CPU:
                kv_cache_cpu = self.get_cpu_copy(request)
                return kv_cache_cpu is not None
            elif target_level == self.STORAGE_LEVEL_SSD:
                kv_cache_cpu = self.get_cpu_copy(request)
                storage_path = self.save_to_storage(kv_cache_cpu)
                if kv_cache_cpu is not None:
                    del kv_cache_cpu
                return storage_path is not None
            else:
                offload_logger.error(f"[DEBUG: offload_kv_cache] Invalid target_level: {target_level}")
                return False
        except Exception as e:
            offload_logger.error(f"[DEBUG: offload_kv_cache] offload_kv_cache failed: {e}")
            return False

    def release_gpu_blocks(self, request: Request) -> None:
        if self.cache_manager is None:
            return

        if request.block_tables:
            blocks_to_release = list(request.block_tables)
            offload_logger.info(
                f"[DEBUG: release_gpu_blocks] Releasing {len(blocks_to_release)} blocks for request {request.request_id}"
            )
            self.cache_manager.recycle_gpu_blocks(blocks_to_release, request.request_id)
            request.block_tables = []

    # ==================== CPU Memory Offload接口 ====================

    def get_cpu_copy(self, request: Request) -> Optional[dict]:
        """
        从GPU获取KV Cache的CPU副本

        使用 paddle.Tensor.to("cpu") 将KV cache从GPU复制到CPU内存

        Args:
            request: 需要offload的请求

        Returns:
            dict: 包含CPU上KV cache数据的字典，失败返回None
            {
                "key_caches": List[paddle.Tensor],  # CPU上的key cache列表
                "value_caches": List[paddle.Tensor],  # CPU上的value cache列表
                "key_scales": List[paddle.Tensor],  # 可选，用于量化
                "value_scales": List[paddle.Tensor],  # 可选，用于量化
                "block_ids": List[int],  # 对应的block IDs
                "num_blocks": int,
            }
        """
        import time

        start_time = time.time()

        if not request.block_tables:
            offload_logger.warning(f"[DEBUG: get_cpu_copy] {request.request_id}: no block_tables")
            return None

        self._init_cache_info()

        if self._key_cache_shape is None:
            offload_logger.error("[DEBUG: get_cpu_copy] key_cache_shape is not initialized")
            return None

        try:
            block_ids = list(request.block_tables)
            num_blocks = len(block_ids)

            key_caches_cpu = []
            value_caches_cpu = []
            key_scales_cpu = []
            value_scales_cpu = []

            offload_logger.info(
                f"[DEBUG: get_cpu_copy] Copying KV cache for request {request.request_id}, "
                f"num_layers={self._num_layers}, num_blocks={num_blocks}, block_ids={block_ids}"
            )

            for layer_id in range(self._num_layers):
                # 获取GPU上的key cache
                key_cache_gpu = self._get_gpu_cache_tensor(layer_id, is_key=True)
                if key_cache_gpu is None:
                    offload_logger.error(f"[DEBUG: get_cpu_copy] Failed to get key cache for layer {layer_id}")
                    return None

                # 获取需要的blocks数据
                key_cache_blocks = []
                for block_id in block_ids:
                    if block_id < key_cache_gpu.shape[0]:
                        key_cache_blocks.append(key_cache_gpu[block_id])
                    else:
                        offload_logger.error(
                            f"[DEBUG: get_cpu_copy] Block {block_id} out of range for key cache (shape={key_cache_gpu.shape})"
                        )
                        return None

                # 拼接并复制到CPU
                key_cache_layer = (
                    paddle.stack(key_cache_blocks) if len(key_cache_blocks) > 1 else key_cache_blocks[0].unsqueeze(0)
                )
                key_cache_cpu = key_cache_layer.to("cpu")
                key_caches_cpu.append(key_cache_cpu)

                # 获取value cache（如果有）
                if self._value_cache_shape and len(self._value_cache_shape) > 0:
                    value_cache_gpu = self._get_gpu_cache_tensor(layer_id, is_key=False)
                    if value_cache_gpu is not None:
                        value_cache_blocks = []
                        for block_id in block_ids:
                            if block_id < value_cache_gpu.shape[0]:
                                value_cache_blocks.append(value_cache_gpu[block_id])
                            else:
                                offload_logger.error(
                                    f"[DEBUG: get_cpu_copy] Block {block_id} out of range for value cache"
                                )
                                return None

                        value_cache_layer = (
                            paddle.stack(value_cache_blocks)
                            if len(value_cache_blocks) > 1
                            else value_cache_blocks[0].unsqueeze(0)
                        )
                        value_cache_cpu = value_cache_layer.to("cpu")
                        value_caches_cpu.append(value_cache_cpu)

                # 获取scale tensors（用于量化）
                if self._cache_dtype == "block_wise_fp8":
                    key_scale_gpu = self._get_gpu_cache_scale_tensor(layer_id, is_key=True)
                    if key_scale_gpu is not None:
                        key_scale_blocks = [key_scale_gpu[block_id] for block_id in block_ids]
                        key_scale_layer = (
                            paddle.stack(key_scale_blocks)
                            if len(key_scale_blocks) > 1
                            else key_scale_blocks[0].unsqueeze(0)
                        )
                        key_scales_cpu.append(key_scale_layer.to("cpu"))

                    value_scale_gpu = self._get_gpu_cache_scale_tensor(layer_id, is_key=False)
                    if value_scale_gpu is not None:
                        value_scale_blocks = [value_scale_gpu[block_id] for block_id in block_ids]
                        value_scale_layer = (
                            paddle.stack(value_scale_blocks)
                            if len(value_scale_blocks) > 1
                            else value_scale_blocks[0].unsqueeze(0)
                        )
                        value_scales_cpu.append(value_scale_layer.to("cpu"))

            result = {
                "key_caches": key_caches_cpu,
                "value_caches": value_caches_cpu,
                "key_scales": key_scales_cpu if key_scales_cpu else None,
                "value_scales": value_scales_cpu if value_scales_cpu else None,
                "block_ids": block_ids,
                "num_blocks": num_blocks,
            }

            elapsed_time = time.time() - start_time
            offload_logger.info(
                f"[DEBUG: get_cpu_copy] Successfully copied KV cache to CPU for request {request.request_id}, "
                f"key_cache_shape={key_caches_cpu[0].shape if key_caches_cpu else None}, "
                f"elapsed_time={elapsed_time:.4f}s"
            )

            return result

        except Exception as e:
            elapsed_time = time.time() - start_time
            offload_logger.error(
                f"[DEBUG: get_cpu_copy] Failed to copy KV cache to CPU: {e}, elapsed_time={elapsed_time:.4f}s"
            )
            return None

    def load_cpu_copy(self, kv_cache_cpu: dict, request: Request) -> bool:
        """
        将CPU上的KV Cache加载回GPU

        使用 paddle.Tensor.to(device) 将KV cache从CPU复制回GPU共享内存

        Args:
            kv_cache_cpu: CPU上的KV Cache信息（get_cpu_copy返回的字典）
            request: 需要恢复的请求

        Returns:
            bool: 加载是否成功
        """
        if kv_cache_cpu is None:
            offload_logger.error("[DEBUG: load_cpu_copy] kv_cache_cpu is None")
            return False

        if not request.block_tables:
            offload_logger.error(f"[DEBUG: load_cpu_copy] {request.request_id}: no block_tables allocated")
            return False

        start_time = time.time()
        try:
            key_caches_cpu = kv_cache_cpu.get("key_caches")
            value_caches_cpu = kv_cache_cpu.get("value_caches")
            key_scales_cpu = kv_cache_cpu.get("key_scales")
            value_scales_cpu = kv_cache_cpu.get("value_scales")

            if not key_caches_cpu:
                offload_logger.error("[DEBUG: load_cpu_copy] key_caches is empty")
                return False

            device = f"gpu:{self._device_id}"
            new_block_ids = list(request.block_tables)

            offload_logger.info(
                f"[DEBUG: load_cpu_copy] Loading KV cache from CPU for request {request.request_id}, "
                f"num_layers={len(key_caches_cpu)}, num_blocks={len(new_block_ids)}"
            )

            for layer_id, key_cache_cpu in enumerate(key_caches_cpu):
                # 获取GPU上的key cache
                key_cache_gpu = self._get_gpu_cache_tensor(layer_id, is_key=True)
                if key_cache_gpu is None:
                    offload_logger.error(f"[DEBUG: load_cpu_copy] Failed to get GPU key cache for layer {layer_id}")
                    return False

                # 复制到GPU
                key_cache_gpu_data = key_cache_cpu.to(device)

                # 写入到新的block IDs
                for idx, block_id in enumerate(new_block_ids):
                    if block_id < key_cache_gpu.shape[0] and idx < key_cache_gpu_data.shape[0]:
                        key_cache_gpu[block_id] = key_cache_gpu_data[idx]
                    else:
                        offload_logger.error(
                            f"[DEBUG: load_cpu_copy] Block ID out of range: block_id={block_id}, "
                            f"gpu_shape={key_cache_gpu.shape}, idx={idx}"
                        )
                        return False

                # 复制value cache（如果有）
                if value_caches_cpu and layer_id < len(value_caches_cpu):
                    value_cache_cpu = value_caches_cpu[layer_id]
                    value_cache_gpu = self._get_gpu_cache_tensor(layer_id, is_key=False)
                    if value_cache_gpu is not None:
                        value_cache_gpu_data = value_cache_cpu.to(device)
                        for idx, block_id in enumerate(new_block_ids):
                            if block_id < value_cache_gpu.shape[0] and idx < value_cache_gpu_data.shape[0]:
                                value_cache_gpu[block_id] = value_cache_gpu_data[idx]
                            else:
                                offload_logger.error(
                                    f"[DEBUG: load_cpu_copy] Block ID out of range for value: block_id={block_id}"
                                )
                                return False

                # 复制scales（用于量化）
                if key_scales_cpu and layer_id < len(key_scales_cpu):
                    key_scale_cpu = key_scales_cpu[layer_id]
                    key_scale_gpu = self._get_gpu_cache_scale_tensor(layer_id, is_key=True)
                    if key_scale_gpu is not None:
                        key_scale_gpu_data = key_scale_cpu.to(device)
                        for idx, block_id in enumerate(new_block_ids):
                            if block_id < key_scale_gpu.shape[0] and idx < key_scale_gpu_data.shape[0]:
                                key_scale_gpu[block_id] = key_scale_gpu_data[idx]

                if value_scales_cpu and layer_id < len(value_scales_cpu):
                    value_scale_cpu = value_scales_cpu[layer_id]
                    value_scale_gpu = self._get_gpu_cache_scale_tensor(layer_id, is_key=False)
                    if value_scale_gpu is not None:
                        value_scale_gpu_data = value_scale_cpu.to(device)
                        for idx, block_id in enumerate(new_block_ids):
                            if block_id < value_scale_gpu.shape[0] and idx < value_scale_gpu_data.shape[0]:
                                value_scale_gpu[block_id] = value_scale_gpu_data[idx]

            elapsed_time = time.time() - start_time
            offload_logger.info(
                f"[DEBUG: load_cpu_copy] Successfully loaded KV cache to GPU for request {request.request_id}, "
                f"elapsed_time={elapsed_time:.4f}s"
            )
            return True

        except Exception as e:
            elapsed_time = time.time() - start_time
            offload_logger.error(
                f"[DEBUG: load_cpu_copy] Failed to load KV cache to GPU: {e}, elapsed_time={elapsed_time:.4f}s"
            )
            return False

    # ==================== SSD Storage Offload接口 ====================

    def save_to_storage(self, kv_cache_cpu) -> Optional[str]:
        """
        将CPU上的KV Cache保存到SSD存储

        Args:
            kv_cache_cpu: CPU上的KV Cache信息

        Returns:
            str: 存储文件路径,失败返回None
        """
        try:
            if kv_cache_cpu is None:
                return None

            storage_file = f"kv_cache_{uuid.uuid4().hex}.pdparams"
            storage_path = os.path.join(self.storage_path, storage_file)

            # 保存KV Cache信息到文件
            paddle.save(kv_cache_cpu, storage_path)

            offload_logger.info(f"[DEBUG: offload_save_to_storage] Saved KV cache to {storage_path}")
            return storage_path

        except Exception as e:
            offload_logger.error(f"[DEBUG: offload_save_to_storage] save_to_storage failed: {e}")
            return None

    def load_from_storage(self, storage_path: str) -> Optional[dict]:
        """
        从SSD存储加载KV Cache到CPU

        Args:
            storage_path: 存储文件路径

        Returns:
            dict: CPU上的KV Cache信息,失败返回None
        """
        try:
            if not os.path.exists(storage_path):
                offload_logger.error(f"[DEBUG: off_load_save_to_storage] Storage file not found: {storage_path}")
                return None

            kv_cache_cpu = paddle.load(storage_path)
            offload_logger.info(f"[DEBUG: off_load_save_to_storage] Loaded KV cache from {storage_path}")
            return kv_cache_cpu

        except Exception as e:
            offload_logger.error(f"[DEBUG: off_load_save_to_storage] load_from_storage failed: {e}")
            return None

    # ==================== Resume接口 ====================

    def resume_decode(self, request: Request) -> Tuple[bool, Optional[int]]:
        """
        恢复被offload的请求到GPU

        执行步骤:
        1. 检查GPU是否有足够内存
        2. 根据存储层级选择恢复路径(L2/L3)
        3. 验证cache数据完整性
        4. 检查是否为decode阶段(token_num > prefill_token_num)
        5. 分配GPU blocks并加载数据
        6. 恢复请求状态

        Returns:
            Tuple[bool, Optional[int]]:
            - bool: resume是否成功
            - int: cache中的token数量(即使resume失败也返回，用于重新计算)
        """
        start_time = time.time()
        if not self.enable_offload:
            return False, None

        # 使用锁保护offloaded_requests的读取
        with self._lock:
            if request.request_id not in self._offloaded_requests:
                offload_logger.warning(f"[DEBUG: resume_decode] Request {request.request_id} is not offloaded")
                return False, None

            offloaded_info = self._offloaded_requests.get(request.request_id)
            if offloaded_info is None:
                return False, None

            # 复制需要的信息，避免长时间持有锁
            storage_level = offloaded_info["storage_level"]
            num_blocks_needed = offloaded_info["num_blocks_needed"]
            saved_num_tokens = offloaded_info["num_tokens"]
            saved_num_computed_tokens = offloaded_info["num_computed_tokens"]
            saved_need_prefill_tokens = offloaded_info["need_prefill_tokens"]
            storage_path = offloaded_info.get("storage_path")
            # 对于CPU层级，需要复制kv_cache_cpu引用（用于完整性检查）
            kv_cache_cpu_ref = offloaded_info.get("kv_cache_cpu") if storage_level == self.STORAGE_LEVEL_CPU else None
            cache_valid_flag = offloaded_info.get("cache_valid", True)
            # 复制output_token_ids和need_prefill_tokens用于恢复
            output_token_ids = list(offloaded_info.get("output_token_ids", []))
            need_prefill_tokens = offloaded_info.get("need_prefill_tokens")

        # 检查是否为decode阶段
        if saved_num_computed_tokens <= saved_need_prefill_tokens:
            offload_logger.warning(
                f"[DEBUG: resume_decode] Request {request.request_id} is not in decode phase "
                f"(num_computed_tokens={saved_num_computed_tokens}, "
                f"need_prefill_tokens={saved_need_prefill_tokens}), "
                f"should recompute instead of resume"
            )
            # 返回token数，让调用者决定是否重新计算
            return False, saved_num_computed_tokens

        if self.cache_manager is None:
            return False, saved_num_computed_tokens

        if not self.cache_manager.can_allocate_gpu_blocks(num_blocks_needed):
            offload_logger.warning(
                f"[DEBUG: resume_decode] Insufficient GPU memory for request {request.request_id}, "
                f"need {num_blocks_needed} blocks"
            )
            return False, saved_num_computed_tokens

        # 检查cache_valid_flag，如果之前已经标记为无效，直接返回失败
        if not cache_valid_flag:
            offload_logger.warning(
                f"[DEBUG: resume_decode] Cache for request {request.request_id} is marked as invalid"
            )
            return False, saved_num_computed_tokens

        try:
            kv_cache_cpu = None
            cache_valid = False

            # 根据存储层级恢复
            if storage_level == self.STORAGE_LEVEL_CPU:
                kv_cache_cpu = kv_cache_cpu_ref
                if kv_cache_cpu is None:
                    offload_logger.error(f"[DEBUG: resume_decode] No CPU cache found for {request.request_id}")
                else:
                    # 构建临时的offloaded_info用于验证
                    temp_offloaded_info = {
                        "num_blocks_needed": num_blocks_needed,
                        "num_tokens": saved_num_tokens,
                    }
                    cache_valid = self._verify_cache_integrity(kv_cache_cpu, temp_offloaded_info)

            elif storage_level == self.STORAGE_LEVEL_SSD:
                if not storage_path or not os.path.exists(storage_path):
                    offload_logger.error(f"[DEBUG: resume_decode] No SSD storage path for {request.request_id}")
                else:
                    kv_cache_cpu = self.load_from_storage(storage_path)
                    if kv_cache_cpu is not None:
                        # 构建临时的offloaded_info用于验证
                        temp_offloaded_info = {
                            "num_blocks_needed": num_blocks_needed,
                            "num_tokens": saved_num_tokens,
                        }
                        cache_valid = self._verify_cache_integrity(kv_cache_cpu, temp_offloaded_info)
                    else:
                        offload_logger.error(f"[DEBUG: resume_decode] Failed to load from storage: {storage_path}")

            # 验证cache完整性
            if not cache_valid:
                offload_logger.error(
                    f"[DEBUG: resume_decode] Cache integrity check failed for {request.request_id}, "
                    f"saved_tokens={saved_num_tokens}, cache may be corrupted"
                )
                # 清理无效的cache资源
                if kv_cache_cpu is not None and isinstance(kv_cache_cpu, dict):
                    cpu_block_ids = kv_cache_cpu.get("cpu_block_ids", [])
                    if cpu_block_ids and self.cache_manager:
                        self.cache_manager.recycle_cpu_blocks(cpu_block_ids)

                # 更新offloaded_info标记为无效，避免后续再次尝试使用无效的CPU blocks
                with self._lock:
                    if request.request_id in self._offloaded_requests:
                        offloaded_info = self._offloaded_requests[request.request_id]
                        offloaded_info["kv_cache_cpu"] = None
                        offloaded_info["cache_valid"] = False
                        # 不删除offloaded_info，保留其他元数据供后续使用

                # 返回token数，让调用者可以重新计算
                return False, saved_num_computed_tokens

            # 分配GPU blocks
            new_block_ids = self.cache_manager.allocate_gpu_blocks(num_blocks_needed, request.request_id)
            request.block_tables = new_block_ids

            # 更新kv_cache_cpu中的block_ids为新的分配
            if kv_cache_cpu is not None:
                kv_cache_cpu["block_ids"] = new_block_ids

            # 加载cache到GPU
            if not self.load_cpu_copy(kv_cache_cpu, request):
                offload_logger.error(f"[DEBUG: resume_decode] Failed to load CPU copy to GPU for {request.request_id}")
                # 释放已分配的blocks
                self.cache_manager.recycle_gpu_blocks(new_block_ids, request.request_id)
                request.block_tables = []
                return False, saved_num_computed_tokens

            # 对于SSD层级，清理临时内存
            if storage_level == self.STORAGE_LEVEL_SSD:
                del kv_cache_cpu

            # 恢复请求状态
            request.output_token_ids = output_token_ids
            request.num_computed_tokens = saved_num_computed_tokens
            request.need_prefill_tokens = need_prefill_tokens
            request.status = RequestStatus.RUNNING
            request.is_offloaded = False

            # 在 resume 成功时，清理 abort 标志，避免后续生成错误的 RequestOutput
            # 注意：Request 类没有 outputs 属性，outputs 是 RequestOutput 的属性
            # 这里我们设置一个标记，表示该请求已成功恢复，后续处理不应生成 abort 的 RequestOutput
            offload_logger.info(
                f"[DEBUG: resume_decode] Request {request.request_id} resumed successfully, "
                f"output_tokens={len(output_token_ids)}, idx={request.idx}"
            )

            # 清理offloaded信息
            with self._lock:
                self._offloaded_requests.pop(request.request_id, None)

            # 清理SSD存储文件
            if storage_level == self.STORAGE_LEVEL_SSD and storage_path:
                try:
                    os.remove(storage_path)
                except Exception as e:
                    offload_logger.warning(f"[DEBUG: resume_decode] Failed to delete storage file: {e}")

            elapsed_time = time.time() - start_time
            offload_logger.info(
                f"[DEBUG: resume_decode] Resumed request {request.request_id} from {storage_level}, "
                f"output_tokens={len(request.output_token_ids)}, elapsed_time={elapsed_time:.4f}s"
            )

            # 尝试预取其他 SSD 数据到 CPU
            self.prefetch_ssd_to_cpu()

            return True, saved_num_computed_tokens

        except Exception as e:
            elapsed_time = time.time() - start_time
            offload_logger.error(
                f"[DEBUG: resume_decode] Failed to resume request {request.request_id}: {e}, elapsed_time={elapsed_time:.4f}s"
            )
            # 失败时保持offload状态,下次可以重试
            return False, saved_num_computed_tokens

    def _verify_cache_integrity(self, kv_cache_cpu: dict, offloaded_info: dict) -> bool:
        """
        验证cache数据的完整性

        Args:
            kv_cache_cpu: CPU上的KV Cache信息
            offloaded_info: 保存的offload信息

        Returns:
            bool: cache是否有效
        """
        if kv_cache_cpu is None:
            return False

        # 检查必要的字段
        cpu_block_ids = kv_cache_cpu.get("block_ids", [])
        num_blocks = kv_cache_cpu.get("num_blocks", 0)

        if not cpu_block_ids or num_blocks == 0:
            offload_logger.warning("[DEBUG: offload] Cache integrity check: missing block_ids or num_blocks")
            return False

        # 检查block数量是否匹配
        expected_num_blocks = offloaded_info.get("num_blocks_needed", 0)
        if len(cpu_block_ids) != num_blocks or num_blocks != expected_num_blocks:
            offload_logger.warning(
                f"[DEBUG: offload] Cache integrity check: block count mismatch, "
                f"expected={expected_num_blocks}, actual={num_blocks}"
            )
            return False

        # 检查key_caches是否存在
        key_caches = kv_cache_cpu.get("key_caches")
        if not key_caches or len(key_caches) == 0:
            offload_logger.warning("[DEBUG: offload] Cache integrity check: missing key_caches")
            return False

        # 检查token数量是否一致
        saved_num_tokens = offloaded_info.get("num_tokens", 0)
        block_size = self.cache_manager.cache_config.block_size if self.cache_manager else 64
        actual_max_tokens = num_blocks * block_size
        if saved_num_tokens > actual_max_tokens:
            offload_logger.warning(
                f"[DEBUG: offload] Cache integrity check: token count exceeds capacity, "
                f"saved_tokens={saved_num_tokens}, max_capacity={actual_max_tokens}"
            )
            return False

        offload_logger.debug(f"Cache integrity check passed: num_blocks={num_blocks}, num_tokens={saved_num_tokens}")
        return True

    # ==================== 辅助接口 ====================

    def cleanup_offloaded_request(self, request_id: str) -> None:
        """清理被offload请求的缓存(请求完成时调用)"""
        with self._lock:
            if request_id not in self._offloaded_requests:
                return

            offloaded_info = self._offloaded_requests[request_id]

            # 清理CPU内存中的KV cache tensors
            kv_cache_cpu = offloaded_info.get("kv_cache_cpu")
            if kv_cache_cpu is not None:
                # 显式删除CPU tensors释放内存
                for key in ["key_caches", "value_caches", "key_scales", "value_scales"]:
                    cache_list = kv_cache_cpu.get(key)
                    if cache_list:
                        for tensor in cache_list:
                            del tensor
                        kv_cache_cpu[key] = None
                del offloaded_info["kv_cache_cpu"]

            # 清理SSD存储文件
            storage_path = offloaded_info.get("storage_path")
            if storage_path and os.path.exists(storage_path):
                try:
                    os.remove(storage_path)
                    offload_logger.info(f"[DEBUG: offload] Deleted storage file: {storage_path}")
                except Exception as e:
                    offload_logger.warning(f"[DEBUG: offload] Failed to delete storage file: {e}")

            self._offloaded_requests.pop(request_id)
            offload_logger.info(f"[DEBUG: offload] Cleaned up offloaded request: {request_id}")

    def get_offloaded_request_count(self) -> int:
        """获取当前offloaded的请求数量"""
        with self._lock:
            return len(self._offloaded_requests)

    def get_offloaded_request_ids(self) -> List[str]:
        """获取当前所有offloaded的请求ID列表"""
        with self._lock:
            return list(self._offloaded_requests.keys())

    def prefetch_ssd_to_cpu(self) -> int:
        """
        后台预取：将 SSD 上的 KV Cache 预取到 CPU 内存

        当 CPU 内存有空闲时调用，减少 resume 时的延迟

        Returns:
            int: 成功预取的请求数量
        """
        if not self.enable_offload or self.cache_manager is None:
            return 0

        prefetched_count = 0

        with self._lock:
            # 找出存储在 SSD 上且还没有 CPU copy 的请求
            ssd_requests = [
                (req_id, info)
                for req_id, info in self._offloaded_requests.items()
                if info.get("storage_level") == self.STORAGE_LEVEL_SSD and info.get("kv_cache_cpu") is None
            ]

        for req_id, info in ssd_requests:
            storage_path = info.get("storage_path")
            if not storage_path or not os.path.exists(storage_path):
                continue

            # 检查 CPU 内存是否充足
            num_blocks = info.get("num_blocks_needed", 0)
            if num_blocks > len(self.cache_manager.cpu_free_block_list):
                break  # CPU 内存不足，停止预取

            try:
                # 从 SSD 加载到 CPU
                kv_cache_cpu = self.load_from_storage(storage_path)
                if kv_cache_cpu is None:
                    continue

                # 更新 offloaded 信息
                with self._lock:
                    if req_id in self._offloaded_requests:
                        self._offloaded_requests[req_id]["kv_cache_cpu"] = kv_cache_cpu
                        self._offloaded_requests[req_id]["storage_level"] = self.STORAGE_LEVEL_CPU

                prefetched_count += 1
                offload_logger.info(
                    f"[DEBUG: offload_prefetch_ssd_to_cpu] Prefetched request {req_id} from SSD to CPU"
                )

            except Exception as e:
                offload_logger.warning(f"[DEBUG: offload_prefetch_ssd_to_cpu] Failed to prefetch {req_id}: {e}")

        return prefetched_count
