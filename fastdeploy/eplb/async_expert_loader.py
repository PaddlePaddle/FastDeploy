"""
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
"""

import ctypes
import os
import time
import traceback
from typing import List, Tuple

import numpy as np
import paddle

try:
    from cuda import cudart
except ImportError:
    cudart = None

from fastdeploy.config import EPLBConfig

REARRANGE_EXPERT_MAGIC_NUM = 147183647
REARRANGE_ORIGINATOR_EP_RANK = 0
CHECK_TIME_INTERNAL = 3
HTTP_RETRY_NUM = 5
CHECK_TIMEOUT = 120

libc = ctypes.CDLL(None)

libc.mmap.argtypes = [
    ctypes.c_void_p,  # void *addr
    ctypes.c_size_t,  # size_t length
    ctypes.c_int,  # int prot
    ctypes.c_int,  # int flags
    ctypes.c_int,  # int fd
    ctypes.c_size_t,  # off_t offset
]
libc.mmap.restype = ctypes.c_void_p
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
libc.munmap.restype = ctypes.c_int

PROT_READ = 0x1
PROT_WRITE = 0x2
MAP_SHARED = 0x01
MAP_ANONYMOUS = 0x20
MAP_FAILED = -1

G = 1024**3
TOTAL_MODEL_SIZE = 350
MAIN_MODEL_REDUNDANT_SHM_SIZE = 5

MODEL_MAIN_NAME = "eplb_main"


def create_mmap(model_name: List, ep_rank: int, ep_size: int, shm_uuid: str, eplb_config: EPLBConfig, logger=None):
    """create_mmap"""
    flags = MAP_SHARED
    prot = PROT_READ | PROT_WRITE

    main_size = 0
    if eplb_config.redundant_expert_async_load_model_shmem_size_gb == 0:
        main_size = TOTAL_MODEL_SIZE // ep_size
    else:
        main_size = eplb_config.redundant_expert_async_load_model_shmem_size_gb
    main_size = main_size * G

    mmap_infos = {}
    for name in model_name:
        expert_weight_file = f"/dev/shm/{name}_rank_{ep_rank}_expert_weight_{shm_uuid}"
        shm_size = main_size

        if not os.path.isfile(expert_weight_file):
            open(expert_weight_file, "wb").close()
        shm_fd = os.open(expert_weight_file, os.O_RDWR)
        os.ftruncate(shm_fd, shm_size)
        if logger is not None:
            logger.info(f"redundant_expert: create_mmap file {expert_weight_file}, fd {shm_fd}, size {shm_size}")

        shm_ptr = libc.mmap(0, ctypes.c_size_t(shm_size), prot, flags, shm_fd, 0)
        if shm_ptr == MAP_FAILED:
            raise OSError(f"redundant_expert: mmap {expert_weight_file} failed: {ctypes.get_errno()}")

        shm_ptr = ctypes.cast(shm_ptr, ctypes.POINTER(ctypes.c_int8))
        addr = ctypes.addressof(shm_ptr.contents)

        if cudart is None:
            raise ImportError(
                "cuda-python not installed. Install the version matching your CUDA toolkit:\n"
                "  CUDA 12.x → pip install cuda-python==12.*\n"
            )

        # Register memory with CUDA
        (ret,) = cudart.cudaHostRegister(addr, shm_size, 0)
        if ret != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(
                f"cudaHostRegister failed: {cudart.cudaGetErrorString(ret)}, "
                f" address {hex(addr)} size {shm_size}, ret: {ret}"
            )

        mmap_infos[name] = shm_ptr

    return mmap_infos


def save_tensor_to_shm_mem(cached_weights, file_path, logger=None):
    """save_tensor_to_shm_mem"""
    tensor_infos = []
    offset = 0
    if not os.path.exists(file_path):
        raise OSError("File is not exist")

    shm_size = os.path.getsize(file_path)

    for name, w in cached_weights:
        size = w.numel().item() * w.element_size()
        # logger.info(f"redundant_expert: save tensor to {name} offset: {offset} size: {size}")
        w_ptr = ctypes.string_at(w.data_ptr(), size)
        with open(file_path, "r+b") as file:
            file.seek(offset)
            if offset + size > shm_size:
                raise IOError(
                    f"redundant_expert: Exceeded {file_path} file's size. "
                    + "Should set a bigger value using env variable."
                )
            n = file.write(w_ptr)
            assert n == size
        tensor_infos.append((name, offset, size, w.shape, w.dtype))

        offset += size

    sz = offset / 1024 / 1024 / 1024
    if logger is not None:
        logger.info(f"redundant_expert: save_tensor_to_shm_mem success. file {file_path} size {sz}G")

    return tensor_infos


def load_tensor_from_shm_mem(tensor_infos, shm_ptr, logger=None):
    """load_tensor_from_shm_mem"""
    # weights_dict = {}
    weights_dict = []
    for name, offset, size, shape, dtype in tensor_infos:
        # 计算共享内存中张量的地址
        w_addr = ctypes.cast(shm_ptr, ctypes.c_void_p).value + offset
        w_ptr = ctypes.cast(w_addr, ctypes.POINTER(ctypes.c_byte))
        # 先读取为字节数组，再通过视图转换成适当类型
        np_array = np.ctypeslib.as_array(w_ptr, shape=(size,))

        if dtype == paddle.float32:
            tmp = np_array.view(np.float32)
            tensor = paddle.Tensor(tmp, dtype=paddle.float32, place=paddle.CPUPlace(), zero_copy=True)
        elif dtype == paddle.uint8:
            tmp = np_array.view(np.uint8)
            tensor = paddle.Tensor(tmp, dtype=paddle.uint8, place=paddle.CPUPlace(), zero_copy=True)
        elif dtype == paddle.int8:
            tmp = np_array.view(np.int8)
            tensor = paddle.Tensor(tmp, dtype=paddle.int8, place=paddle.CPUPlace(), zero_copy=True)
        elif dtype == paddle.bfloat16:
            # NumPy 不支持 bfloat16，因此先以 uint16 读取原始数据，再用 Paddle cast 为 bfloat16
            tmp = np_array.view(np.uint16)
            tensor = paddle.Tensor(tmp, dtype=paddle.bfloat16, place=paddle.CPUPlace(), zero_copy=True)
        elif dtype == paddle.float8_e4m3fn:
            tmp = np_array.view(np.uint8)
            tensor = paddle.Tensor(tmp, dtype=paddle.float8_e4m3fn, place=paddle.CPUPlace(), zero_copy=True)
        else:
            raise TypeError(f"Unsupported dtype: {dtype}")

        assert w_addr == tensor.data_ptr()
        # weights_dict[name] = tensor.view(shape)
        weights_dict.append((name, tensor.view(shape)))

    if logger is not None:
        logger.info("redundant_expert: load_tensor_from_shm_mem succ")
    return weights_dict


class AsyncEPLoader(object):
    """Aynsc Expert loader"""

    def __init__(
        self,
        model_dir,
        eplb_config,
        rank=8,
        expert_per_rank=8,
        moe_layer_start_index=3,
        moe_quant_type="",
        logger=None,
    ):
        """
        __init__
        """
        self.model_path = model_dir
        self.eplb_config = eplb_config

        self.expert_per_rank = expert_per_rank
        self.moe_layer_start_index = moe_layer_start_index
        self.ep_rank = rank
        self.moe_quant_type = moe_quant_type

        self.old_model_ep_rank_to_expert_id_list = None
        self.new_model_ep_rank_to_expert_id_list = None

        self.cached_weights = []
        # self.state_dicts = {}
        self.moe_file_names = []

        self.logger = logger

    def reset(self):
        """
        reset
        """
        self.old_model_ep_rank_to_expert_id_list = None
        self.new_model_ep_rank_to_expert_id_list = None
        self.cached_weights = []
        self.moe_file_names = []

    def load_experts_weight_from_disk(self):
        """
        return value: (all_succ whether_load_weight exist_fatal_error message),
        exist_fatal_error means all rank need restart
        """
        ep_rank = self.ep_rank
        start_idx = ep_rank * self.expert_per_rank
        end_idx = start_idx + self.expert_per_rank
        try:
            old_expert_ids_all = self.old_model_ep_rank_to_expert_id_list[:, start_idx:end_idx]
            new_expert_ids_all = self.new_model_ep_rank_to_expert_id_list[:, start_idx:end_idx]
            need_to_reload = list()
            for layer_id in range(len(old_expert_ids_all)):
                if layer_id < self.moe_layer_start_index:
                    continue
                new_expert_ids = new_expert_ids_all[layer_id]
                old_expert_ids = old_expert_ids_all[layer_id]
                if len(new_expert_ids) != len(old_expert_ids):
                    message = f"redundant_expert: new_expert_ids length not equal to old_expert_ids \
                        length layer_id: {layer_id}"
                    # this is very dangerous and unepxpected, should be fixed
                    return False, message
                # TODO: 按需加载，过滤重复专家
                self.logger.info(
                    f"redundant_expert: rank {ep_rank} layer {layer_id} old_experts {old_expert_ids}"
                    + f" new_experts {new_expert_ids}"
                )
                need_to_reload.extend([(layer_id, expert_id) for expert_id in new_expert_ids])

            succ = True
            message = ""
            if len(need_to_reload) > 0:
                if self.eplb_config.model_use_safetensors:
                    succ, message = self.load_safetensor_fp8_from_disk(need_to_reload)
                else:
                    succ, message = self.load_weight_bf16_from_disk(need_to_reload)
            if not succ:
                self.logger.info(
                    f"redundant_expert: load_experts_weight_from_disk fail. rank {ep_rank}, error: {message}"
                )
                new_message = f"redundant_expert: load_experts_weight_from_disk fail. rank {ep_rank}, error: {message}"
                return False, new_message
            self.logger.info(f"redundant_expert: load_experts_weight_from_disk success. rank {ep_rank}")
            return True, "redundant_expert: load_experts_weight_from_disk success"
        except Exception as e:
            message = f"redundant_expert: Failed to load_experts_weight_from_disk ep_rank {ep_rank} excep: {e}"
            error_message = traceback.format_exc()
            self.logger.error(f"redundant_expert: message: {message} traceback: {error_message}")
            return False, message

    def load_weight_bf16_from_disk(self, need_to_reload: List[Tuple[int, int]]):
        """load_weight_bf16_from_disk"""
        try:
            ckpt_up_gate_proj_name = "up_gate_proj"
            ckpt_down_proj_name = "down_proj"
            for layer_id, expert_id in need_to_reload:
                for weight_name in [ckpt_up_gate_proj_name, ckpt_down_proj_name]:
                    ckpt_file_name = f"ernie.layers.{layer_id}.mlp.experts.{expert_id}.{weight_name}.weight"
                    if ckpt_file_name not in self.moe_file_names:
                        self.logger.info(f"record redundant_expert: {ckpt_file_name}")
                        self.moe_file_names.append(ckpt_file_name)

            last_device = paddle.device.get_device()
            paddle.set_device("cpu")

            for file_name in self.moe_file_names:
                # 判断文件是否存在
                if not os.path.exists(self.model_path + "/merged_tp1_state_split/" + file_name):
                    # self.logger.info(f"redundant_expert: {file_name} not exist.")
                    continue
                # self.logger.info(f"redundant_expert: Loading expert weights: {file_name}.")
                # self.state_dicts[file_name] = paddle.load(self.model_path + "/merged_tp1_state_split/" + file_name)

            paddle.set_device(last_device)
            self.logger.info("redundant_expert: Loading expert weights end.")
            return True, "redundant_expert: Succeeded to loading expert weights."
        except Exception as e:
            message = f"redundant_expert: Failed to get weights iterator: {e}."
            return False, message

    def load_safetensor_fp8_from_disk(self, need_to_reload: List[Tuple[int, int]]):
        """load_safetensor_fp8_from_disk"""
        """
        ernie.layers.52.mlp.experts.58.up_gate_proj.quant_weight
        ernie.layers.52.mlp.experts.58.up_gate_proj.weight_scale
        ernie.layers.52.mlp.experts.58.down_proj.quant_weight
        ernie.layers.52.mlp.experts.58.down_proj.weight_scale
        """
        up_gate_down = ["up_gate_proj", "down_proj"]
        quant_weight_scale = ["quant_weight", "weight_scale"]
        ckpt_name = [
            (f"ernie.layers.{layer_id}.mlp.experts.{expert_id}.{proj_name}.{quant_name}")
            for layer_id, expert_id in need_to_reload
            for proj_name in up_gate_down
            for quant_name in quant_weight_scale
        ]
        ckpt_name_to_safetensor_file = load_ep_checkpoint(self.model_path)
        hf_weights_files = list(set(ckpt_name_to_safetensor_file.values()))
        state_dicts = {}

        last_device = paddle.device.get_device()
        paddle.set_device("cpu")

        from safetensors import safe_open

        for st_file in hf_weights_files:
            with safe_open(st_file, framework="paddle", device="cpu") as f:
                for name in f.keys():
                    if name in ckpt_name:
                        weight = f.get_tensor(name)
                        state_dicts[name] = paddle.Tensor(weight, zero_copy=True)
        weights_list = []
        for name in ckpt_name:
            weights_list.append((name, state_dicts[name]))
        self.cached_weights = weights_list

        paddle.set_device(last_device)
        return True, "load_expert_weight_from_disk_safetensor success"


def load_ep_checkpoint(model_path):
    """
    load ep checkpoint
    """
    file_path = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.exists(file_path):
        return {}
    import json

    with open(file_path, "r") as f:
        weight_map = json.load(f)["weight_map"]
        state_dict = {k: os.path.join(model_path, v) for k, v in weight_map.items()}
    return state_dict


def load_model_weights_process(
    rank: int,
    model_dir: str,
    expert_per_rank: int,
    moe_layer_start_index: int,
    moe_quant_type: str,
    shm_uuid: str,
    eplb_config: EPLBConfig,
    data_conn,
    mg_conn,
):
    """
    load_model_weights_process
    """
    import faulthandler

    from setproctitle import setproctitle

    setproctitle(f"eplb::async_load_model_{rank}")
    faulthandler.enable()
    from fastdeploy.utils import get_logger

    logger = get_logger("eplb_async_loader", "eplb_{0}.log".format(rank))
    logger.info("redundant_expert: load_model_weights_process start")

    paddle.set_device("cpu")
    ep_loader = AsyncEPLoader(
        model_dir=model_dir,
        rank=rank,
        expert_per_rank=expert_per_rank,
        moe_layer_start_index=moe_layer_start_index,
        moe_quant_type=moe_quant_type,
        logger=logger,
        eplb_config=eplb_config,
    )

    while True:
        ep_loader.reset()
        data = mg_conn.recv()

        result = True
        weight_infos = []
        try:
            ep_loader.old_model_ep_rank_to_expert_id_list = data["old_model_ep_rank_to_expert_id_list"]
            ep_loader.new_model_ep_rank_to_expert_id_list = data["new_model_ep_rank_to_expert_id_list"]

            begin_time_disk = int(time.time())
            success, message = ep_loader.load_experts_weight_from_disk()
            begin_time_shm = int(time.time())
            logger.info(
                "redundant_expert: async load load_weight_from_disk, "
                + f"succ {success}, cost {begin_time_shm-begin_time_disk}s"
            )
            if success:
                model_name = MODEL_MAIN_NAME
                file_path = f"/dev/shm/{model_name}_rank_{rank}_expert_weight_{shm_uuid}"
                weight_infos = save_tensor_to_shm_mem(ep_loader.cached_weights, file_path, logger)
                logger.info(
                    "redundant_expert: async load save_tensor_to_shm_mem, "
                    + f"tensor nums {len(weight_infos)}, cost {int(time.time()-begin_time_shm)}s"
                )
            else:
                logger.error(f"redundant_expert: async load load_weight_from_disk failed, error {message}")
                result = False

        except Exception as e:
            logger.error(f"redundant_expert: async load weights failed, rank {rank} error {e}")
            result = False
            weight_infos = []
        finally:
            request_data = {"result": result, "weights": weight_infos}
            data_conn.send(request_data)
