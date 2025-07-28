# Copyright (c) 2023  PaddlePaddle Authors. All Rights Reserved.
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
This module is used to store environmental variables in PaddleMIX.
PPMIX_HOME              -->  the root directory for storing PaddleMIX related data. Default to ~/.paddlemix. Users can change the
├                            default value through the PPMIX_HOME environment variable.
├─ MODEL_HOME              -->  Store model files.
└─ DATA_HOME         -->  Store automatically downloaded datasets.
"""
import os
import random

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import get_rng_state_tracker


def _get_user_home():
    return os.path.expanduser("~")


def _get_ppmix_home():
    if "PPMIX_HOME" in os.environ:
        home_path = os.environ["PPMIX_HOME"]
        if os.path.exists(home_path):
            if os.path.isdir(home_path):
                return home_path
            else:
                raise RuntimeError("The environment variable PPMIX_HOME {} is not a directory.".format(home_path))
        else:
            return home_path
    return os.path.join(_get_user_home(), ".paddlemix")


def _get_sub_home(directory, parent_home=_get_ppmix_home()):
    home = os.path.join(parent_home, directory)
    if not os.path.exists(home):
        os.makedirs(home, exist_ok=True)
    return home


def _get_bool_env(env_key: str, default_value: str) -> bool:
    """get boolean environment variable, which can be "true", "True", "1"

    Args:
        env_key (str): key of env variable
    """
    value = os.getenv(env_key, default_value).lower()
    return value in ["true", "1"]


USER_HOME = _get_user_home()
PPMIX_HOME = _get_ppmix_home()
MODEL_HOME = _get_sub_home("models")
HF_CACHE_HOME = os.environ.get("HUGGINGFACE_HUB_CACHE", MODEL_HOME)
DATA_HOME = _get_sub_home("datasets")
PACKAGE_HOME = _get_sub_home("packages")
DOWNLOAD_SERVER = "http://paddlepaddle.org.cn/paddlehub"
FAILED_STATUS = -1
SUCCESS_STATUS = 0

LEGACY_CONFIG_NAME = "model_config.json"
CONFIG_NAME = "config.json"
TOKENIZER_CONFIG_NAME = "tokenizer_config.json"
PYTORCH_WEIGHT_FILE_NAME = "pytorch_model.bin"
PADDLE_WEIGHT_FILE_NAME = "model_state.pdparams"
LORA_CONFIG_NAME = "lora_config.json"
PREFIX_CONFIG_NAME = "prefix_config.json"
LORA_WEIGHT_FILE_NAME = "lora_model_state.pdparams"
PREFIX_WEIGHT_FILE_NAME = "prefix_model_state.pdparams"
PAST_KEY_VALUES_FILE_NAME = "pre_caches.npy"

# for conversion
ENABLE_TORCH_CHECKPOINT = _get_bool_env("ENABLE_TORCH_CHECKPOINT", "true")


def set_hybrid_parallel_seed(basic_seed, data_world_rank, mp_rank, pp_rank=0):
    device_id = paddle.device.get_device()
    assert "gpu" in device_id

    random.seed(basic_seed + data_world_rank)
    np.random.seed(basic_seed + data_world_rank)
    paddle.seed(basic_seed + data_world_rank)

    # local_seed/ global_seed is used to control dropout in ModelParallel
    local_seed = 1024 + basic_seed + mp_rank * 100 + data_world_rank
    global_seed = 2048 + basic_seed + data_world_rank
    tracker = get_rng_state_tracker()
    tracker.add("global_seed", global_seed)
    tracker.add("local_seed", local_seed)


def setdistenv(args):
    if dist.get_world_size() > 1:
        args.dp_degree = dist.get_world_size() // (
            args.tensor_parallel_degree * args.sharding_parallel_degree * args.pipeline_parallel_degree
        )
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": args.dp_degree,
            "mp_degree": args.tensor_parallel_degree,
            "sharding_degree": args.sharding_parallel_degree,
            "pp_degree": args.pipeline_parallel_degree,
        }
        # strategy.find_unused_parameters = True

        # set control in tensor parallel
        strategy.tensor_parallel_configs = {"tensor_init_seed": args.seed}

        fleet.init(is_collective=True, strategy=strategy)

        args.rank = dist.get_rank()
        # obtain rank message of hybrid parallel
        hcg = fleet.get_hybrid_communicate_group()
        args.mp_rank = hcg.get_model_parallel_rank()
        args.dp_rank = hcg.get_data_parallel_rank()
        args.sharding_rank = hcg.get_sharding_parallel_rank()

        args.data_world_rank = args.dp_rank * args.sharding_parallel_degree + args.sharding_rank
        args.data_world_size = dist.get_world_size() // abs(
            args.tensor_parallel_degree * args.pipeline_parallel_degree
        )
    else:
        args.data_world_rank = 0
        args.data_world_size = 1
        args.mp_rank = 0
        args.rank = 0

    # seed control in hybrid parallel
    set_hybrid_parallel_seed(args.seed, args.data_world_rank, args.mp_rank)
