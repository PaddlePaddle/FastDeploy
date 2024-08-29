# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import json
import os
import sys
from datetime import datetime
from paddlenlp.generation import GenerationConfig

from server.utils import model_server_logger


class Config:
    """
    初始化配置，各参数优先以环境变量配置的值为准
    """

    def __init__(self):
        self.read_from_env()

    def read_from_env(self):
        """
        从环境变量中读取参数
        """
        env = os.environ
        self.model_dir = env.get(
            "MODEL_DIR", "/opt/output/Serving/models")
        if not self.model_dir:
            raise Exception("The parameter MODEL_DIR is None.")
        self.mp_num = int(env.get("MP_NUM", 8))
        self.config_json_file = env.get("CONFIG_JSON_FILE", "config.json")
        self.model_config_path = os.path.join(self.model_dir, self.config_json_file)
        if env.get("FD_MODEL_CONFIG_PATH", None):
            self.model_config_path = env.get("FD_MODEL_CONFIG_PATH")

        # 分布式配置文件
        self.distributed_config_path = os.path.join(self.model_dir, "rank_mapping.csv")
        if os.getenv("DISTRIBUTED_CONFIG", None):
            self.distributed_config_path = os.getenv("DISTRIBUTED_CONFIG")

        # 硬件配置信息
        self.device = env.get("DEVICE", "GPU")
        self.device_ids = ",".join([str(i) for i in range(self.mp_num)])
        if self.device == "GPU":
            self.device_ids = os.getenv("CUDA_VISIBLE_DEVICES",
                                        self.device_ids)
        else:
            raise Exception(f"unsupported device type: {self.device}")

        # Triton服务层参数
        self.max_prefill_batch = int(os.getenv("MAX_PREFILL_BATCH", 1))
        if self.max_prefill_batch <= 0:
            raise Exception(f"MAX_PREFILL_BATCH ({self.max_prefill_batch}) must be greater than 0")
        self.disable_streaming = int(os.getenv("DISABLE_STREAMING", 0))

        # 最大支持缓存的task数
        self.max_cached_task_num = int(os.getenv("MAX_CACHED_TASK_NUM", "128"))
        # 如果没有配置PUSH_MODE_HTTP_PORT, 则只支持 GRPC 服务模式
        self.push_mode_http_port = int(os.getenv("PUSH_MODE_HTTP_PORT", "-1"))
        if self.push_mode_http_port > 0:
            grpc_port = os.getenv("GRPC_PORT", None)
            if grpc_port is None:
                raise Exception("GRPC_PORT cannot be None, while PUSH_MODE_HTTP_PORT>0")
            self.grpc_port = int(grpc_port)

        # http服务线的worker数
        self.push_mode_http_workers = int(os.getenv("PUSH_MODE_HTTP_WORKERS", "1"))
        if self.push_mode_http_workers < 1:
            raise Exception(f"PUSH_MODE_HTTP_WORKERS ({self.push_mode_http_workers}) must be positive")

        # 导出Paddle代码版本，便于对比版本号
        import paddle
        self.paddle_commit_id = paddle.version.commit

        # 探活时检测engine主循环是否正常的时间间隔
        self.check_health_interval = int(os.getenv("CHECK_HEALTH_INTERVAL", 10))

        # 与模型相关信息（注意要与导出的模型保持一致，否则存在效果问题）
        self.dtype = env.get("DTYPE", "bfloat16")
        self.block_size = int(env.get("BLOCK_SIZE", 64))
        self.use_cache_kv_int8 = int(os.getenv("USE_CACHE_KV_INT8", 0))
        self.use_cache_kv_int4 = int(os.getenv("USE_CACHE_KV_INT4", 0))

        # 推理引擎配置
        self.max_batch_size = int(env.get("BATCH_SIZE", 50))
        self.max_seq_len = int(env.get("MAX_SEQ_LEN", 8192))
        self.max_dec_len = int(env.get("MAX_DEC_LEN", 1024))
        self.enc_dec_block_num = int(os.getenv("ENC_DEC_BLOCK_NUM", 2))
        self.block_bs = float(env.get("BLOCK_BS", 50))
        self.block_ratio = float(os.getenv("BLOCK_RATIO", 0.75))
        self.bad_tokens = str(env.get("BAD_TOKENS", "-1"))
        self.first_token_id = int(os.getenv("FIRST_TOKEN_ID", 1))

        # 引擎输入队列端口号
        self.infer_port = int(os.getenv("INFER_QUEUE_PORT", 56666))

        # 是否开启探活服务
        self.use_custom_health_checker = int(os.getenv("USE_CUSTOM_HEALTH_CHECKER", 1))

        # 环境变量配置MAX_SEQ_LEN，MAX_DEC_LEN将用于控制服务请求合法性检查
        self.seq_len_limit = int(env.get("MAX_SEQ_LEN", 7168))
        self.dec_len_limit = int(env.get("MAX_DEC_LEN", 1024))

        # warmup
        self.use_warmup = int(os.getenv("USE_WARMUP", 0)) == 1

        # uuid
        self.shm_uuid = os.getenv("SHM_UUID", '')

        # 加载 Generation 文件
        try:
            self.generation_config = GenerationConfig.from_pretrained(self.model_dir)
        except:
            model_server_logger.warning(
                "Can't find generation config, so it will not use generation_config field in the model config"
            )
            self.generation_config = None

        self.read_from_config()
        self.postprocess()
        self.check()

    def postprocess(self):
        """
        根据配置参数，计算部分额外的参数
        """
        if self.block_ratio >= 1.0:
            self.enc_dec_block_num = (self.max_dec_len + self.block_size - 1) // self.block_size
        self.max_query_block_num = (max(self.max_dec_len, self.max_seq_len) +
                               self.block_size - 1) // self.block_size
        self.max_query_block_num = (self.max_dec_len + self.max_seq_len +
                                self.block_size - 1) // self.block_size
        self.dec_token_num = self.enc_dec_block_num * self.block_size
        self.total_block_num = int(self.block_bs * self.max_query_block_num)
        self.max_block_num = int(self.total_block_num * self.block_ratio)
        model_server_logger.info(f"max_block_num:{self.max_block_num}")

    def check(self):
        """
        检查参数配置合法性
        """
        assert self.max_batch_size <= 256, (
            "The parameter `max_batch_size` is not allowed to exceed 256, "
            "but now it's {}.".format(self.max_batch_size)
        )
        assert self.seq_len_limit <= self.max_seq_len, (
            f"The seq_len_limit shouldn't greater than max_seq_len in model, "
            f"which means the exported MAX_SEQ_LEN should less than "
            f"{self.max_seq_len}, but now it's {self.seq_len_limit}."
        )
        assert self.dec_len_limit <= self.max_seq_len, (
            f"The dec_len_limit shouldn't greater than max_seq_len in model, "
            f"which means the exported MAX_DEC_LEN should less than "
            f"{self.max_seq_len}, but now it's {self.dec_len_limit}."
        )

    def print(self, file=None):
        """
        输出所有参数配置

        file: 如若指定file路径，同时将日志以追加方式写入到另外的文件中
              解决当前日志系统仅保留7天，无法追查启动信息问题
        """
        model_server_logger.info(
            "=================== Configuration Information ===============")
        for k, v in self.__dict__.items():
            if k == "generation_config" and v is not None:
                for gck, gcv in v.to_dict().items():
                    model_server_logger.info("{:<20}:{:<6}{}".format(gck, "", gcv))
            else:
                model_server_logger.info("{:<20}:{:<6}{}".format(k, "", v))
        model_server_logger.info(
            "=============================================================")
        if file is not None:
            f = open(file, "a")
            now_time = datetime.now()
            f.write(f"{now_time} configuration information as below,\n")
            for k, v in self.__dict__.items():
                f.write("{:<20}:{:<6}{}\n".format(k, "", v))
            f.close()

    def get_model_config(self):
        """
        读取模型配置文件
        """
        model_config_json = json.load(open(self.model_config_path, 'r', encoding='utf-8'))
        return model_config_json

    def read_from_config(self):
        """
        从配置文件中读取参数
        """
        from server.utils import get_logger
        logger = get_logger("model_server", "infer_config.log")
        config = self.get_model_config()

        def reset_value(self, value_name, key, config):
            if key in config:
                value = config[key]
                setattr(self, value_name, value)
                logger.info(f"Reset parameter {value_name} = {value} from configuration.")

        reset_value(self, "block_size", "infer_model_block_size", config)
        reset_value(self, "max_seq_len", "infer_model_max_seq_len", config)

        assert self.seq_len_limit <= self.max_seq_len, f"The loading model requires len(input_ids) <= {self.max_seq_len}, but now the setting MAX_SEQ_LEN={self.seq_len_limit}."
        assert self.dec_len_limit <= self.max_seq_len, f"The loading model requires MAX_DEC_LEN <= {self.max_seq_len}, but now the setting MAX_DEC_LEN={self.dec_len_limit}."

    def get_unique_name(self, name):
        return name + f"_{self.shm_uuid}"

    def __str__(self) -> str:
        return json.dumps(self.__dict__, indent=4)
