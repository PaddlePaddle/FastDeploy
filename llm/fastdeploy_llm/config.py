# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import os
import json
import logging
import fastdeploy_llm
from fastdeploy_llm.utils.utils import check_model
from fastdeploy_llm.utils.logging_util import logger, Logger


class Config:
    def __init__(self, model_dir, decode_strategy="sampling", mp_num=None):
        self.model_dir = model_dir
        is_static, rank = check_model(model_dir)

        if os.getenv("ENABLE_DEBUG_LOG", "0") == "1":
            logger.info(
                "Detect enviroment variable `ENABLE_DEBUG_LOG`, all the debug log information will output to fastdeploy_llm_serving.log."
            )
            fastdeploy_llm.utils.logging_util.logger = Logger(
                log_file="fastdeploy_llm_serving.log",
                time_rotation=7,
                level=logging.DEBUG)
        else:
            logger.info(
                "The logging level is set as INFO, if more information needed, please execute `export ENABLE_DEBUG_LOG=1` before launching service."
            )
            fastdeploy_llm.utils.logging_util.logger = Logger(
                log_file="fastdeploy_llm_serving.log",
                time_rotation=7,
                level=logging.INFO)

        assert decode_strategy in [
            "sampling", "greedy_search"
        ], "The decode strategy only supports sampling/greedy_search, now it's {}.".format(
            decode_strategy)
        self.mp_num = mp_num
        if is_static:
            logger.warning(
                "The model {} is detected as static inference model, the decode_strategy is not configurable.".
                format(model_dir))
            if mp_num is None:
                self.mp_num = rank
            elif self.mp_num != rank:
                raise Exception(
                    "The model {} is a static inference model with rank={}, you need to set mp_num equal to the rank.".
                    format(model_dir, rank))
        else:
            logger.info(
                "The model {} is detected as a dygraph model, decode with {}.".
                format(model_dir, decode_strategy))
            if mp_num is None:
                logger.warning(
                    "The model {} is a dygraph model, since the mp_num is None, we will set it to 1 automatically.".
                    format(model_dir))
                self.mp_num = 1

        self.is_static_model = is_static
        self.max_batch_size = 1
        self.max_dec_len = 1024
        self.max_seq_len = 1024
        self.decode_strategy = decode_strategy

        self.stop_threshold = 2
        self.disable_dynamic_batching = False
        self.max_queue_num = 512

        if not os.path.exists(os.path.join(model_dir, "config.json")):
            raise Exception("Cannot find file {}.".format(
                os.path.join(model_dir, "config.json")))
        with open(os.path.join(model_dir, "config.json")) as f:
            config = json.loads(f.read())
        if "num_hidden_layers" in config:
            self.num_layers = config["num_hidden_layers"]
        elif "n_layer" in config:
            self.num_layers = config["n_layer"]
        else:
            raise Exception("Cannot find num layers in {}.".format(
                os.path.join(model_dir, "config.json")))

        if "num_attention_heads" in config:
            self.num_attention_heads = config["num_attention_heads"]
        elif "n_head" in config:
            self.num_attention_heads = config["n_head"]
        else:
            raise Exception("Cannot find num_attention_heads in {}.".format(
                os.path.join(model_dir, "config.json")))

        self.hidden_size = int(config["hidden_size"])

        self.eos_token_id = set([int(config["eos_token_id"])])
        self.architecture = config["architectures"][0].lower()

        self.is_ptuning = config.get("is_ptuning", 0)
        self.model_prompt_dir_path = config.get("prompt_dir_path",
                                                "./prompt_embedding")
        self.max_prefix_len = config.get("max_prefix_len", 128)

    def is_arch(self, arch):
        return arch in self.architecture

    def load_environment_variables(self):
        # If environment variables are set, override the value in configuration file
        if os.getenv("BATCH_SIZE", None):
            self.max_batch_size = int(os.getenv("BATCH_SIZE"))
            logger.warning(
                "Detect environment BATCH_SIZE={}, will reset `max_batch_size` to {}!".
                format(self.max_batch_size, self.max_batch_size))
        if os.getenv("MAX_SEQ_LEN", None):
            self.max_seq_len = int(os.getenv("MAX_SEQ_LEN"))
            logger.warning(
                "Detect environment MAX_SEQ_LEN={}, will reset `max_seq_len` to {}!".
                format(self.max_seq_len, self.max_seq_len))
        if os.getenv("MAX_DEC_LEN", None):
            self.max_dec_len = int(os.getenv("MAX_DEC_LEN"))
            logger.warning(
                "Detect environment MAX_DEC_LEN={}, will reset `max_dec_len` to {}!".
                format(self.max_dec_len, self.max_dec_len))
        if os.getenv("PROMPT_NUM", None):
            self.max_prefix_len = int(os.getenv("PROMPT_NUM"))
            logger.warning(
                "Detect environment PROMPT_NUM={}, will reset `max_prefix_len` to {}!".
                format(self.max_prefix_len, self.max_prefix_len))
        if os.getenv("IS_PTUNING", None):
            self.is_ptuning = int(os.getenv("IS_PTUNING"))
            logger.warning(
                "Detect environment IS_PTUNING={}, will reset `is_ptuning` to {}!".
                format(self.is_ptuning, self.is_ptuning))
        if os.getenv("PROMPT_DIR_PATH", None):
            self.model_prompt_dir_path = os.getenv("PROMPT_DIR_PATH")
            logger.warning(
                "Detect environment PROMPT_DIR_PATH={}, will reset `model_prompt_dir_path` to {}!".
                format(self.model_prompt_dir_path, self.model_prompt_dir_path))
        if os.getenv("STOP_THRESHOLD", None):
            self.stop_threshold = int(os.getenv("STOP_THRESHOLD"))
            logger.warning(
                "Detect environment STOP_THRESHOLD={}, will reset `stop_threshold` to {}!".
                format(self.stop_threshold, self.stop_threshold))
        if os.getenv("MP_NUM", None):
            self.mp_num = int(os.getenv("MP_NUM"))
            logger.warning(
                "detect environment mp_num={}, will reset `mp_num` to {}!".
                format(self.mp_num, self.mp_num))
        if os.getenv("MAX_QUEUE_NUM", None):
            self.max_queue_num = int(os.getenv("MAX_QUEUE_NUM"))
            logger.warning(
                "detect environment MAX_QUEUE_NUM={}, will reset `max_queue_num` to {}!".
                format(self.max_queue_num, self.max_queue_num))
        if os.getenv("DISABLE_DYNAMIC_BATCHING", None):
            self.disable_dynamic_batching = bool(
                os.getenv("DISABLE_DYNAMIC_BATCHING"))
            logger.warning(
                "detect environment DISABLE_DYNAMIC_BATCHING={}, will reset `disable_dynamic_batching` to {}!".
                format(self.disable_dynamic_batching,
                       self.disable_dynamic_batching))
