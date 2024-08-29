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


def check_basic_params(req_dict):
    """
    对单个输入请求进行基础的校验检查，适用于推拉模式。
    对输入的全部字段进行检查，统一将报错信息发送给用户，注意同一个字段的检查逻辑是独立的，避免重复的报错信息。

    Args:
        req_dict (dict): 请求的字典格式数据，包含文本、模型、序列长度、最大token数等字段。

    Returns:
        list[str]: 如果校验有错误，返回错误信息列表，如果校验正确，返回空列表。
    """

    error_msg = []

    # text、input_ids和messages必须设置一个
    bools = ("text" in req_dict, "input_ids" in req_dict, "messages" in req_dict)
    if sum(bools) == 0:
        error_msg.append("The input parameters should contain either `text`, `input_ids` or `messages`")
    else:
        if "text" in req_dict:
            if not isinstance(req_dict["text"], str):
                error_msg.append("The `text` in input parameters must be a string")
            elif req_dict["text"] == "":
                error_msg.append("The `text` in input parameters cannot be empty")
        if "system" in req_dict and not isinstance(req_dict["system"], str):
            error_msg.append("The `system` in input parameters must be a string")
        if "input_ids" in req_dict and not isinstance(req_dict["input_ids"], list):
            error_msg.append("The `input_ids` in input parameters must be a list")
        if "messages" in req_dict:
            msg_len = len(req_dict["messages"])
            if msg_len % 2 == 0:
                error_msg.append(f"The number of the message {msg_len} must be odd")
            if not all("content" in item for item in req_dict["messages"]):
                error_msg.append("The item in messages must include `content`")

    if "req_id" not in req_dict:
        error_msg.append("The input parameters should contain `req_id`.")

    if "min_dec_len" in req_dict and \
        (not isinstance(req_dict["min_dec_len"], int) or req_dict["min_dec_len"] < 1):
        error_msg.append("The `min_dec_len` must be an integer and greater than 0")

    # 如果设置了seq_len和max_tokens，最终都赋值给max_dec_len
    keys = ("max_dec_len", "seq_len", "max_tokens")
    for key in keys:
        if key in req_dict and (not isinstance(req_dict[key], int) or req_dict[key] < 1):
            error_msg.append(f"The `{key}` must be an integer and greater than 0")
    if "seq_len" in req_dict and "max_dec_len" not in req_dict:
        req_dict["max_dec_len"] = req_dict["seq_len"]
    if "max_tokens" in req_dict and "max_dec_len" not in req_dict:
        req_dict["max_dec_len"] = req_dict["max_tokens"]

    # 简化处理，topp和top_p只允许有一个，且最终都赋值给topp
    keys = ("topp", "top_p")
    if sum([key in req_dict for key in keys]) > 1:
        error_msg.append(f"Only one of {keys} should be set")
    else:
        for key in keys:
            if key in req_dict and not 0 <= req_dict[key] <= 1:
                error_msg.append(f"The `{key}` must be in [0, 1]")
        if "top_p" in req_dict and "topp" not in req_dict:
            req_dict["topp"] = req_dict["top_p"]

    if "temperature" in req_dict and not 0 <= req_dict["temperature"]:
        error_msg.append(f"The `temperature` must be >= 0")

    if "eos_token_ids" in req_dict:
        if isinstance(req_dict["eos_token_ids"], int):
            req_dict["eos_token_ids"] = [req_dict["eos_token_ids"]]
        elif isinstance(req_dict["eos_token_ids"], tuple):
            req_dict["eos_token_ids"] = list(req_dict["eos_token_ids"])
        if not isinstance(req_dict["eos_token_ids"], list):
            error_msg.append("The `eos_token_ids` must be an list")
        elif len(req_dict["eos_token_ids"]) > 1:
            error_msg.append("The length of `eos_token_ids` must be 1 if you set it")

    # 简化处理，infer_seed和seed只允许有一个，且最终都赋值给infer_seed
    keys = ("infer_seed", "seed")
    if sum([key in req_dict for key in keys]) > 1:
        error_msg.append(f"Only one of {keys} should be set")
    else:
        if "seed" in req_dict and "infer_seed" not in req_dict:
            req_dict["infer_seed"] = req_dict["seed"]

    if "stream" in req_dict and not isinstance(req_dict["stream"], bool):
        error_msg.append("The `stream` must be a boolean")

    if "response_type" in req_dict and (req_dict["response_type"].lower() not in ("fastdeploy", "openai")):
        error_msg.append("The `response_type` must be either `fastdeploy` or `openai`.")

    # 返回信息
    return error_msg

def add_default_params(req_dict):
    """
    给req_dict字典添加默认值。
    注意：虽然infer.py中设置请求参数有默认值，但为了统一，这里提前设置默认值。请保证此处默认值和infer.py中一致。
    返回添加默认值后的req_dict字典。

    """
    assert isinstance(req_dict, dict), "The `req_dict` must be a dict."
    if "min_dec_len" not in req_dict:
        req_dict["min_dec_len"] = 1
    if "topp" not in req_dict:
        req_dict["topp"] = 0.7
    if "temperature" not in req_dict:
        req_dict["temperature"] = 0.95
    if "penalty_score" not in req_dict:
        req_dict["penalty_score"] = 1.0
    if "frequency_score" not in req_dict:
        req_dict["frequency_score"] = 0.0
    if "presence_score" not in req_dict:
        req_dict["presence_score"] = 0.0
    return req_dict
