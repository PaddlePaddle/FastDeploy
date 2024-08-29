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
import queue
import time
import uuid
from datetime import datetime
from functools import partial
from typing import Dict, List, Optional

import numpy as np
import tritonclient.grpc as grpcclient
from pydantic import BaseModel, Field
from tritonclient import utils as triton_utils


class Req(BaseModel):
    """请求参数的类"""
    # 传入模型服务的请求参数
    req_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    input_ids: Optional[List[int]] = None
    text: Optional[str] = None
    messages: Optional[List] = None
    max_dec_len: Optional[int] = None
    seq_len: Optional[int] = None  # 保留seq_len为了兼容支持
    min_dec_len: Optional[int] = None
    temperature: Optional[float] = None
    topp: Optional[float] = None
    penalty_score: Optional[float] = None
    frequency_score: Optional[float] = None
    presence_score: Optional[float] = None
    system: Optional[str] = None
    return_all_tokens: Optional[bool] = None
    eos_token_ids: Optional[List[int]] = None
    benchmark: bool = False
    # http服务使用的请求参数
    stream: bool = False
    timeout: int = 300

    def to_dict_for_infer(self):
        """将请求参数转化为字典，去掉为None的字段，避免传递给模型服务出错"""
        req_dict = {}
        for key, value in self.dict().items():
            if value is not None:
                req_dict[key] = value
        return req_dict


def chat_completion_generator(infer_grpc_url: str, req: Req, yield_json: bool) -> Dict:
    """
    基于Triton推理服务的聊天补全结果的生成器。
    Args:
        infer_grpc_url (str): Triton推理服务的gRPC URL。
        req (Request): 聊天补全请求。
        yield_json (bool): 是否返回json格式，否则返回Resp类
    Returns:
        dict: 聊天补全结果的生成器。
            如果正常，返回{'token': xxx, 'is_end': xxx, 'send_idx': xxx, ..., 'error_msg': '', 'error_code': 0}
            如果异常，返回{'error_msg': xxx, 'error_code': xxx}，error_msg字段不为空，error_code字段不为0
    """
    class _TritonOutputData:
        """接收Triton服务返回的数据"""
        def __init__(self):
            self._completed_requests = queue.Queue()

    def _triton_callback(output_data, result, error):
        """Triton客户端的回调函数"""
        if error:
            output_data._completed_requests.put(error)
        else:
            output_data._completed_requests.put(result)

    def _format_resp(resp_dict):
        if yield_json:
            return json.dumps(resp_dict, ensure_ascii=False) + "\n"
        else:
            return resp_dict

    # 准备请求数据
    timeout = req.timeout
    req_id = req.req_id
    req_dict = req.to_dict_for_infer()
    http_received_time = datetime.now()

    inputs = [grpcclient.InferInput("IN", [1], triton_utils.np_to_triton_dtype(np.object_))]
    inputs[0].set_data_from_numpy(np.array([json.dumps([req_dict])], dtype=np.object_))
    outputs = [grpcclient.InferRequestedOutput("OUT")]
    output_data = _TritonOutputData()

    # 建立连接
    with grpcclient.InferenceServerClient(url=infer_grpc_url, verbose=False) as triton_client:
        triton_client.start_stream(callback=partial(_triton_callback, output_data))

        # 发送请求
        triton_client.async_stream_infer(model_name="model",
                                            inputs=inputs,
                                            request_id=req_dict['req_id'],
                                            outputs=outputs)
        # 处理返回结果
        while True:
            output_item = output_data._completed_requests.get(timeout=timeout)
            if type(output_item) == triton_utils.InferenceServerException:
                error_msg = f"status is {output_item.status()}, msg is {output_item.message()}"
                yield _format_resp({"error_msg": error_msg, "error_code": 500})
                break
            else:
                result = json.loads(output_item.as_numpy("OUT")[0])
                result = result[0] if isinstance(result, list) else result
                result["error_msg"] = result.get("error_msg", "")
                result["error_code"] = result.get("error_code", 0)
                if req.benchmark:
                    result["http_received_time"] = str(http_received_time)
                yield _format_resp(result)
                if (result.get("error_msg") or result.get("error_code")) or result.get("is_end") == 1:
                    break

        # 手动关闭连接
        triton_client.stop_stream()
        triton_client.close()

def chat_completion_result(infer_grpc_url: str, req: Req) -> Dict:
    """
    获取非流式生成结果
    Args:
        infer_grpc_url (str): gRPC服务地址
        req (Req): 请求参数对象
    Returns:
        dict: 聊天补全结果的生成器。
            如果正常，返回{'result': xxx, 'error_msg': '', 'error_code': 0}
            如果异常，返回{'result': '', 'error_msg': xxx, 'error_code': xxx}，error_msg字段不为空，error_code字段不为0
    """
    result = None
    error_resp = None
    for resp in chat_completion_generator(infer_grpc_url, req, yield_json=False):
        if resp.get("error_msg") or resp.get("error_code"):
            error_resp = resp
            error_resp["result"] = ""
        else:
            if resp.get('is_end') == 1:
                result = resp
                for key in ['token', 'is_end', 'send_idx', 'return_all_tokens', 'token']:
                    if key in result:
                        del result[key]
    if not result:
        error_resp = {
            "error_msg": "HTTP parsing data error",
            "error_code": 500,
            "result": "",
            "is_end": 1,
        }
    return error_resp if error_resp else result
