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
import logging
import queue
import traceback
import uuid
from functools import partial

import numpy as np
import tritonclient.grpc as grpcclient
from fastdeploy_client.message import ChatMessage
from fastdeploy_client.utils import is_enable_benchmark
from tritonclient import utils as triton_utils


class ChatBotClass(object):
    """
    initiating conversations through the tritonclient interface of the model service.
    """
    def __init__(self, hostname, port, timeout=120):
        """
        Initialization function

        Args:
            hostname (str): gRPC hostname
            port (int): gRPC port
            timeout (int): Request timeout, default is 120 seconds.

        Returns:
            None
        """
        self.url = f"{hostname}:{port}"
        self.timeout = timeout

    def stream_generate(self,
                        message,
                        max_dec_len=1024,
                        min_dec_len=1,
                        topp=0.7,
                        temperature=0.95,
                        frequency_score=0.0,
                        penalty_score=1.0,
                        presence_score=0.0,
                        system=None,
                        **kwargs):
        """
        Streaming interface

        Args:
            message (Union[str, List[str], ChatMessage]): 消息内容或ChatMessage对象
            max_dec_len (int, optional): 最大解码长度. Defaults to 1024.
            min_dec_len (int, optional): 最小解码长度. Defaults to 1.
            topp (float, optional): 控制随机性参数，数值越大则随机性越大，范围是0~1. Defaults to 0.7.
            temperature (float, optional): 温度值. Defaults to 0.95.
            frequency_score (float, optional): 频率分数. Defaults to 0.0.
            penalty_score (float, optional): 惩罚分数. Defaults to 1.0.
            presence_score (float, optional): 存在分数. Defaults to 0.0.
            system (str, optional): 系统设定. Defaults to None.
            **kwargs: 其他参数
                req_id (str, optional): 请求ID，用于区分不同的请求. Defaults to None.
                eos_token_ids (List[int], optional): 指定结束的token id. Defaults to None.
                benchmark (bool, optional): 设置benchmark模式，如果是则返回完整的response. Defaults to False.
                timeout (int, optional): 请求超时时间，不设置则使用120s. Defaults to None.

        Returns:
            返回一个生成器，每次yield返回一个字典。
            正常情况下，生成器返回字典的示例{"req_id": "xxx", "token": "好的", "is_end": 0}，其中token为生成的字符，is_end表明是否为最后一个字符（0表示否，1表示是）
            错误情况下，生成器返回错误信息的字典，示例 {"req_id": "xxx", "error_msg": "error message"}
        """
        try:
            # 准备输入
            model_name = "model"
            inputs = [grpcclient.InferInput("IN", [1], triton_utils.np_to_triton_dtype(np.object_))]
            outputs = [grpcclient.InferRequestedOutput("OUT")]
            output_data = OutputData()

            msg = message.message if isinstance(message, ChatMessage) else message
            input_data = self._prepare_input_data(msg, max_dec_len, min_dec_len,
                                        topp, temperature, frequency_score,
                                        penalty_score, presence_score, **kwargs)
            req_id = input_data["req_id"]
            inputs[0].set_data_from_numpy(np.array([json.dumps([input_data])], dtype=np.object_))
            timeout = kwargs.get("timeout", self.timeout)

            with grpcclient.InferenceServerClient(url=self.url, verbose=False) as triton_client:
                # 建立连接
                triton_client.start_stream(callback=partial(triton_callback, output_data))
                # 发送请求
                triton_client.async_stream_infer(model_name=model_name,
                                                    inputs=inputs,
                                                    request_id=req_id,
                                                    outputs=outputs)
                # 处理结果
                answer_str = ""
                enable_benchmark = is_enable_benchmark(**kwargs)
                while True:
                    try:
                        response = output_data._completed_requests.get(timeout=timeout)
                    except queue.Empty:
                        yield {"req_id": req_id, "error_msg": f"Fetch response from server timeout ({timeout}s)"}
                        break
                    if type(response) == triton_utils.InferenceServerException:
                        yield {"req_id": req_id, "error_msg": f"InferenceServerException raised by inference: {response.message()}"}
                        break
                    else:
                        if enable_benchmark:
                            response = json.loads(response.as_numpy("OUT")[0])
                            if isinstance(response, (list, tuple)):
                                response = response[0]
                        else:
                            response = self._format_response(response, req_id)
                            token = response.get("token", "")
                            if isinstance(token, list):
                                token = token[0]
                            answer_str += token
                        yield response
                        if response.get("is_end") == 1 or response.get("error_msg") is not None:
                            break
            # 手动关闭
            triton_client.stop_stream(cancel_requests=True)
            triton_client.close()

            if isinstance(message, ChatMessage):
                message.message.append({"role": "assistant", "content": answer_str})
        except Exception as e:
            yield {"error_msg": f"{e}, details={str(traceback.format_exc())}"}

    def generate(self,
                 message,
                 max_dec_len=1024,
                 min_dec_len=1,
                 topp=0.7,
                 temperature=0.95,
                 frequency_score=0.0,
                 penalty_score=1.0,
                 presence_score=0.0,
                 system=None,
                 **kwargs):
        """
        整句返回，直接使用流式返回的接口。

        Args:
            message (Union[str, List[str], ChatMessage]): 消息内容或ChatMessage对象
            max_dec_len (int, optional): 最大解码长度. Defaults to 1024.
            min_dec_len (int, optional): 最小解码长度. Defaults to 1.
            topp (float, optional): 控制随机性参数，数值越大则随机性越大，范围是0~1. Defaults to 0.7.
            temperature (float, optional): 温度值. Defaults to 0.95.
            frequency_score (float, optional): 频率分数. Defaults to 0.0.
            penalty_score (float, optional): 惩罚分数. Defaults to 1.0.
            presence_score (float, optional): 存在分数. Defaults to 0.0.
            system (str, optional): 系统设定. Defaults to None.
            **kwargs: 其他参数
                req_id (str, optional): 请求ID，用于区分不同的请求. Defaults to None.
                eos_token_ids (List[int], optional): 指定结束的token id. Defaults to None.
                timeout (int, optional): 请求超时时间，不设置则使用120s. Defaults to None.

        Returns:
            返回一个字典。
            正常情况下，返回字典的示例{"req_id": "xxx", "results": "好的，我知道了。"}
            错误情况下，返回错误信息的字典，示例 {"req_id": "xxx", "error_msg": "error message"}
        """
        stream_response = self.stream_generate(message, max_dec_len,
                                               min_dec_len, topp, temperature,
                                               frequency_score, penalty_score,
                                               presence_score, system, **kwargs)
        results = ""
        token_ids = list()
        error_msg = None
        for res in stream_response:
            if "token" not in res or "error_msg" in res:
                error_msg = {"error_msg": f"response error, please check the info: {res}"}
            elif isinstance(res["token"], list):
                results = res["token"]
                token_ids = res["token_ids"]
            else:
                results += res["token"]
                token_ids += res["token_ids"]
        if error_msg:
            return {"req_id": res["req_id"], "error_msg": error_msg}
        else:
            return {"req_id": res["req_id"], "results": results, "token_ids": token_ids}

    def _prepare_input_data(self,
                        message,
                        max_dec_len=1024,
                        min_dec_len=2,
                        topp=0.0,
                        temperature=1.0,
                        frequency_score=0.0,
                        penalty_score=1.0,
                        presence_score=0.0,
                        system=None,
                        **kwargs):
        """
        准备输入数据。
        """
        inputs = {
            "max_dec_len": max_dec_len,
            "min_dec_len": min_dec_len,
            "topp": topp,
            "temperature": temperature,
            "frequency_score": frequency_score,
            "penalty_score": penalty_score,
            "presence_score": presence_score,
        }

        if system is not None:
            inputs["system"] = system

        inputs["req_id"] = kwargs.get("req_id", str(uuid.uuid4()))
        if "eos_token_ids" in kwargs and kwargs["eos_token_ids"] is not None:
            inputs["eos_token_ids"] = kwargs["eos_token_ids"]
        inputs["response_timeout"] = kwargs.get("timeout", self.timeout)

        if isinstance(message, str):
            inputs["text"] = message
        elif isinstance(message, list):
            assert len(message) % 2 == 1, \
                "The length of message should be odd while it's a list."
            assert message[-1]["role"] == "user", \
                "The {}-th element key should be 'user'".format(len(message) - 1)
            for i in range(0, len(message) - 1, 2):
                assert message[i]["role"] == "user", \
                    "The {}-th element key should be 'user'".format(i)
                assert message[i + 1]["role"] == "assistant", \
                    "The {}-th element key should be 'assistant'".format(i + 1)
            inputs["messages"] = message
        else:
            raise Exception(
                "The message should be string or list of dict like [{'role': "
                "'user', 'content': 'Hello, what's your name?''}]"
            )

        return inputs

    def _format_response(self, response, req_id):
        """
        对服务返回字段进行格式化
        """
        response = json.loads(response.as_numpy("OUT")[0])
        if isinstance(response, (list, tuple)):
            response = response[0]
        is_end = response.get("is_end", False)

        if "error_msg" in response:
            return {"req_id": req_id, "error_msg": response["error_msg"]}
        elif "choices" in response:
            token = [x["token"] for x in response["choices"]]
            token_ids = [x["token_ids"] for x in response["choices"]]
            return {"req_id": req_id, "token": token, "token_ids": token_ids, "is_end": 1}
        elif "token" not in response and "result" not in response:
            return {"req_id": req_id, "error_msg": f"The response should contain 'token' or 'result', but got {response}"}
        else:
            token_ids = response.get("token_ids", [])
            if "result" in response:
                token = response["result"]
            elif "token" in response:
                token = response["token"]
            return {"req_id": req_id, "token": token, "token_ids": token_ids, "is_end": is_end}


class OutputData:
    """接收Triton服务返回的数据"""
    def __init__(self):
        self._completed_requests = queue.Queue()


def triton_callback(output_data, result, error):
    """Triton客户端的回调函数"""
    if error:
        output_data._completed_requests.put(error)
    else:
        output_data._completed_requests.put(result)


class ChatBot(object):
    """
    对外的接口，用于创建ChatBotForPushMode的示例
    """
    def __new__(cls, hostname, port, timeout=120):
        """
        初始化函数，用于创建一个GRPCInferenceService客户端对象
        Args:
            hostname (str): 服务器的地址
            port (int): 服务器的端口号
            timeout (int): 请求超时时间，单位为秒，默认120秒
        Returns:
            ChatBotClass: 返回一个BaseChatBot对象
        """
        if not isinstance(hostname, str) or not hostname:
            raise ValueError("Invalid hostname")
        if not isinstance(port, int) or port <= 0 or port > 65535:
            raise ValueError("Invalid port number")

        return ChatBotClass(hostname, port, timeout)
