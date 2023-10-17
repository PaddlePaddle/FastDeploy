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

import json
import logging
from logging.handlers import TimedRotatingFileHandler
import numpy as np
import argparse

import time
import random
from http import HTTPStatus
import tornado
from tornado import web
from tornado.concurrent import run_on_executor
from concurrent.futures import ThreadPoolExecutor

from utils.conversation import *
from Client import *

parse = argparse.ArgumentParser()
parse.add_argument('--url', type=str, help='grpc server url')
parse.add_argument('--port', type=int, help='openai http port', default=2001)
parse.add_argument('--model', type=str, help='model name', default="model")


def parse_parameters(parameters_config, name, default_value):
    if name not in parameters_config:
        return default_value
    return parameters_config[name]


def create_error_response(status_code, msg):
    output = {
        "status": status_code,
        "errResponse": {
            "message": msg,
            "type": "invalid_request_error"
        }
    }
    return output


class ChatCompletionApiHandler(web.RequestHandler):
    """
    This handler provides OpenAI's ChatCompletion API。

    See  https://platform.openai.com/docs/api-reference/chat/create
    for the API specification. This API mimics the OpenAI ChatCompletion API.

    NOTE: Currently we do not support the following features:
        - n （currently only support 1）
        - logit_bias 
        - logprobs
        - stop （currently support token id）
        - function_call (Users should implement this by themselves)
        - function (Users should implement this by themselves)
    """
    executor = ThreadPoolExecutor(20)

    def __init__(self, application, request, **kwargs):
        web.RequestHandler.__init__(self, application, request, **kwargs)

    def initialize(self, url, model_name):
        self._client = grpcClient(base_url=url, model_name=model_name)

    @tornado.gen.coroutine
    def post(self):
        """
        POST METHOD
        """
        body = self.request.body
        remote_ip = self.request.remote_ip
        start_time = time.time()
        if not body:
            out_json = {"errorCode": 4000101}
            result_str = json.dumps(out_json, ensure_ascii=False)
            logging.warning(
                f"request receieved from remote ip:{remote_ip}, body=None,\
                result={result_str}, time_cost={time.time() - start_time : 0.5f}"
            )
            self.write(result_str)
        else:
            body = json.loads(body)
            logging.info(
                f"request receieved from remote ip:{remote_ip}, body={json.dumps(body, ensure_ascii=False)}"
            )
            err = self.valid_body(body)
            if err is None:
                data = yield self.run_req(body)
                if data is None:
                    out_json = create_error_response(4000102,
                                                     "result is empty")
                else:
                    out_json = {"outputs": [data], "status": 0}
                result_str = json.dumps(out_json, ensure_ascii=False)
            else:
                result_str = json.dumps(err, ensure_ascii=False)

            logging.info(
                f"request returned, result={result_str}, time_cost={time.time() - start_time : 0.5f}"
            )
            self.write(result_str)

    def valid_body(self, request):
        """
        Check whether the request body is legal
        
        Args:
            request (dict):
        
        Returns:
            Union[dict, None]: 
            If the request body is valid, return None; 
            otherwise, return json with the error message 
        """
        if request['model'] != self._client._model_name:
            return create_error_response(
                HTTPStatus.BAD_REQUEST,
                "current model is not currently supported")
        if 'n' in request and request['n'] != 1:
            return create_error_response(HTTPStatus.BAD_REQUEST,
                                         "n only support 1")
        if 'logit_bias' in request and request['logit_bias'] is not None:
            return create_error_response(
                HTTPStatus.BAD_REQUEST,
                "logit_bias is not currently supported")
        if 'functions' in request and request['functions'] is not None:
            return create_error_response(
                HTTPStatus.BAD_REQUEST, "functions is not currently supported")
        if 'function_call' in request and request['function_call'] is not None:
            return create_error_response(
                HTTPStatus.BAD_REQUEST,
                "function_call is not currently supported")
        return None

    def gen_prompt(self, request):
        conv = get_conv_template(request['model'])
        if isinstance(request['messages'], str):
            prompt = request['messages']
        else:
            for message in request['messages']:
                msg_role = message["role"]
                if msg_role == "system":
                    conv.system_message = message["content"]
                elif msg_role == "user":
                    conv.append_message(conv.roles[0], message["content"])
                elif msg_role == "assistant":
                    conv.append_message(conv.roles[1], message["content"])
                else:
                    raise ValueError(f"Unknown role: {msg_role}")

            # Add a blank message for the assistant.
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

        return prompt

    @run_on_executor
    def run_req(self, body):
        req_id = random.randint(0, 100000)
        prompt = self.gen_prompt(body)
        result = self._client.generate(
            request_id=str(req_id),
            prompt=prompt,
            top_p=parse_parameters(body, 'top_p', 0.0),
            temperature=parse_parameters(body, 'temperature', 1.0),
            max_dec_len=parse_parameters(body, 'max_tokens', 1024),
            frequency_score=parse_parameters(body, 'frequency_penalty', 0.99),
            presence_score=parse_parameters(body, 'presence_penalty', 0.0),
            stream=parse_parameters(body, 'stream', False))
        return result


class CompletionApiHandler(web.RequestHandler):
    """
    This handler provides OpenAI's Completion API。

    See https://platform.openai.com/docs/api-reference/completions/create
    for the API specification. This API mimics the OpenAI Completion API.

    NOTE: Currently we do not support the following features:
        - best_of （currently only support 1）
        - n （currently only support 1）
        - echo (not currently support getting the logprobs of prompt tokens)
        - suffix (the language models we currently support do not support
          suffix)
        - logit_bias 
        - logprobs
        - stop （currently support token id）
    """
    executor = ThreadPoolExecutor(20)

    def __init__(self, application, request, **kwargs):
        web.RequestHandler.__init__(self, application, request, **kwargs)

    def initialize(self, url, model_name):
        self._client = grpcClient(base_url=url, model_name=model_name)

    @tornado.gen.coroutine
    def post(self):
        """
        POST METHOD
        """
        body = self.request.body
        remote_ip = self.request.remote_ip
        start_time = time.time()
        if not body:
            out_json = {"errorCode": 4000101}
            result_str = json.dumps(out_json, ensure_ascii=False)
            logging.warning(
                f"request receieved from remote ip:{remote_ip}, body=None,\
                result={result_str}, time_cost={time.time() - start_time : 0.5f}"
            )

            self.write(result_str)
        else:
            body = json.loads(body)
            logging.info(
                f"request receieved from remote ip:{remote_ip}, body={json.dumps(body, ensure_ascii=False)}"
            )
            err = self.valid_body(body)
            if err is None:
                data = yield self.run_req(body)
                if data is None:
                    out_json = create_error_response(4000102,
                                                     "result is empty")
                else:
                    out_json = {"outputs": [data], "status": 0}
                result_str = json.dumps(out_json, ensure_ascii=False)
            else:
                result_str = json.dumps(err, ensure_ascii=False)

            logging.info(
                f"request returned, result={result_str}, time_cost={time.time() - start_time : 0.5f}"
            )
            self.write(result_str)

    def valid_body(self, request):
        """
        Check whether the request body is legal
        
        Args:
            request (dict):
        
        Returns:
            Union[dict, None]: 
            If the request body is valid, return None; 
            otherwise, return json with the error message 
        """
        if request['model'] != self._client._model_name:
            return create_error_response(
                HTTPStatus.BAD_REQUEST,
                "current model is not currently supported")
        if 'n' in request and request['n'] != 1:
            return create_error_response(HTTPStatus.BAD_REQUEST,
                                         "n only support 1")
        if 'best_of' in request and request['best_of'] != 1:
            return create_error_response(HTTPStatus.BAD_REQUEST,
                                         "best_of only support 1")
        if 'echo' in request and request['echo']:
            return create_error_response(HTTPStatus.BAD_REQUEST,
                                         "not suport echo")
        if 'suffix' in request and request['suffix'] is not None:
            return create_error_response(HTTPStatus.BAD_REQUEST,
                                         "not suport suffix")
        if 'logit_bias' in request and request['logit_bias'] is not None:
            return create_error_response(
                HTTPStatus.BAD_REQUEST,
                "logit_bias is not currently supported")
        if 'logprobs' in request and request['logprobs'] is not None:
            return create_error_response(
                HTTPStatus.BAD_REQUEST, "logprobs is not currently supported")

        return None

    @run_on_executor
    def run_req(self, body):
        req_id = random.randint(0, 100000)
        result = self._client.generate(
            request_id=str(req_id),
            prompt=body['prompt'],
            top_p=parse_parameters(body, 'top_p', 0.0),
            temperature=parse_parameters(body, 'temperature', 1.0),
            max_dec_len=parse_parameters(body, 'max_tokens', 1024),
            frequency_score=parse_parameters(body, 'frequency_penalty', 0.99),
            presence_score=parse_parameters(body, 'presence_penalty', 0.0),
            stream=parse_parameters(body, 'stream', False))
        return result


if __name__ == '__main__':
    args = parse.parse_args()
    port = args.port
    app = web.Application([("/v1/completions", CompletionApiHandler,
                            dict(url=args.url, model_name=args.model)),
                           ("/v1/chat/completions", ChatCompletionApiHandler,
                            dict(url=args.url, model_name=args.model))])

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = tornado.log.LogFormatter(
        fmt=
        '%(levelname)-8s %(asctime)s %(process)-5s %(filename)s[line:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    file_handler = TimedRotatingFileHandler(filename='log/server.log',
                                            when='D',
                                            interval=3,
                                            backupCount=90,
                                            encoding='utf-8',
                                            delay=False)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    app.listen(port)
    print("Server started")
    logging.info(f"Server started at port:{port}")
    tornado.ioloop.IOLoop.current().start()