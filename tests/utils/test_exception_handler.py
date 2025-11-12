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

import json
import unittest
from http import HTTPStatus

from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from fastdeploy.utils import ErrorCode, ExceptionHandler, ParameterError


class DummyRequest:
    url = "http://testserver/test"


class TestParameterError(unittest.TestCase):
    def test_parameter_error_init(self):
        exc = ParameterError("param1", "error message")
        self.assertEqual(exc.param, "param1")
        self.assertEqual(exc.message, "error message")
        self.assertEqual(str(exc), "error message")


class TestExceptionHandler(unittest.IsolatedAsyncioTestCase):

    async def test_handle_exception(self):
        """普通异常应返回 500 + internal_error"""
        exc = RuntimeError("Something went wrong")
        resp: JSONResponse = await ExceptionHandler.handle_exception(None, exc)
        body = json.loads(resp.body.decode())
        self.assertEqual(resp.status_code, HTTPStatus.INTERNAL_SERVER_ERROR)
        self.assertEqual(body["error"]["type"], "internal_error")
        self.assertIn("Something went wrong", body["error"]["message"])

    async def test_handle_request_validation_missing_messages(self):
        """缺少 messages 参数时，应返回 missing_required_parameter"""
        exc = RequestValidationError([{"loc": ("body", "messages"), "msg": "Field required", "type": "missing"}])
        dummy_request = DummyRequest()
        resp: JSONResponse = await ExceptionHandler.handle_request_validation_exception(dummy_request, exc)
        data = json.loads(resp.body.decode())
        self.assertEqual(resp.status_code, HTTPStatus.BAD_REQUEST)
        self.assertEqual(data["error"]["param"], "messages")
        self.assertEqual(data["error"]["code"], ErrorCode.MISSING_REQUIRED_PARAMETER)
        self.assertIn("Field required", data["error"]["message"])

    async def test_handle_request_validation_invalid_value(self):
        """参数非法时，应返回 invalid_value"""
        exc = RequestValidationError(
            [{"loc": ("body", "top_p"), "msg": "Input should be less than or equal to 1", "type": "value_error"}]
        )
        dummy_request = DummyRequest()
        resp: JSONResponse = await ExceptionHandler.handle_request_validation_exception(dummy_request, exc)
        data = json.loads(resp.body.decode())
        self.assertEqual(resp.status_code, HTTPStatus.BAD_REQUEST)
        self.assertEqual(data["error"]["param"], "top_p")
        self.assertEqual(data["error"]["code"], ErrorCode.INVALID_VALUE)
        self.assertIn("less than or equal to 1", data["error"]["message"])


if __name__ == "__main__":
    unittest.main()
