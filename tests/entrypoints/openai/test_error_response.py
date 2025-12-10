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

import unittest

from pydantic import ValidationError

from fastdeploy.entrypoints.openai.protocol import ErrorResponse


class TestErrorResponse(unittest.TestCase):
    def test_valid_error_response(self):
        data = {
            "error": {
                "message": "Invalid top_p value",
                "type": "invalid_request_error",
                "param": "top_p",
                "code": "invalid_value",
            }
        }
        err_resp = ErrorResponse(**data)
        self.assertEqual(err_resp.error.message, "Invalid top_p value")
        self.assertEqual(err_resp.error.param, "top_p")
        self.assertEqual(err_resp.error.code, "invalid_value")

    def test_missing_message_field(self):
        data = {"error": {"type": "invalid_request_error", "param": "messages", "code": "missing_required_parameter"}}
        with self.assertRaises(ValidationError):
            ErrorResponse(**data)


if __name__ == "__main__":
    unittest.main()
