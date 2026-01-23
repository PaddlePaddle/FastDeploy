# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
from unittest.mock import patch

from fastapi.responses import JSONResponse

from fastdeploy.engine.request import ControlRequest, ControlResponse


class TestControlRequest(unittest.TestCase):
    """Test cases for ControlRequest class."""

    def test_initialization_basic(self):
        """Test basic initialization of ControlRequest."""
        request_id = "test_request_123"
        method = "get_metrics"

        request = ControlRequest(request_id=request_id, method=method)

        self.assertEqual(request.request_id, request_id)
        self.assertEqual(request.method, method)
        self.assertEqual(request.args, {})

    def test_initialization_with_args(self):
        """Test initialization with arguments."""
        request_id = "test_request_456"
        method = "reset_scheduler"
        args = {"force": True, "timeout": 30}

        request = ControlRequest(request_id=request_id, method=method, args=args)

        self.assertEqual(request.request_id, request_id)
        self.assertEqual(request.method, method)
        self.assertEqual(request.args, args)

    def test_from_dict_basic(self):
        """Test creating ControlRequest from dictionary (basic case)."""
        data = {"request_id": "test_from_dict", "method": "status_check"}

        request = ControlRequest.from_dict(data)

        self.assertEqual(request.request_id, data["request_id"])
        self.assertEqual(request.method, data["method"])
        self.assertEqual(request.args, {})

    def test_from_dict_with_args(self):
        """Test creating ControlRequest from dictionary with arguments."""
        data = {
            "request_id": "test_from_dict_args",
            "method": "configure",
            "args": {"max_requests": 100, "queue_timeout": 60},
        }

        request = ControlRequest.from_dict(data)

        self.assertEqual(request.request_id, data["request_id"])
        self.assertEqual(request.method, data["method"])
        self.assertEqual(request.args, data["args"])

    def test_to_dict_basic(self):
        """Test converting ControlRequest to dictionary (basic case)."""
        request = ControlRequest(request_id="test_to_dict", method="health_check")

        result = request.to_dict()

        expected = {"request_id": "test_to_dict", "method": "health_check", "args": {}}
        self.assertEqual(result, expected)

    def test_to_dict_with_args(self):
        """Test converting ControlRequest to dictionary with arguments."""
        args = {"setting1": "value1", "setting2": 42}
        request = ControlRequest(request_id="test_to_dict_args", method="update_settings", args=args)

        result = request.to_dict()

        expected = {"request_id": "test_to_dict_args", "method": "update_settings", "args": args}
        self.assertEqual(result, expected)

    def test_get_method(self):
        """Test get_method method."""
        method = "custom_operation"
        request = ControlRequest(request_id="test", method=method)

        self.assertEqual(request.get_method(), method)

    def test_get_args(self):
        """Test get_args method."""
        args = {"param1": "value1", "param2": 123}
        request = ControlRequest(request_id="test", method="test", args=args)

        result_args = request.get_args()

        self.assertEqual(result_args, args)
        # Ensure it returns a copy, not the original dict
        self.assertIsNot(result_args, args)

    def test_is_control_request_valid(self):
        """Test is_control_request method with valid data."""
        valid_data = [
            {"request_id": "test1", "method": "method1"},
            {"request_id": "test2", "method": "method2", "args": {}},
            {"request_id": "test3", "method": "method3", "args": {"key": "value"}},
        ]

        for data in valid_data:
            with self.subTest(data=data):
                self.assertTrue(ControlRequest.is_control_request(data))

    def test_is_control_request_invalid(self):
        """Test is_control_request method with invalid data."""
        invalid_data = [
            # Missing required fields
            {"method": "test"},  # missing request_id
            {"request_id": "test"},  # missing method
            # Wrong field types
            {"request_id": 123, "method": "test"},  # request_id not string
            {"request_id": "test", "method": 456},  # method not string
            {"request_id": "test", "method": "test", "args": "not_a_dict"},  # args not dict
            # Not a dict
            "not_a_dict",
            123,
            None,
        ]

        for data in invalid_data:
            with self.subTest(data=data):
                self.assertFalse(ControlRequest.is_control_request(data))

    def test_repr_simple(self):
        """Test __repr__ method in simple mode."""
        with patch("fastdeploy.envs.FD_DEBUG", False):
            request = ControlRequest(request_id="test_repr", method="test_method")
            repr_str = repr(request)

            self.assertIn("ControlRequest", repr_str)
            self.assertIn("test_repr", repr_str)
            self.assertIn("test_method", repr_str)
            self.assertNotIn("args", repr_str)  # Args not shown in simple mode

    def test_repr_debug_mode(self):
        """Test __repr__ method in debug mode."""
        with patch("fastdeploy.envs.FD_DEBUG", True):
            args = {"debug_param": "debug_value"}
            request = ControlRequest(request_id="test_repr", method="test_method", args=args)
            repr_str = repr(request)

            self.assertIn("ControlRequest", repr_str)
            self.assertIn("test_repr", repr_str)
            self.assertIn("test_method", repr_str)
            self.assertIn("debug_param", repr_str)  # Args shown in debug mode


class TestControlResponse(unittest.TestCase):
    """Test cases for ControlResponse class."""

    def test_initialization_basic(self):
        """Test basic initialization of ControlResponse."""
        request_id = "test_response_123"

        response = ControlResponse(request_id=request_id)

        self.assertEqual(response.request_id, request_id)
        self.assertEqual(response.error_code, 200)
        self.assertIsNone(response.error_message)
        self.assertIsNone(response.result)
        self.assertTrue(response.finished)

    def test_initialization_with_all_params(self):
        """Test initialization with all parameters."""
        request_id = "test_response_456"
        error_code = 404
        error_message = "Not found"
        result = {"data": "some_result"}
        finished = False

        response = ControlResponse(
            request_id=request_id, error_code=error_code, error_message=error_message, result=result, finished=finished
        )

        self.assertEqual(response.request_id, request_id)
        self.assertEqual(response.error_code, error_code)
        self.assertEqual(response.error_message, error_message)
        self.assertEqual(response.result, result)
        self.assertEqual(response.finished, finished)

    def test_initialization_error_cases(self):
        """Test initialization with various error codes."""
        test_cases = [
            (200, None, True),  # Success case
            (400, "Bad Request", False),  # Client error
            (500, "Internal Error", True),  # Server error
        ]

        for error_code, error_message, finished in test_cases:
            with self.subTest(error_code=error_code):
                response = ControlResponse(
                    request_id="test", error_code=error_code, error_message=error_message, finished=finished
                )

                self.assertEqual(response.error_code, error_code)
                self.assertEqual(response.error_message, error_message)
                self.assertEqual(response.finished, finished)

    def test_from_dict_basic(self):
        """Test creating ControlResponse from dictionary (basic case)."""
        data = {"request_id": "test_from_dict"}

        response = ControlResponse.from_dict(data)

        self.assertEqual(response.request_id, data["request_id"])
        self.assertEqual(response.error_code, 200)
        self.assertIsNone(response.error_message)
        self.assertIsNone(response.result)
        self.assertTrue(response.finished)

    def test_from_dict_with_all_fields(self):
        """Test creating ControlResponse from dictionary with all fields."""
        data = {
            "request_id": "test_from_dict_full",
            "error_code": 500,
            "error_message": "Test error",
            "result": {"key": "value"},
            "finished": False,
        }

        response = ControlResponse.from_dict(data)

        self.assertEqual(response.request_id, data["request_id"])
        self.assertEqual(response.error_code, data["error_code"])
        self.assertEqual(response.error_message, data["error_message"])
        self.assertEqual(response.result, data["result"])
        self.assertEqual(response.finished, data["finished"])

    def test_to_dict_basic(self):
        """Test converting ControlResponse to dictionary (basic case)."""
        response = ControlResponse(request_id="test_to_dict")

        result = response.to_dict()

        expected = {
            "request_id": "test_to_dict",
            "finished": True,
            "error_code": 200,
            "error_message": None,
            "result": None,
        }
        self.assertEqual(result, expected)

    def test_to_dict_with_all_fields(self):
        """Test converting ControlResponse to dictionary with all fields."""
        response = ControlResponse(
            request_id="test_to_dict_full",
            error_code=400,
            error_message="Validation failed",
            result={"valid": False, "reason": "missing_field"},
            finished=False,
        )

        result = response.to_dict()

        expected = {
            "request_id": "test_to_dict_full",
            "finished": False,
            "error_code": 400,
            "error_message": "Validation failed",
            "result": {"valid": False, "reason": "missing_field"},
        }
        self.assertEqual(result, expected)

    def test_to_api_json_response_success(self):
        """Test converting to JSONResponse for successful response."""
        result_data = {"metrics": {"cpu_usage": 0.5, "memory_used": 1024}}
        response = ControlResponse(request_id="test_json_success", result=result_data)

        json_response = response.to_api_json_response()

        self.assertIsInstance(json_response, JSONResponse)
        self.assertEqual(json_response.status_code, 200)

        content = json_response.body.decode("utf-8")
        self.assertIn("success", content)
        self.assertIn("test_json_success", content)
        self.assertIn("cpu_usage", content)

    def test_to_api_json_response_error(self):
        """Test converting to JSONResponse for error response."""
        response = ControlResponse(request_id="test_json_error", error_code=503, error_message="Service unavailable")

        json_response = response.to_api_json_response()

        self.assertIsInstance(json_response, JSONResponse)
        self.assertEqual(json_response.status_code, 503)

        content = json_response.body.decode("utf-8")
        self.assertIn("error", content)
        self.assertIn("test_json_error", content)
        self.assertIn("Service unavailable", content)

    def test_repr_method(self):
        """Test __repr__ method."""
        response = ControlResponse(
            request_id="test_repr", error_code=200, error_message=None, result={"data": "test"}, finished=True
        )

        repr_str = repr(response)

        # Check that all important fields are represented
        self.assertIn("ControlResponse", repr_str)
        self.assertIn("test_repr", repr_str)
        self.assertIn("200", repr_str)
        self.assertIn("test", repr_str)  # from result
        self.assertIn("True", repr_str)  # finished flag


if __name__ == "__main__":
    unittest.main()
