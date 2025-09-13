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
