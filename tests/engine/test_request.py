import unittest
from unittest import mock

from fastdeploy import envs
from fastdeploy.engine.request import Request


class TestRequestRepr(unittest.TestCase):
    def test_repr_normal_mode(self):
        """FD_DEBUG=False 时只显示 request_id"""
        envs.FD_DEBUG = False
        req = Request(request_id="abc123", prompt="Hello")
        self.assertEqual(repr(req), "Request(request_id=abc123)")

    def test_repr_debug_mode(self):
        """FD_DEBUG=True 时输出全部非私有、非 None 字段"""
        envs.FD_DEBUG = True
        req = Request(
            request_id="req1",
            prompt="Hi",
            prompt_token_ids=[1, 2, 3],
            prompt_token_ids_len=3,
            system="sys",
        )
        result = repr(req)
        self.assertIn("request_id='req1'", result)
        self.assertIn("prompt='Hi'", result)
        self.assertNotIn("_private_data", result)
        self.assertNotIn("none_field", result)
        self.assertTrue(result.startswith("Request("))
        self.assertTrue(result.endswith(")"))

    def test_repr_handles_exception(self):
        """测试 repr 异常分支"""
        envs.FD_DEBUG = True
        req = Request(request_id="err_test")
        with mock.patch("builtins.vars", side_effect=Exception("fail!")):
            result = repr(req)
        self.assertTrue(result.startswith("<Request repr failed:"))
        self.assertIn("fail!", result)


if __name__ == "__main__":
    unittest.main()
