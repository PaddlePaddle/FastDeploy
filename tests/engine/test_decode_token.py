import unittest
from unittest.mock import MagicMock, patch


class DummyDataProcessor:
    def __init__(self):
        self.decode_status = {}

    def ids2tokens(self, token_ids, req_id):
        return "", [], None


class TestDecodeToken(unittest.TestCase):
    @patch("fastdeploy.engine.common_engine.EngineSevice.__init__", return_value=None)
    def setUp(self, mock_init):
        from fastdeploy.engine.common_engine import EngineSevice

        self.obj = EngineSevice(None)
        self.obj.data_processor = DummyDataProcessor()

    @patch("fastdeploy.engine.common_engine.envs.FD_ENABLE_RETURN_TEXT", True)
    def test_decode_token_with_text(self):
        """测试：env 启用 + 返回非空 delta_text"""
        self.obj.data_processor.ids2tokens = MagicMock(return_value=("hello", [10, 11, 12, 13], None))
        self.obj.data_processor.decode_status = {"req_1": (1, 3)}

        delta_text, token_ids = self.obj._decode_token([1, 2, 3], "req_1", is_end=False)

        assert delta_text == "hello"
        assert token_ids == [11, 12]

    @patch("fastdeploy.engine.common_engine.envs.FD_ENABLE_RETURN_TEXT", True)
    def test_decode_token_empty_text(self):
        """测试：env 启用 + 返回空 delta_text"""
        self.obj.data_processor.ids2tokens = MagicMock(return_value=("", [10, 11, 12], None))
        self.obj.data_processor.decode_status = {"req_1": (0, 2)}

        delta_text, token_ids = self.obj._decode_token([1, 2], "req_1", is_end=False)

        assert delta_text == ""
        assert token_ids == []

    @patch("fastdeploy.engine.common_engine.envs.FD_ENABLE_RETURN_TEXT", True)
    def test_decode_token_with_is_end(self):
        """测试：is_end=True 时 decode_status 被删除"""
        self.obj.data_processor.ids2tokens = MagicMock(return_value=("bye", [1, 2, 3, 4], None))
        self.obj.data_processor.decode_status = {"req_2": (0, 2)}

        delta_text, token_ids = self.obj._decode_token([1, 2, 3], "req_2", is_end=True)

        assert "req_2" not in self.obj.data_processor.decode_status
