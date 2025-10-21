import time
from unittest import TestCase
from unittest.mock import MagicMock, patch


class TestZmqSendGeneratedTokens(TestCase):
    @patch("time.sleep", return_value=None)
    @patch("fastdeploy.engine.common_engine.EngineSevice.__init__", return_value=None)
    def setUp(self, mock_init, mock_sleep):
        from fastdeploy.engine.common_engine import EngineSevice

        self.obj = EngineSevice(None)
        self.obj.running = True

        # mock 依赖组件
        self.obj.scheduler = MagicMock()
        self.obj.send_response_server = MagicMock()
        self.obj._decode_token = MagicMock()
        self.obj._decode_token.return_value = ("decoded_text", [101, 102])
        self.obj.llm_logger = MagicMock()

    def test_zmq_send_generated_tokens_normal_case(self):
        mock_output = MagicMock()
        mock_output.outputs.decode_type = 0
        mock_output.outputs.token_ids = [1, 2, 3]
        mock_output.finished = True

        self.obj.scheduler.get_results.side_effect = [
            {"req_1": [mock_output]},
            {},
        ]

        def stop_running():
            time.sleep(0.01)
            self.obj.running = False

        import threading

        threading.Thread(target=stop_running).start()

        self.obj._zmq_send_generated_tokens()

        self.obj.send_response_server.send_response.assert_called_once()
        args, kwargs = self.obj.send_response_server.send_response.call_args
        assert args[0] == "req_1"
        assert isinstance(args[1], list)
        assert args[1][0].outputs.text == "decoded_text"
