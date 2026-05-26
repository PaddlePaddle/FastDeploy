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

import unittest
from unittest.mock import MagicMock, patch


class TestCacheRequestHandlerLogsError(unittest.TestCase):
    def test_cache_request_handler_logs_error_on_exception(self):
        """Test cache_request_handler logs error with traceback when exception occurs."""
        import fastdeploy.cache_manager.multimodal_cache_manager as mm_module

        with patch("fastdeploy.cache_manager.multimodal_cache_manager.zmq") as mock_zmq:
            mock_ctx = MagicMock()
            mock_socket = MagicMock()
            mock_poller = MagicMock()
            mock_zmq.Context.return_value = mock_ctx
            mock_ctx.socket.return_value = mock_socket
            mock_zmq.Poller.return_value = mock_poller
            mock_zmq.ROUTER = 6
            mock_zmq.POLLIN = 1
            mock_zmq.SNDHWM = 23
            mock_zmq.ROUTER_MANDATORY = 33
            mock_zmq.SNDTIMEO = 28

            mock_poller.poll.side_effect = RuntimeError("poll failed")

            manager = mm_module.ProcessorCacheManager.__new__(mm_module.ProcessorCacheManager)
            manager.cache = {}
            manager.current_cache_size = 0
            manager.max_cache_size = 1024
            manager.router = mock_socket
            manager.poller = mock_poller

            with patch.object(mm_module.logger, "error") as mock_error:
                manager.cache_request_handler()

            mock_error.assert_called_once()
            error_msg = mock_error.call_args[0][0]
            self.assertIn("Error happened while handling processor cache request", error_msg)
            self.assertIn("poll failed", error_msg)


if __name__ == "__main__":
    unittest.main()
