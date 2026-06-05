"""
# Copyright (c) 2026  PaddlePaddle Authors. All Rights Reserved.
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

import asyncio
import json
import unittest
from unittest.mock import MagicMock, patch


class TestHealthEndpoint(unittest.TestCase):
    """Test /health endpoint."""

    def test_health_returns_200(self):
        """GET /health returns 200."""
        from fastdeploy.entrypoints.api_server import health

        response = asyncio.run(health())
        self.assertEqual(response.status_code, 200)


class TestGenerateEndpointNonStream(unittest.TestCase):
    """Test /generate endpoint in non-stream mode."""

    @patch("fastdeploy.entrypoints.api_server.llm_engine")
    def test_non_stream_returns_result(self, mock_engine):
        """Non-stream mode returns final result."""
        from fastdeploy.entrypoints.api_server import generate

        mock_engine.generate.return_value = iter(
            [
                {"text": "partial"},
                {"text": "Hello, world!"},
            ]
        )

        result = asyncio.run(generate({"prompt": "Hi", "stream": 0}))
        self.assertEqual(result, {"text": "Hello, world!"})

    @patch("fastdeploy.entrypoints.api_server.llm_engine")
    def test_non_stream_default_no_stream_key(self, mock_engine):
        """When 'stream' key is missing, defaults to non-stream (0)."""
        from fastdeploy.entrypoints.api_server import generate

        mock_engine.generate.return_value = iter([{"text": "result"}])

        result = asyncio.run(generate({"prompt": "Hi"}))
        self.assertEqual(result, {"text": "result"})

    @patch("fastdeploy.entrypoints.api_server.llm_engine")
    def test_non_stream_exception_returns_error(self, mock_engine):
        """Non-stream mode returns error dict on exception."""
        from fastdeploy.entrypoints.api_server import generate

        mock_engine.generate.side_effect = ValueError("generation failed")

        result = asyncio.run(generate({"prompt": "Hi", "stream": 0}))
        self.assertEqual(result["error"], "generation failed")
        self.assertEqual(result["error_type"], "ValueError")


class TestGenerateEndpointStream(unittest.TestCase):
    """Test /generate endpoint in stream mode."""

    @patch("fastdeploy.entrypoints.api_server.llm_engine")
    def test_stream_returns_sse_events(self, mock_engine):
        """Stream mode returns StreamingResponse with SSE events."""
        from fastdeploy.entrypoints.api_server import generate

        mock_engine.generate.return_value = iter(
            [
                {"text": "Hello"},
                {"text": "Hello, world!"},
            ]
        )

        response = asyncio.run(generate({"prompt": "Hi", "stream": 1}))

        # StreamingResponse - consume the body_iterator
        async def collect_body():
            chunks = []
            async for chunk in response.body_iterator:
                chunks.append(chunk)
            return "".join(chunks)

        body = asyncio.run(collect_body())
        events = [
            json.loads(line.replace("data: ", "")) for line in body.strip().split("\n\n") if line.startswith("data:")
        ]
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["text"], "Hello")
        self.assertEqual(events[1]["text"], "Hello, world!")

    @patch("fastdeploy.entrypoints.api_server.llm_engine")
    def test_stream_exception_yields_error_event(self, mock_engine):
        """Stream mode yields error event on exception."""
        from fastdeploy.entrypoints.api_server import generate

        def _failing_generator(request, stream):
            yield {"text": "partial"}
            raise RuntimeError("stream error")

        mock_engine.generate.side_effect = _failing_generator

        response = asyncio.run(generate({"prompt": "Hi", "stream": 1}))

        async def collect_body():
            chunks = []
            async for chunk in response.body_iterator:
                chunks.append(chunk)
            return "".join(chunks)

        body = asyncio.run(collect_body())
        events = [
            json.loads(line.replace("data: ", "")) for line in body.strip().split("\n\n") if line.startswith("data:")
        ]

        # Last event should be error
        last_event = events[-1]
        self.assertEqual(last_event["error"], "stream error")
        self.assertEqual(last_event["error_type"], "RuntimeError")


class TestInitApp(unittest.TestCase):
    """Test init_app function."""

    @patch("fastdeploy.entrypoints.api_server.LLMEngine")
    @patch("fastdeploy.entrypoints.api_server.EngineArgs")
    def test_init_app_success(self, mock_engine_args_cls, mock_engine_cls):
        """init_app returns True on successful engine start."""
        import fastdeploy.entrypoints.api_server as module

        mock_args = MagicMock()
        mock_engine_args_cls.from_cli_args.return_value = MagicMock()
        mock_engine = MagicMock()
        mock_engine.start.return_value = True
        mock_engine_cls.from_engine_args.return_value = mock_engine

        result = module.init_app(mock_args)

        self.assertTrue(result)
        self.assertIs(module.llm_engine, mock_engine)

    @patch("fastdeploy.entrypoints.api_server.LLMEngine")
    @patch("fastdeploy.entrypoints.api_server.EngineArgs")
    def test_init_app_engine_start_fails(self, mock_engine_args_cls, mock_engine_cls):
        """init_app returns False when engine.start() fails."""
        import fastdeploy.entrypoints.api_server as module

        mock_args = MagicMock()
        mock_engine_args_cls.from_cli_args.return_value = MagicMock()
        mock_engine = MagicMock()
        mock_engine.start.return_value = False
        mock_engine_cls.from_engine_args.return_value = mock_engine

        result = module.init_app(mock_args)

        self.assertFalse(result)


class TestLaunchApiServer(unittest.TestCase):
    """Test launch_api_server function."""

    @patch("fastdeploy.entrypoints.api_server.uvicorn.run")
    @patch("fastdeploy.entrypoints.api_server.init_app", return_value=True)
    @patch("fastdeploy.entrypoints.api_server.is_port_available", return_value=True)
    def test_launch_success(self, mock_port, mock_init, mock_uvicorn):
        """launch_api_server starts uvicorn when init succeeds."""
        from fastdeploy.entrypoints.api_server import launch_api_server

        args = MagicMock()
        args.host = "0.0.0.0"
        args.port = 9904
        args.workers = 4
        args.__dict__ = {"host": "0.0.0.0", "port": 9904, "workers": 4}

        launch_api_server(args)

        mock_port.assert_called_once_with("0.0.0.0", 9904)
        mock_init.assert_called_once_with(args)
        mock_uvicorn.assert_called_once()

    @patch("fastdeploy.entrypoints.api_server.is_port_available", return_value=False)
    def test_launch_port_in_use_raises(self, mock_port):
        """launch_api_server raises when port is unavailable."""
        from fastdeploy.entrypoints.api_server import launch_api_server

        args = MagicMock()
        args.host = "0.0.0.0"
        args.port = 9904

        with self.assertRaises(Exception) as ctx:
            launch_api_server(args)

        self.assertIn("already in use", str(ctx.exception))

    @patch("fastdeploy.entrypoints.api_server.init_app", return_value=False)
    @patch("fastdeploy.entrypoints.api_server.is_port_available", return_value=True)
    def test_launch_init_fails_returns_early(self, mock_port, mock_init):
        """launch_api_server returns early when init_app fails."""
        from fastdeploy.entrypoints.api_server import launch_api_server

        args = MagicMock()
        args.host = "0.0.0.0"
        args.port = 9904
        args.__dict__ = {"host": "0.0.0.0", "port": 9904}

        with patch("fastdeploy.entrypoints.api_server.uvicorn.run") as mock_uvicorn:
            launch_api_server(args)
            mock_uvicorn.assert_not_called()

    @patch("fastdeploy.entrypoints.api_server.uvicorn.run", side_effect=OSError("bind error"))
    @patch("fastdeploy.entrypoints.api_server.init_app", return_value=True)
    @patch("fastdeploy.entrypoints.api_server.is_port_available", return_value=True)
    def test_launch_uvicorn_exception_handled(self, mock_port, mock_init, mock_uvicorn):
        """launch_api_server handles uvicorn exception."""
        from fastdeploy.entrypoints.api_server import launch_api_server

        args = MagicMock()
        args.host = "0.0.0.0"
        args.port = 9904
        args.workers = 4
        args.__dict__ = {"host": "0.0.0.0", "port": 9904, "workers": 4}

        # Should not raise
        launch_api_server(args)


class TestMain(unittest.TestCase):
    """Test main function."""

    @patch("fastdeploy.entrypoints.api_server.launch_api_server")
    @patch("fastdeploy.entrypoints.api_server.EngineArgs.add_cli_args", side_effect=lambda p: p)
    def test_main_parses_args_and_launches(self, mock_add_args, mock_launch):
        """main() parses arguments and calls launch_api_server."""
        from fastdeploy.entrypoints.api_server import main

        with patch("sys.argv", ["api_server.py", "--port", "8080", "--host", "127.0.0.1"]):
            main()

        mock_launch.assert_called_once()
        args = mock_launch.call_args[0][0]
        self.assertEqual(args.port, 8080)
        self.assertEqual(args.host, "127.0.0.1")
        self.assertEqual(args.workers, 4)


if __name__ == "__main__":
    unittest.main()
