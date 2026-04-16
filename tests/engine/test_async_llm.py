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

import asyncio
import os
import sys
import unittest
import uuid
import weakref

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from e2e.utils.serving_utils import clean_ports

from fastdeploy.engine.args_utils import EngineArgs
from fastdeploy.engine.async_llm import AsyncLLM
from fastdeploy.engine.sampling_params import SamplingParams
from fastdeploy.utils import EngineError

MODEL_NAME = os.getenv("MODEL_PATH", "/path/to/models") + "/ERNIE-4.5-0.3B-Paddle"


class TestAsyncLLMEngine(unittest.TestCase):
    """Test case for AsyncLLM functionality"""

    PROMPTS = [
        "Hello, my name is",
        "The capital of China is",
        "The future of AI is",
        "人工智能是",
    ]

    @classmethod
    def setUpClass(cls):
        """Set up AsyncLLM for testing"""
        try:
            # Clean ports before starting the engine
            print("Pre-test port cleanup...")
            clean_ports()

            # Use unique ports to avoid conflicts
            base_port = int(os.getenv("FD_ENGINE_QUEUE_PORT", "6778"))
            cache_port = int(os.getenv("FD_CACHE_QUEUE_PORT", "6779"))

            engine_args = EngineArgs(
                model=MODEL_NAME,
                max_model_len=8192,
                tensor_parallel_size=1,
                engine_worker_queue_port=base_port,
                cache_queue_port=cache_port,
            )

            # Use base_port as async engine pid to align with ZMQ routing id
            cls.engine = AsyncLLM.from_engine_args(engine_args, pid=base_port)

            cls.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(cls.loop)
            success = cls.loop.run_until_complete(cls.engine.start())

            # Initialize connections after engine service is ready
            cls.loop.run_until_complete(cls.engine.init_connections())

            if not success:
                raise RuntimeError("Failed to start AsyncLLM")

            # Use weak reference to avoid circular reference
            cls.engine_ref = weakref.ref(cls.engine)

        except Exception as e:
            print(f"Setting up AsyncLLM failed: {e}")
            raise

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests have run"""
        if hasattr(cls, "engine") and cls.engine is not None:
            try:

                # Force stop the engine first
                cls.engine.running = False

                # asyncio.run(cls.engine.shutdown())
                cls.loop.run_until_complete(cls.engine.shutdown())

                # Try sync cleanup first
                if hasattr(cls.engine, "_exit_sub_services"):
                    try:
                        cls.engine._exit_sub_services()
                        print("_exit_sub_services completed")
                    except Exception as e:
                        print(f"_exit_sub_services failed: {e}")

                print("Engine cleanup completed")

            except Exception as e:
                print(f"Error during engine cleanup: {e}")
            finally:
                print("Deleting engine...")
                del cls.engine
                print("Engine deleted")

        print("=== tearDownClass completed ===")

        # Force garbage collection
        import gc

        gc.collect()
        print("Garbage collection completed")

    def setUp(self):
        """Set up before each test method"""

        if hasattr(self, "engine") and self.engine:
            print(f"Test setup completed: {self._testMethodName}")

    def tearDown(self):
        """Clean up after each test method"""
        if hasattr(self, "engine") and self.engine:
            print(f"Test cleanup completed: {self._testMethodName}")

    def run_async_test(self, coro):
        """Helper method to run async tests"""

        try:
            return self.loop.run_until_complete(coro)
        finally:
            pass

    def test_engine_initialization(self):
        """Test that the engine initializes correctly"""
        self.assertIsNotNone(self.engine)
        # EngineServiceClient._running indicates underlying engine_service started
        self.assertTrue(self.engine._running)
        self.assertTrue(self.engine.running)

    def test_engine_service_start_exception_logs_and_reraises(self):
        """EngineServiceClient.start should log and re-raise on internal exception"""

        async def _test():
            from unittest.mock import patch

            from fastdeploy.engine.async_llm import EngineServiceClient

            class DummyCfg:
                pass

            client = EngineServiceClient(DummyCfg(), pid=12345)

            # Force _start_engine_process to raise so that start() enters exception block
            with patch.object(client, "_start_engine_process", side_effect=RuntimeError("boom")):
                with self.assertRaises(RuntimeError):
                    await client.start()

            return True

        result = self.run_async_test(_test())
        self.assertTrue(result)

    def test_engine_service_start_process_failure(self):
        """_start_engine_process should log and re-raise on process creation failure"""

        async def _test():
            from unittest.mock import patch

            from fastdeploy.engine.async_llm import EngineServiceClient

            class DummyCfg:
                pass

            client = EngineServiceClient(DummyCfg(), pid=12345)

            # Patch multiprocessing.Process to raise so that exception block is hit
            with patch("multiprocessing.Process", side_effect=RuntimeError("boom")):
                with self.assertRaises(RuntimeError):
                    client._start_engine_process()

            return True

        result = self.run_async_test(_test())
        self.assertTrue(result)

    def test_single_prompt_generation(self):
        """Test generating response for a single prompt"""

        async def _test():
            prompt = "Hello, my name is"
            sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=50)

            outputs = []
            generator = None
            try:
                generator = self.engine.generate(prompt, sampling_params)
                count = 0
                async for output in generator:
                    outputs.append(output)
                    count += 1
                    self.assertIsNotNone(output)
                    self.assertIsNotNone(output.outputs)

            finally:
                # Explicitly close the generator
                if generator is not None:
                    try:
                        await generator.aclose()
                    except:
                        pass

            print(f"Total outputs: {len(outputs)}")
            self.assertGreater(len(outputs), 0)
            return outputs

        outputs = self.run_async_test(_test())
        self.assertGreater(len(outputs), 0)

    def test_multiple_prompts_generation(self):
        """Test generating responses for multiple prompts concurrently"""

        async def _test():
            sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=50)

            # Test concurrent generation
            tasks = []
            for i, prompt in enumerate(self.PROMPTS[:2]):  # Test with first 2 prompts
                request_id = f"test_request_{i}_{uuid.uuid4()}"
                task = self._generate_single(prompt, sampling_params, request_id)
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Check that all tasks completed successfully
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.fail(f"Task {i} failed with exception: {result}")
                self.assertGreater(len(result), 0)
                self.assertTrue(result[-1].finished)

            return results

        results = self.run_async_test(_test())
        self.assertEqual(len(results), 2)

    def test_generation_with_multiple_choices(self):
        """Test generating multiple choices with SamplingParams.n"""

        async def _test():
            # Use dict prompt to cover stream/include_stop_str_in_output flags
            prompt = {
                "prompt": "Hello, my name is",
                "stream": True,
                "include_stop_str_in_output": False,
                "n": 2,
            }
            # Do not set n in SamplingParams so that prompt['n'] takes effect
            sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=20)

            outputs = []
            generator = None
            try:
                generator = self.engine.generate(prompt, sampling_params)
                async for output in generator:
                    outputs.append(output)
            finally:
                if generator is not None:
                    try:
                        await generator.aclose()
                    except Exception:
                        pass

            # Expect at least 2 finished outputs (one per choice)
            finished_outputs = [o for o in outputs if getattr(o, "finished", False)]
            self.assertGreaterEqual(len(finished_outputs), 2)
            return outputs

        outputs = self.run_async_test(_test())
        self.assertGreater(len(outputs), 0)

    async def _generate_single(self, prompt, sampling_params, request_id=None):
        """Helper method to generate response for a single prompt"""
        outputs = []
        generator = None
        try:
            generator = self.engine.generate(prompt, sampling_params, request_id)
            async for output in generator:
                outputs.append(output)
        finally:
            # Explicitly close the generator
            if generator is not None:
                try:
                    await generator.aclose()
                except:
                    pass
        return outputs

    def test_process_output_error_handling(self):
        """Test _process_output error handling"""

        async def _test():
            from unittest.mock import Mock

            from fastdeploy.engine.async_llm import AsyncOutputProcessor

            # Create processor with mock data_processor that raises exception
            mock_data_processor = Mock()
            mock_data_processor.process_response_dict.side_effect = Exception("Decode error")
            processor = AsyncOutputProcessor(mock_data_processor)

            # Create response dict without text field
            response_dict = {
                "request_id": "test",
                "finished": True,
                "outputs": {
                    "index": 0,
                    "send_idx": 0,
                    "token_ids": [1, 2, 3],
                },
                "metrics": {"arrival_time": 0.0},
            }

            # Process the output
            result = processor._process_output(response_dict)

            # Verify text was set to empty string on error
            self.assertIn("outputs", result)
            self.assertEqual(result["outputs"].get("text", ""), "")

            return True

        result = self.run_async_test(_test())
        self.assertTrue(result)

    def test_process_output_processor_returns_none(self):
        """Test _process_output when data_processor returns None"""

        async def _test():
            from unittest.mock import Mock

            from fastdeploy.engine.async_llm import AsyncOutputProcessor

            # Create processor with mock data_processor that returns None
            mock_data_processor = Mock()
            mock_data_processor.process_response_dict.return_value = None
            processor = AsyncOutputProcessor(mock_data_processor)

            # Create response dict without text field
            response_dict = {
                "request_id": "test",
                "finished": True,
                "outputs": {
                    "index": 0,
                    "send_idx": 0,
                    "token_ids": [1, 2, 3],
                },
                "metrics": {"arrival_time": 0.0},
            }

            # Process the output
            result = processor._process_output(response_dict)

            # Verify text was set to empty string when processor returns None
            self.assertIn("outputs", result)
            self.assertEqual(result["outputs"].get("text", ""), "")

            return True

        result = self.run_async_test(_test())
        self.assertTrue(result)

    def test_engine_abort_request(self):
        """Test AsyncLLM abort_request functionality"""

        async def _test():
            # Test calling abort_request directly without mocking
            request_id = "test_abort_request"

            # This should not raise an exception
            await self.engine.abort_request(request_id)

            return True

        result = self.run_async_test(_test())
        self.assertTrue(result)

    def test_engine_abort_request_with_cleanup_error(self):
        """abort_request should handle cleanup_request exceptions gracefully"""

        async def _test():
            from unittest.mock import AsyncMock, patch

            mock_cm = AsyncMock()
            mock_cm.cleanup_request.side_effect = Exception("cleanup failed")
            mock_cm.running = True

            with patch.object(self.engine, "connection_manager", mock_cm):
                # Should not raise even if cleanup_request fails
                await self.engine.abort_request("test_abort_error")

            return True

        result = self.run_async_test(_test())
        self.assertTrue(result)

    def test_generate_with_exception_abort(self):
        """Test that generate handles exceptions properly"""

        async def _test():
            # Test with invalid prompt type
            try:
                generator = self.engine.generate(123, SamplingParams(max_tokens=10))  # Invalid prompt type
                async for _ in generator:
                    pass
            except Exception:
                # This is expected
                pass

            return True

        result = self.run_async_test(_test())
        self.assertTrue(result)

    def test_request_validation_errors(self):
        """Test request validation error scenarios"""

        async def _test():
            # Test input length validation (lines 438-443, 446-448)
            try:
                prompts = [0, 1, 2]
                # Create sampling params with very high min_tokens to trigger error
                sampling_params = SamplingParams(min_tokens=999999, n=1)

                # This should trigger the min_tokens validation error
                await self.engine.add_request("test_validation", prompts, sampling_params)
            except Exception as e:
                # Expected to fail due to validation
                self.assertIn("min_dec_len", str(e).lower())

            # Test max model len validation
            try:
                # Create a very long prompt to trigger max_model_len error
                long_prompts = {"prompt_token_ids": [1] * 3000, "prompt_token_ids_len": 3000}  # 超过max_model_len
                await self.engine.add_request("test_long", long_prompts)
            except EngineError as e:
                # 根据实际错误消息调整断言
                error_msg = str(e).lower()
                self.assertTrue(
                    "exceeds the limit" in error_msg
                    or "input text is too long" in error_msg
                    or "input_ids_len" in error_msg
                )
            except Exception:
                # Expected to fail due to length validation
                pass

            return True

        result = self.run_async_test(_test())
        self.assertTrue(result)

    def test_get_methods_coverage(self):
        """Test get_model_config and get_tokenizer methods"""

        async def _test():
            # Test get_model_config (lines 326-328)
            model_config = await self.engine.get_model_config()
            self.assertIsNotNone(model_config)

            # Test get_tokenizer (lines 330-334)
            tokenizer = await self.engine.get_tokenizer()
            if hasattr(self.engine, "data_processor"):
                # This should hit line 333: return self.data_processor.tokenizer
                self.assertIsNotNone(tokenizer)

            # Test _has_guided_input method
            from unittest.mock import Mock

            # Test with guided input
            request_with_guided = Mock()
            request_with_guided.guided_json = {"type": "object"}
            request_with_guided.guided_regex = None
            request_with_guided.guided_choice = None
            request_with_guided.structural_tag = None
            request_with_guided.guided_grammar = None
            request_with_guided.guided_json_object = None

            result = self.engine._has_guided_input(request_with_guided)
            self.assertTrue(result)

            return True

        result = self.run_async_test(_test())
        self.assertTrue(result)

    def test_generate_engine_not_started(self):
        """Test add_request and generate method when engine is not started"""

        async def _test():
            # Create a new engine instance without starting it
            engine_args = EngineArgs(
                model=MODEL_NAME,
                max_model_len=8192,
                tensor_parallel_size=1,
                engine_worker_queue_port=int(os.getenv("FD_ENGINE_QUEUE_PORT", "6778")) + 2,
                cache_queue_port=int(os.getenv("FD_CACHE_QUEUE_PORT", "6779")) + 2,
            )

            async_pid = int(os.getenv("FD_ENGINE_QUEUE_PORT", "6778")) + 2
            unstarted_engine = AsyncLLM.from_engine_args(engine_args, pid=async_pid)
            # Don't call start() or init_connections() - engine is not fully initialized

            # Test add_request method when engine is not fully initialized
            try:
                sampling_params = SamplingParams(max_tokens=10)
                await unstarted_engine.add_request("test_request", "Test prompt", sampling_params)
                self.fail("Expected EngineError was not raised in add_request")
            except EngineError as e:
                # Uninitialized engine should wrap error from add_request with error_code 400
                self.assertEqual(e.error_code, 400)
                self.assertIn("async_llm add request failed", str(e))
            except Exception as e:
                self.fail(f"Unexpected exception type in add_request: {type(e).__name__}: {e}")

            # Test generate method when engine is not fully initialized (ZMQ not connected)
            try:
                sampling_params = SamplingParams(max_tokens=10)
                generator = unstarted_engine.generate("Test prompt", sampling_params)
                async for _ in generator:
                    pass
                self.fail("Expected EngineError was not raised in generate")
            except EngineError as e:
                # Generate should fail fast with initialization error
                self.assertEqual(e.error_code, 500)
                self.assertIn("init_connections", str(e))
            except Exception as e:
                self.fail(f"Unexpected exception type in generate: {type(e).__name__}: {e}")

            return True

        result = self.run_async_test(_test())
        self.assertTrue(result)

    def test_zmq_connection_initialization_failure(self):
        """Test ZMQ connection initialization failure"""

        async def _test():
            from unittest.mock import Mock, patch

            # Create a new engine instance
            engine_args = EngineArgs(
                model=MODEL_NAME,
                max_model_len=8192,
                tensor_parallel_size=1,
                engine_worker_queue_port=int(os.getenv("FD_ENGINE_QUEUE_PORT", "6778")) + 4,
                cache_queue_port=int(os.getenv("FD_CACHE_QUEUE_PORT", "6779")) + 4,
            )

            async_pid = int(os.getenv("FD_ENGINE_QUEUE_PORT", "6778")) + 4
            test_engine = AsyncLLM.from_engine_args(engine_args, pid=async_pid)

            # Test connection manager initialization failure
            with (
                patch("fastdeploy.engine.async_llm.ZmqIpcClient") as mock_client_class,
                patch("fastdeploy.engine.async_llm.DealerConnectionManager") as mock_manager_class,
            ):

                # Mock successful client creation
                mock_client = Mock()
                mock_client_class.return_value = mock_client

                # Mock DealerConnectionManager to fail on initialize
                mock_manager = Mock()
                mock_manager.running = False
                mock_manager.initialize.side_effect = Exception("Failed to initialize connection manager")
                mock_manager_class.return_value = mock_manager

                try:
                    await test_engine.init_connections()
                    self.fail("Expected exception was not raised")
                except Exception as e:
                    self.assertIn("Failed to initialize connection manager", str(e))

            return True

        result = self.run_async_test(_test())
        self.assertTrue(result)

    def test_add_request_exception_handling(self):
        """Test add_request exception handling (lines 447-448 in async_llm.py)"""

        async def _test():
            from unittest.mock import patch

            # Mock data_processor to raise exception
            with patch.object(self.engine, "data_processor") as mock_processor:
                mock_processor.process_request_dict.side_effect = RuntimeError("Processing failed")

                try:
                    await self.engine.add_request("test_id", "test prompt", SamplingParams(max_tokens=10))
                    self.fail("Expected EngineError was not raised")
                except EngineError as e:
                    self.assertEqual(e.error_code, 400)
                    self.assertIn("async_llm add request failed", str(e))
                    self.assertIn("Processing failed", str(e))

            return True

        result = self.run_async_test(_test())
        self.assertTrue(result)

    def test_generate_generator_exit_handled(self):
        """Test generate handles GeneratorExit from response queue gracefully"""

        async def _test():
            from unittest.mock import AsyncMock, patch

            # Ensure engine has a valid request_client and connection_manager.running
            self.assertIsNotNone(self.engine.request_client)
            self.assertIsNotNone(self.engine.connection_manager)

            # Mock connection_manager to simulate GeneratorExit from response_queue.get()
            mock_connection_manager = AsyncMock()
            mock_queue = AsyncMock()
            mock_queue.get.side_effect = GeneratorExit("Generator closed")
            mock_connection_manager.get_connection.return_value = (AsyncMock(), mock_queue)
            mock_connection_manager.running = True
            mock_connection_manager.worker_pid = os.getpid()

            with patch.object(self.engine, "connection_manager", mock_connection_manager):
                generator = self.engine.generate("test", SamplingParams(max_tokens=10))

                # generate should swallow GeneratorExit and not propagate it to caller
                try:
                    async for _ in generator:
                        pass
                except GeneratorExit:
                    self.fail("GeneratorExit should be handled inside generate")
                except Exception as e:
                    self.fail(f"Unexpected exception: {e}")

            return True

        result = self.run_async_test(_test())
        self.assertTrue(result)

    def test_generate_cleanup_request_error_handled(self):
        """generate should swallow cleanup_request errors in finally block"""

        async def _test():
            from unittest.mock import AsyncMock, patch

            from fastdeploy.engine.request import (
                CompletionOutput,
                RequestMetrics,
                RequestOutput,
            )

            # Build a minimal RequestOutput dict that generate() can consume
            metrics = RequestMetrics(arrival_time=0.0)
            completion = CompletionOutput(index=0, send_idx=0, token_ids=[], text="")
            ro = RequestOutput(request_id="cmpl-test_0", outputs=completion, finished=True, metrics=metrics)
            ro_dict = ro.to_dict()

            engine = self.engine

            # Mock connection_manager and response queue
            mock_queue = AsyncMock()
            mock_queue.get.return_value = [ro_dict]
            mock_dealer = AsyncMock()
            mock_cm = AsyncMock()
            mock_cm.get_connection.return_value = (mock_dealer, mock_queue)
            mock_cm.running = True
            # Force cleanup_request to raise so we hit the except/pass branch
            mock_cm.cleanup_request.side_effect = Exception("cleanup error")

            # Stub add_request to avoid touching real ZMQ or data_processor
            async def fake_add_request(*args, **kwargs):
                return None

            # Simple output processor that returns the dict unchanged
            class DummyOutputProcessor:
                def _process_output(self, response_dict, **kwargs):
                    return response_dict

            with (
                patch.object(engine, "connection_manager", mock_cm),
                patch.object(engine, "add_request", side_effect=fake_add_request),
                patch.object(engine, "request_client", object()),
                patch.object(engine, "output_processor", DummyOutputProcessor()),
            ):
                outputs = []
                async for out in engine.generate("test", SamplingParams(max_tokens=5)):
                    outputs.append(out)

                # We should get exactly one finished output and no exception
                self.assertEqual(len(outputs), 1)
                self.assertTrue(outputs[0].finished)

            return True

        result = self.run_async_test(_test())
        self.assertTrue(result)

    def test_shutdown_exception_handling(self):
        """Test shutdown method exception handling"""

        async def _test():
            from unittest.mock import Mock, patch

            # Create test engine
            engine_args = EngineArgs(
                model=MODEL_NAME,
                max_model_len=8192,
                tensor_parallel_size=1,
                engine_worker_queue_port=int(os.getenv("FD_ENGINE_QUEUE_PORT", "6778")) + 6,
                cache_queue_port=int(os.getenv("FD_CACHE_QUEUE_PORT", "6779")) + 6,
            )
            async_pid = int(os.getenv("FD_ENGINE_QUEUE_PORT", "6778")) + 6
            test_engine = AsyncLLM.from_engine_args(engine_args, pid=async_pid)

            # Mock components that raise exceptions during shutdown
            test_engine.connection_manager = Mock()
            test_engine.connection_manager.close.side_effect = Exception("Connection manager close failed")

            test_engine.request_client = Mock()
            test_engine.request_client.close.side_effect = Exception("Request client close failed")

            # Patch EngineServiceClient.shutdown to raise as well so we hit
            # the exception handling path in AsyncLLM.shutdown (lines 566-567)
            with patch("fastdeploy.engine.async_llm.EngineServiceClient.shutdown", side_effect=Exception("boom")):
                # Test that shutdown handles all exceptions gracefully
                try:
                    await test_engine.shutdown()
                    # Should not raise exception despite internal failures
                except Exception as e:
                    self.fail(f"Shutdown should handle exceptions gracefully: {e}")

            return True

        result = self.run_async_test(_test())
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
