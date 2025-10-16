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
import unittest
import uuid
import weakref

from fastdeploy.engine.args_utils import EngineArgs
from fastdeploy.engine.async_llm import AsyncLLMEngine
from fastdeploy.engine.sampling_params import SamplingParams

MODEL_NAME = os.getenv("MODEL_PATH", "/path/to/models") + "/ERNIE-4.5-0.3B-Paddle"


class TestAsyncLLMEngine(unittest.TestCase):
    """Test case for AsyncLLMEngine functionality"""

    PROMPTS = [
        "Hello, my name is",
        "The capital of China is",
        "The future of AI is",
        "人工智能是",
    ]

    @classmethod
    def setUpClass(cls):
        """Set up AsyncLLMEngine for testing"""
        try:
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

            cls.engine = AsyncLLMEngine.from_engine_args(engine_args)
            success = cls.engine.start()

            if not success:
                raise RuntimeError("Failed to start AsyncLLMEngine")

            # Use weak reference to avoid circular reference
            cls.engine_ref = weakref.ref(cls.engine)

        except Exception as e:
            print(f"Setting up AsyncLLMEngine failed: {e}")
            raise unittest.SkipTest(f"AsyncLLMEngine initialization failed: {e}")

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests have run"""
        if hasattr(cls, "engine") and cls.engine is not None:
            try:

                # Force stop the engine first
                cls.engine.running = False

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
            # 清理可能残留的output_handler
            if hasattr(self.engine, "output_handler") and self.engine.output_handler:
                if not self.engine.output_handler.done():
                    print("Cleaning up previous output_handler...")
                    self.engine.output_handler.cancel()
                self.engine.output_handler = None

            # 清理输出处理器的队列
            if hasattr(self.engine, "output_processor") and self.engine.output_processor:
                self.engine.output_processor.request_queues.clear()

            print(f"Test setup completed: {self._testMethodName}")

    def tearDown(self):
        """Clean up after each test method"""
        if hasattr(self, "engine") and self.engine:

            if hasattr(self.engine, "output_handler") and self.engine.output_handler:
                if not self.engine.output_handler.done():
                    print("Cleaning up output_handler after test...")
                    self.engine.output_handler.cancel()
                self.engine.output_handler = None

            if hasattr(self.engine, "output_processor") and self.engine.output_processor:
                self.engine.output_processor.request_queues.clear()

            print(f"Test cleanup completed: {self._testMethodName}")

    def run_async_test(self, coro):
        """Helper method to run async tests"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    def test_engine_initialization(self):
        """Test that the engine initializes correctly"""
        self.assertIsNotNone(self.engine)
        self.assertTrue(self.engine.is_started)
        self.assertTrue(self.engine.running)

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


if __name__ == "__main__":
    unittest.main()
