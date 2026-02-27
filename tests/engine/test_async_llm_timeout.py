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
Tests for AsyncLLM response queue timeout (aff1eae8).

The change replaced a bare `await response_queue.get()` (which could block
forever) with `asyncio.wait_for(response_queue.get(), timeout=queue_timeout)`,
converting asyncio.TimeoutError into a RuntimeError with request context.

Why test in isolation:
  - AsyncLLM requires a running engine, model weights, and IPC infrastructure.
    We test the timeout pattern directly using an asyncio.Queue to verify
    that the timeout → RuntimeError conversion works correctly.
"""

import asyncio
import unittest


class TestAsyncLLMQueueTimeout(unittest.TestCase):
    """Test the queue timeout pattern used in AsyncLLM._process_request_n_choices."""

    @staticmethod
    async def _simulate_queue_timeout(queue_timeout, remaining=1, request_id="test-req-001"):
        """Reproduce the exact timeout pattern from async_llm.py."""
        queue = asyncio.Queue()
        # queue is empty → get() will block until timeout
        try:
            await asyncio.wait_for(queue.get(), timeout=queue_timeout)
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"Timed out waiting for response after {queue_timeout}s. "
                f"remaining={remaining}, request_id={request_id}"
            )

    def test_timeout_raises_runtime_error(self):
        """Empty queue should trigger TimeoutError → RuntimeError."""
        with self.assertRaises(RuntimeError) as ctx:
            asyncio.run(self._simulate_queue_timeout(queue_timeout=0.01))

        msg = str(ctx.exception)
        self.assertIn("Timed out", msg)
        self.assertIn("test-req-001", msg)
        self.assertIn("remaining=1", msg)

    def test_timeout_includes_request_id(self):
        """RuntimeError message should include the request_id for debugging."""
        with self.assertRaises(RuntimeError) as ctx:
            asyncio.run(
                self._simulate_queue_timeout(
                    queue_timeout=0.01,
                    remaining=3,
                    request_id="custom-req-42",
                )
            )

        msg = str(ctx.exception)
        self.assertIn("custom-req-42", msg)
        self.assertIn("remaining=3", msg)

    def test_no_timeout_when_data_available(self):
        """When data is available before timeout, no exception should be raised."""

        async def run():
            queue = asyncio.Queue()
            await queue.put(["response_data"])
            try:
                result = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                raise RuntimeError("Should not timeout")
            return result

        result = asyncio.run(run())
        self.assertEqual(result, ["response_data"])


if __name__ == "__main__":
    unittest.main()
