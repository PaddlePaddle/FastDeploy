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

import multiprocessing as mp
import os
import queue
import unittest
from multiprocessing import Process, Queue

import numpy as np
import paddle

from fastdeploy.model_executor.ops.gpu import set_data_ipc, share_external_data


def _create_test_tensor(shape, dtype):
    if "float" in str(dtype):
        return paddle.rand(shape=shape, dtype=dtype)
    elif "int" in str(dtype):
        return paddle.randint(-100, 100, shape=shape, dtype=dtype)
    elif "bool" in str(dtype):
        return paddle.rand(shape=shape, dtype=dtype) > 0.5


def _producer_proc(shm_name, shape, dtype, ready_q, done_q, error_q):
    # Create shared memory
    try:
        paddle.device.set_device("gpu:0")
        t = _create_test_tensor(shape, dtype)
        set_data_ipc(t, shm_name)
        ready_q.put(("ready", t.numpy().tolist()))
        _ = done_q.get(timeout=20)
    except Exception as e:
        error_q.put(("producer_error", str(e)))


def _consumer_proc(shm_name, shape, dtype, result_q, error_q):
    # Shard data
    try:
        paddle.device.set_device("gpu:0")
        dummy = paddle.zeros(shape, dtype=dtype)
        shared = share_external_data(dummy, shm_name, shape)
        result_q.put(("ok", shared.numpy().tolist()))
    except Exception as e:
        error_q.put(("consumer_error", str(e)))


# Use spawn to avoid forking CUDA contexts
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass


class TestShareExternalData(unittest.TestCase):
    def setUp(self):
        paddle.seed(2024)
        np.random.seed(42)

        if not paddle.device.is_compiled_with_cuda():
            self.skipTest("CUDA not available, skipping GPU tests")

        # Set device to GPU
        paddle.device.set_device("gpu:0")

        self.test_shape = [4, 8]
        self.dtype = paddle.float32
        self.shm_prefix = f"test_share_external_{os.getpid()}"

    def _run_minimal_cross_process(self):
        ready_q = Queue()
        result_q = Queue()
        error_q = Queue()
        done_q = Queue()

        p = Process(
            target=_producer_proc, args=(self.shm_prefix, self.test_shape, self.dtype, ready_q, done_q, error_q)
        )
        p.start()

        # wait producer ready
        try:
            status, original_data = ready_q.get(timeout=20)
            self.assertEqual(status, "ready")
        except Exception:
            p.terminate()
            self.fail("Producer did not become ready in time")

        c = Process(target=_consumer_proc, args=(self.shm_prefix, self.test_shape, self.dtype, result_q, error_q))
        c.start()
        c.join(timeout=30)

        # signal producer to exit now
        done_q.put("done")
        p.join(timeout=30)

        # check errors first (non-blocking)
        errors = []
        try:
            while True:
                errors.append(error_q.get_nowait())
        except queue.Empty:
            pass
        self.assertFalse(errors, f"Errors occurred: {errors}")

        # verify data
        self.assertFalse(result_q.empty(), "No result from consumer")
        status, shared_data = result_q.get()
        self.assertEqual(status, "ok")
        np.testing.assert_allclose(np.array(original_data), np.array(shared_data), rtol=1e-5)

    def test_producer_consumer_processes(self):
        self._run_minimal_cross_process()

    def tearDown(self):
        paddle.device.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
