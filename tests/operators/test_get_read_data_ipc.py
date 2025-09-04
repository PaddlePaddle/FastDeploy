# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import multiprocessing as mp
import os
import queue
import unittest
from multiprocessing import Process, Queue

import numpy as np
import paddle

from fastdeploy.model_executor.ops.gpu import (
    get_data_ptr_ipc,
    read_data_ipc,
    set_data_ipc,
)


def _create_test_tensor(shape, dtype):
    # Create GPU tensor with deterministic data type
    paddle.device.set_device("gpu:0")
    return paddle.rand(shape=shape, dtype=dtype)


def _producer_set_ipc(shm_name, shape, dtype, ready_q, done_q, error_q):
    try:
        paddle.device.set_device("gpu:0")
        t = _create_test_tensor(shape, dtype)
        set_data_ipc(t, shm_name)
        ready_q.put(("ready", True))
        _ = done_q.get(timeout=20)
    except Exception as e:
        error_q.put(("producer_error", str(e)))


def _consumer_get_and_read(shm_name, shape, dtype, result_q, error_q):
    try:
        paddle.device.set_device("gpu:0")
        ptr_tensor = get_data_ptr_ipc(paddle.zeros([1], dtype=paddle.float32), shm_name)
        ptr_val = int(ptr_tensor.numpy()[0])
        if ptr_val == 0:
            raise RuntimeError("get_data_ptr_ipc returned null pointer")

        # Note(ooooo): Because it can print the ptr of shm, make it to check in stdout with `hex(ptr_val)`.
        read_data_ipc(paddle.zeros([1], dtype=paddle.float32), ptr_val, shm_name)

        result_q.put(("ok", ptr_val))
    except Exception as e:
        error_q.put(("consumer_error", str(e)))


# Ensure spawn to avoid inheriting CUDA context
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass


class TestGetReadDataIPC(unittest.TestCase):
    def setUp(self):
        paddle.seed(2024)
        np.random.seed(42)
        if not paddle.device.is_compiled_with_cuda():
            self.skipTest("CUDA not available, skipping GPU tests")
        paddle.device.set_device("gpu:0")

        # ensure >= 10 elems since read_data_ipc prints first 10
        self.shape = [4, 8]
        self.dtype = paddle.float32
        self.shm_name = f"test_get_read_ipc_{os.getpid()}"

    def test_get_then_read_ipc_cross_process(self):
        ready_q, result_q, error_q, done_q = Queue(), Queue(), Queue(), Queue()

        # producer: set IPC and hold until done
        p = Process(target=_producer_set_ipc, args=(self.shm_name, self.shape, self.dtype, ready_q, done_q, error_q))
        p.start()

        try:
            status, _ = ready_q.get(timeout=20)
            self.assertEqual(status, "ready")
        except Exception:
            p.terminate()
            self.fail("Producer did not become ready in time")

        # consumer: get ptr and invoke read
        c = Process(target=_consumer_get_and_read, args=(self.shm_name, self.shape, self.dtype, result_q, error_q))
        c.start()
        c.join(timeout=30)

        # let producer exit
        done_q.put("done")
        p.join(timeout=30)

        # collect errors
        errors = []
        try:
            while True:
                errors.append(error_q.get_nowait())
        except queue.Empty:
            pass
        self.assertFalse(errors, f"Errors occurred: {errors}")

        # result check
        self.assertFalse(result_q.empty(), "No result from consumer")
        status, ptr_val = result_q.get()
        self.assertEqual(status, "ok")
        self.assertTrue(isinstance(ptr_val, int) and ptr_val != 0)
        print(hex(ptr_val))


if __name__ == "__main__":
    unittest.main()
