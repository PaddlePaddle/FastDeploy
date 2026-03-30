"""
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

import ctypes
import json
import logging
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import paddle

import fastdeploy.eplb.async_expert_loader as _ael_mod
from fastdeploy.config import EPLBConfig
from fastdeploy.eplb.async_expert_loader import (
    AsyncEPLoader,
    create_mmap,
    load_ep_checkpoint,
    load_model_weights_process,
    load_tensor_from_shm_mem,
    save_tensor_to_shm_mem,
)

_logger = logging.getLogger("test_eplb")
_GC_GUARD = []


def _shm_buffer(data_bytes):
    """Create a ctypes pointer from raw bytes for shared memory tests."""
    buf = (ctypes.c_byte * len(data_bytes))(*data_bytes)
    _GC_GUARD.append(buf)
    return ctypes.cast(buf, ctypes.POINTER(ctypes.c_int8))


def _eplb_config(**overrides):
    defaults = {
        "redundant_expert_async_load_model_shmem_size_gb": 1,
        "model_use_safetensors": False,
        "moe_quant_type": "",
    }
    defaults.update(overrides)
    return EPLBConfig(defaults)


class _StubSafeFile:
    """Safetensors file context-manager stub with real tensors."""

    def __init__(self, tensors):
        self._tensors = tensors

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def keys(self):
        return list(self._tensors)

    def get_tensor(self, name):
        return self._tensors[name]


class _CudaErr:
    cudaSuccess = 0
    cudaErrorInvalidValue = 1


class _StubCudart:
    """Cudart stub — CUDA not available in CPU-only CI."""

    cudaError_t = _CudaErr

    def __init__(self, ok=True):
        self._ret = _CudaErr.cudaSuccess if ok else _CudaErr.cudaErrorInvalidValue

    def cudaHostRegister(self, addr, size, flags):
        return (self._ret,)

    def cudaGetErrorString(self, err):
        return (_CudaErr.cudaSuccess, b"err")


class _StubLibc:
    def __init__(self, mmap_ret=-1):
        self._ret = mmap_ret

    def mmap(self, *a):
        return self._ret


class _StubPtr:
    contents = None


class _DummyFileCtx:
    def close(self):
        pass


class _StubConn:
    """Multiprocessing Connection stub — records sent data."""

    def __init__(self, messages=None):
        self._msgs = list(messages or [])
        self._i = 0
        self.sent = []

    def recv(self):
        if self._i >= len(self._msgs):
            raise KeyboardInterrupt
        msg = self._msgs[self._i]
        self._i += 1
        return msg

    def send(self, data):
        self.sent.append(data)


class TestAsyncExpertLoader(unittest.TestCase):
    """Test cases for async_expert_loader.py"""

    def setUp(self):
        paddle.set_device("cpu")
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def _make_loader(self, safetensors=False, **kw):
        cfg = _eplb_config(model_use_safetensors=safetensors)
        defaults = dict(
            model_dir=self.temp_dir,
            eplb_config=cfg,
            rank=0,
            expert_per_rank=2,
            moe_layer_start_index=1,
            moe_quant_type="",
            logger=_logger,
        )
        defaults.update(kw)
        return AsyncEPLoader(**defaults)

    # -- save/load shared memory --

    def test_save_tensor_to_shm_mem(self):
        """save_tensor_to_shm_mem: single + multiple tensors with offsets."""
        fp = os.path.join(self.temp_dir, "shm")
        with open(fp, "wb") as f:
            f.write(b"\x00" * 8192)
        t1 = paddle.ones([4], dtype="float32")
        t2 = paddle.zeros([8], dtype="float32")
        infos = save_tensor_to_shm_mem([("w1", t1), ("w2", t2)], fp, logger=_logger)
        self.assertEqual(infos[0][:3], ("w1", 0, 16))
        self.assertEqual(infos[1][1], 16)

    def test_save_tensor_errors(self):
        """save_tensor_to_shm_mem: file not exist + overflow."""
        with self.assertRaises(OSError):
            save_tensor_to_shm_mem([], "/nonexistent/path")
        fp = os.path.join(self.temp_dir, "tiny")
        with open(fp, "wb") as f:
            f.write(b"\x00" * 4)
        with self.assertRaises(IOError):
            save_tensor_to_shm_mem([("big", paddle.ones([100], dtype="float32"))], fp)

    def test_load_tensor_numeric_dtypes(self):
        """load_tensor_from_shm_mem: float32, uint8, int8, int32."""
        cases = [
            (np.float32, paddle.float32, [1.0, 2.0, 3.0]),
            (np.uint8, paddle.uint8, [0, 128, 255]),
            (np.int8, paddle.int8, [-1, 0, 127]),
            (np.int32, paddle.int32, [10, 20, 30]),
        ]
        for np_dtype, pd_dtype, vals in cases:
            with self.subTest(dtype=str(pd_dtype)):
                arr = np.array(vals, dtype=np_dtype)
                raw = arr.tobytes()
                result = load_tensor_from_shm_mem([("w", 0, len(raw), [len(vals)], pd_dtype)], _shm_buffer(raw))
                np.testing.assert_array_equal(result[0][1].numpy(), arr)

    def test_load_tensor_special_dtypes(self):
        """load_tensor_from_shm_mem: bfloat16, float8_e4m3fn, unsupported."""
        arr16 = np.array([0x3F80, 0x4000], dtype=np.uint16)
        result = load_tensor_from_shm_mem(
            [("w", 0, len(arr16.tobytes()), [2], paddle.bfloat16)],
            _shm_buffer(arr16.tobytes()),
            logger=_logger,
        )
        self.assertEqual(list(result[0][1].shape), [2])

        arr8 = np.array([0x38, 0x40], dtype=np.uint8)
        result2 = load_tensor_from_shm_mem(
            [("w", 0, len(arr8.tobytes()), [2], paddle.float8_e4m3fn)],
            _shm_buffer(arr8.tobytes()),
        )
        self.assertEqual(list(result2[0][1].shape), [2])

        with self.assertRaises(TypeError):
            load_tensor_from_shm_mem([("w", 0, 8, [2], paddle.complex64)], _shm_buffer(b"\x00" * 8))

    # -- load_ep_checkpoint --

    def test_load_ep_checkpoint(self):
        """load_ep_checkpoint: missing dir returns empty; valid index parsed."""
        self.assertEqual(load_ep_checkpoint("/nonexistent"), {})
        data = {"weight_map": {"a": "s1.safetensors", "b": "s2.safetensors"}}
        with open(os.path.join(self.temp_dir, "model.safetensors.index.json"), "w") as f:
            json.dump(data, f)
        self.assertEqual(len(load_ep_checkpoint(self.temp_dir)), 2)

    # -- AsyncEPLoader --

    def test_init_and_reset(self):
        """AsyncEPLoader: constructor sets fields; reset clears them."""
        loader = self._make_loader()
        self.assertEqual(loader.model_path, self.temp_dir)
        loader.old_model_ep_rank_to_expert_id_list = np.array([[1, 2]])
        loader.cached_weights = [("x", "y")]
        loader.reset()
        self.assertIsNone(loader.old_model_ep_rank_to_expert_id_list)
        self.assertEqual(loader.cached_weights, [])

    def test_load_experts_weight_bf16(self):
        """load_experts_weight_from_disk: bf16 path with real logic."""
        loader = self._make_loader()
        loader.old_model_ep_rank_to_expert_id_list = np.array([[0, 1], [0, 1]])
        loader.new_model_ep_rank_to_expert_id_list = np.array([[0, 1], [2, 3]])
        ok, _ = loader.load_experts_weight_from_disk()
        self.assertTrue(ok)

    def test_load_experts_weight_safetensors(self):
        """load_experts_weight_from_disk: safetensors routing."""
        loader = self._make_loader(safetensors=True)
        loader.old_model_ep_rank_to_expert_id_list = np.array([[0, 1], [0, 1]])
        loader.new_model_ep_rank_to_expert_id_list = np.array([[0, 1], [2, 3]])
        with patch.object(loader, "load_safetensor_fp8_from_disk", return_value=(True, "ok")):
            ok, _ = loader.load_experts_weight_from_disk()
        self.assertTrue(ok)

    def test_load_experts_weight_failure(self):
        """load_experts_weight_from_disk: failure from inner loader."""
        loader = self._make_loader()
        loader.old_model_ep_rank_to_expert_id_list = np.array([[0, 1], [0, 1]])
        loader.new_model_ep_rank_to_expert_id_list = np.array([[0, 1], [2, 3]])
        with patch.object(loader, "load_weight_bf16_from_disk", return_value=(False, "err")):
            ok, msg = loader.load_experts_weight_from_disk()
        self.assertFalse(ok)

    def test_load_experts_weight_mismatch(self):
        """load_experts_weight_from_disk: mismatched expert id lengths."""
        loader = self._make_loader(moe_layer_start_index=0, expert_per_rank=3)
        loader.old_model_ep_rank_to_expert_id_list = np.array([[0, 1]], dtype=object)
        loader.new_model_ep_rank_to_expert_id_list = np.array([[0, 1, 2]], dtype=object)
        ok, msg = loader.load_experts_weight_from_disk()
        self.assertFalse(ok)
        self.assertIn("length not equal", msg)

    def test_load_experts_weight_exception(self):
        """load_experts_weight_from_disk: exception from None old list."""
        loader = self._make_loader()
        loader.old_model_ep_rank_to_expert_id_list = None
        loader.new_model_ep_rank_to_expert_id_list = np.array([[0, 1]])
        ok, msg = loader.load_experts_weight_from_disk()
        self.assertFalse(ok)
        self.assertIn("Failed to load_experts_weight_from_disk", msg)

    def test_load_weight_bf16_from_disk(self):
        """load_weight_bf16_from_disk: records file names with real logic."""
        loader = self._make_loader(expert_per_rank=8, moe_layer_start_index=3)
        ok, _ = loader.load_weight_bf16_from_disk([(3, 0), (4, 1)])
        self.assertTrue(ok)
        self.assertEqual(len(loader.moe_file_names), 4)

    def test_load_weight_bf16_exception(self):
        """load_weight_bf16_from_disk: bad input triggers exception path."""
        loader = self._make_loader()
        ok, msg = loader.load_weight_bf16_from_disk(None)
        self.assertFalse(ok)

    def test_load_safetensor_fp8(self):
        """load_safetensor_fp8_from_disk: loads with stub safetensors."""
        loader = self._make_loader(safetensors=True, expert_per_rank=8, moe_layer_start_index=3)
        names, fake_map = [], {}
        for proj in ["up_gate_proj", "down_proj"]:
            for quant in ["quant_weight", "weight_scale"]:
                n = f"ernie.layers.3.mlp.experts.0.{proj}.{quant}"
                fake_map[n] = os.path.join(self.temp_dir, "shard.safetensors")
                names.append(n)
        tensors = {n: paddle.ones([4], dtype="float32") for n in names}
        with (
            patch.object(_ael_mod, "load_ep_checkpoint", return_value=fake_map),
            patch("safetensors.safe_open", return_value=_StubSafeFile(tensors)),
        ):
            ok, _ = loader.load_safetensor_fp8_from_disk([(3, 0)])
        self.assertTrue(ok)
        self.assertEqual(len(loader.cached_weights), 4)

    # -- create_mmap (requires OS/CUDA stubs — cannot run real mmap in CI) --

    def test_create_mmap_mmap_failure(self):
        """create_mmap: mmap returns MAP_FAILED → OSError."""
        with (
            patch.object(_ael_mod, "cudart", _StubCudart()),
            patch.object(_ael_mod, "libc", _StubLibc(mmap_ret=-1)),
            patch.object(os.path, "isfile", return_value=True),
            patch.object(os, "open", return_value=5),
            patch.object(os, "ftruncate"),
        ):
            with self.assertRaises(OSError):
                create_mmap(["m"], 0, 1, "u", _eplb_config(redundant_expert_async_load_model_shmem_size_gb=0))

    def test_create_mmap_no_cudart(self):
        """create_mmap: cudart=None → ImportError."""
        with (
            patch.object(_ael_mod, "cudart", None),
            patch.object(_ael_mod, "libc", _StubLibc(mmap_ret=12345)),
            patch.object(os.path, "isfile", return_value=False),
            patch("builtins.open", return_value=_DummyFileCtx()),
            patch.object(os, "open", return_value=5),
            patch.object(os, "ftruncate"),
        ):
            with self.assertRaises(ImportError):
                create_mmap(["m"], 0, 1, "u", _eplb_config())

    def test_create_mmap_cuda_register_fail(self):
        """create_mmap: cudaHostRegister failure → RuntimeError."""
        with (
            patch.object(_ael_mod, "cudart", _StubCudart(ok=False)),
            patch.object(_ael_mod, "libc", _StubLibc(mmap_ret=12345)),
            patch.object(os.path, "isfile", return_value=False),
            patch("builtins.open", return_value=_DummyFileCtx()),
            patch.object(os, "open", return_value=5),
            patch.object(os, "ftruncate"),
            patch.object(ctypes, "cast", return_value=_StubPtr()),
            patch.object(ctypes, "addressof", return_value=0x1000),
        ):
            with self.assertRaises(RuntimeError):
                create_mmap(["m"], 0, 1, "u", _eplb_config())

    def test_create_mmap_success(self):
        """create_mmap: full success path."""
        with (
            patch.object(_ael_mod, "cudart", _StubCudart()),
            patch.object(_ael_mod, "libc", _StubLibc(mmap_ret=12345)),
            patch.object(os.path, "isfile", return_value=False),
            patch("builtins.open", return_value=_DummyFileCtx()),
            patch.object(os, "open", return_value=5),
            patch.object(os, "ftruncate"),
            patch.object(ctypes, "cast", return_value=_StubPtr()),
            patch.object(ctypes, "addressof", return_value=0x1000),
        ):
            result = create_mmap(["m"], 0, 1, "u", _eplb_config(), _logger)
        self.assertIn("m", result)

    # -- load_model_weights_process --

    def _run_process(self, disk_ok=True, raise_exc=False):
        """Helper: run load_model_weights_process with connection stubs."""
        mg = _StubConn(
            [
                {
                    "old_model_ep_rank_to_expert_id_list": np.array([[0, 1]]),
                    "new_model_ep_rank_to_expert_id_list": np.array([[0, 1]]),
                }
            ]
        )
        data = _StubConn()
        with (
            patch("setproctitle.setproctitle"),
            patch("faulthandler.enable"),
            patch("fastdeploy.utils.get_logger", return_value=_logger),
            patch.object(paddle, "set_device"),
        ):
            if raise_exc:

                def _boom(self_inner):
                    raise RuntimeError("load boom")

                with patch.object(AsyncEPLoader, "load_experts_weight_from_disk", _boom):
                    try:
                        load_model_weights_process(0, self.temp_dir, 8, 3, "", "uuid", _eplb_config(), data, mg)
                    except KeyboardInterrupt:
                        pass
            else:
                with patch.object(
                    AsyncEPLoader,
                    "load_experts_weight_from_disk",
                    return_value=(disk_ok, "ok" if disk_ok else "fail"),
                ):
                    if disk_ok:
                        with patch.object(
                            _ael_mod,
                            "save_tensor_to_shm_mem",
                            return_value=[("w", 0, 4, [1], paddle.float32)],
                        ):
                            try:
                                load_model_weights_process(
                                    0, self.temp_dir, 8, 3, "", "uuid", _eplb_config(), data, mg
                                )
                            except KeyboardInterrupt:
                                pass
                    else:
                        try:
                            load_model_weights_process(0, self.temp_dir, 8, 3, "", "uuid", _eplb_config(), data, mg)
                        except KeyboardInterrupt:
                            pass
        return data

    def test_process_success(self):
        """load_model_weights_process: success path sends result=True."""
        data = self._run_process(disk_ok=True)
        self.assertEqual(len(data.sent), 1)
        self.assertTrue(data.sent[0]["result"])

    def test_process_failure(self):
        """load_model_weights_process: disk load failure sends result=False."""
        data = self._run_process(disk_ok=False)
        self.assertEqual(len(data.sent), 1)
        self.assertFalse(data.sent[0]["result"])

    def test_process_exception(self):
        """load_model_weights_process: exception sends result=False, empty weights."""
        data = self._run_process(raise_exc=True)
        self.assertEqual(len(data.sent), 1)
        self.assertFalse(data.sent[0]["result"])
        self.assertEqual(data.sent[0]["weights"], [])


if __name__ == "__main__":
    unittest.main()
