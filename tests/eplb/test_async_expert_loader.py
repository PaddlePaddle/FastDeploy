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

import ctypes
import json
import logging
import os

import numpy as np
import paddle
import pytest

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


# -- Lightweight stubs (real objects, no mocking) --


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
    """Cudart stub with minimal register/error interface."""

    cudaError_t = _CudaErr

    def __init__(self, ok=True):
        self._ret = _CudaErr.cudaSuccess if ok else _CudaErr.cudaErrorInvalidValue

    def cudaHostRegister(self, addr, size, flags):
        return (self._ret,)

    def cudaGetErrorString(self, err):
        return (_CudaErr.cudaSuccess, b"err")


class _StubLibc:
    """Libc stub — only mmap is needed."""

    def __init__(self, mmap_ret=-1):
        self._ret = mmap_ret

    def mmap(self, *a):
        return self._ret


class _StubPtr:
    """Pointer stub for ctypes.cast result."""

    contents = None


class _DummyFileCtx:
    """Minimal file object returned by builtins.open stub."""

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


# -- Helpers --


def _eplb_config(**overrides):
    defaults = {
        "redundant_expert_async_load_model_shmem_size_gb": 1,
        "model_use_safetensors": False,
        "moe_quant_type": "",
    }
    defaults.update(overrides)
    return EPLBConfig(defaults)


def _make_loader(safetensors=False, **kw):
    cfg = _eplb_config(model_use_safetensors=safetensors)
    defaults = dict(
        model_dir="/fake/model",
        eplb_config=cfg,
        rank=0,
        expert_per_rank=2,
        moe_layer_start_index=1,
        moe_quant_type="",
        logger=_logger,
    )
    defaults.update(kw)
    return AsyncEPLoader(**defaults)


_GC_GUARD: list = []  # prevents ctypes buffers from being garbage-collected


@pytest.fixture(autouse=True)
def _force_cpu():
    """Prevent segfaults on CI GPU — ctypes.string_at reads CPU pointers only."""
    paddle.set_device("cpu")


def _shm_buffer(data_bytes):
    buf = (ctypes.c_byte * len(data_bytes))(*data_bytes)
    _GC_GUARD.append(buf)
    return ctypes.cast(buf, ctypes.POINTER(ctypes.c_int8))


# -- save/load shared memory I/O --


def test_save_single_and_multiple(tmp_path):
    """save_tensor_to_shm_mem: single tensor + multiple with offsets."""
    fp = tmp_path / "shm"
    fp.write_bytes(b"\x00" * 8192)
    t1 = paddle.ones([4], dtype="float32")
    t2 = paddle.zeros([8], dtype="float32")
    infos = save_tensor_to_shm_mem([("w1", t1), ("w2", t2)], str(fp))
    assert infos[0][:3] == ("w1", 0, 16)
    assert infos[1][1] == 16  # offset


def test_save_errors(tmp_path):
    """save: file not exist + overflow."""
    with pytest.raises(OSError):
        save_tensor_to_shm_mem([], "/nonexistent/path")
    fp = tmp_path / "tiny"
    fp.write_bytes(b"\x00" * 4)
    with pytest.raises(IOError):
        save_tensor_to_shm_mem([("big", paddle.ones([100], dtype="float32"))], str(fp))


@pytest.mark.parametrize(
    "np_dtype,pd_dtype,vals",
    [
        (np.float32, paddle.float32, [1.0, 2.0, 3.0]),
        (np.uint8, paddle.uint8, [0, 128, 255]),
        (np.int8, paddle.int8, [-1, 0, 127]),
        (np.int32, paddle.int32, [10, 20, 30]),
    ],
)
def test_load_numeric_dtypes(np_dtype, pd_dtype, vals):
    """load_tensor_from_shm_mem: standard numeric dtypes."""
    arr = np.array(vals, dtype=np_dtype)
    raw = arr.tobytes()
    result = load_tensor_from_shm_mem([("w", 0, len(raw), [len(vals)], pd_dtype)], _shm_buffer(raw))
    np.testing.assert_array_equal(result[0][1].numpy(), arr)


def test_load_special_dtypes():
    """load: bfloat16, float8_e4m3fn, and unsupported."""
    arr16 = np.array([0x3F80, 0x4000], dtype=np.uint16)
    result = load_tensor_from_shm_mem(
        [("w", 0, len(arr16.tobytes()), [2], paddle.bfloat16)], _shm_buffer(arr16.tobytes())
    )
    assert list(result[0][1].shape) == [2]
    arr8 = np.array([0x38, 0x40], dtype=np.uint8)
    result2 = load_tensor_from_shm_mem(
        [("w", 0, len(arr8.tobytes()), [2], paddle.float8_e4m3fn)], _shm_buffer(arr8.tobytes())
    )
    assert list(result2[0][1].shape) == [2]
    with pytest.raises(TypeError):
        load_tensor_from_shm_mem([("w", 0, 8, [2], paddle.complex64)], _shm_buffer(b"\x00" * 8))


# -- AsyncEPLoader + helpers --


def test_checkpoint_load(tmp_path):
    """load_ep_checkpoint: missing dir + valid index."""
    assert load_ep_checkpoint("/nonexistent") == {}
    data = {"weight_map": {"a": "s1.safetensors", "b": "s2.safetensors"}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(data))
    assert len(load_ep_checkpoint(str(tmp_path))) == 2


def test_loader_init_and_reset():
    """AsyncEPLoader constructor + reset."""
    loader = _make_loader()
    assert loader.model_path == "/fake/model"
    loader.old_model_ep_rank_to_expert_id_list = np.array([[1, 2]])
    loader.cached_weights = [("x", "y")]
    loader.reset()
    assert loader.old_model_ep_rank_to_expert_id_list is None
    assert loader.cached_weights == []


def test_load_experts_weight_paths(monkeypatch):
    """load_experts_weight_from_disk: bf16 path, safetensor path, failure."""
    loader = _make_loader(safetensors=False)
    loader.old_model_ep_rank_to_expert_id_list = np.array([[0, 1], [0, 1]])
    loader.new_model_ep_rank_to_expert_id_list = np.array([[0, 1], [2, 3]])
    monkeypatch.setattr(loader, "load_weight_bf16_from_disk", lambda *a: (True, "ok"))
    ok, _ = loader.load_experts_weight_from_disk()
    assert ok
    # safetensor path
    loader2 = _make_loader(safetensors=True)
    loader2.old_model_ep_rank_to_expert_id_list = np.array([[0, 1], [0, 1]])
    loader2.new_model_ep_rank_to_expert_id_list = np.array([[0, 1], [2, 3]])
    monkeypatch.setattr(loader2, "load_safetensor_fp8_from_disk", lambda *a: (True, "ok"))
    assert loader2.load_experts_weight_from_disk()[0]
    # failure path
    loader3 = _make_loader()
    loader3.old_model_ep_rank_to_expert_id_list = np.array([[0, 1], [0, 1]])
    loader3.new_model_ep_rank_to_expert_id_list = np.array([[0, 1], [2, 3]])
    monkeypatch.setattr(loader3, "load_weight_bf16_from_disk", lambda *a: (False, "err"))
    ok, msg = loader3.load_experts_weight_from_disk()
    assert not ok


def test_load_experts_weight_mismatch_length(monkeypatch):
    """load_experts_weight_from_disk: mismatch old/new expert id lengths."""
    loader = _make_loader(safetensors=False, moe_layer_start_index=0, expert_per_rank=3)
    loader.old_model_ep_rank_to_expert_id_list = np.array([[0, 1]], dtype=object)
    loader.new_model_ep_rank_to_expert_id_list = np.array([[0, 1, 2]], dtype=object)
    monkeypatch.setattr(loader, "load_weight_bf16_from_disk", lambda *a: (True, "ok"))
    ok, msg = loader.load_experts_weight_from_disk()
    assert ok is False
    assert "length not equal" in msg


def test_load_experts_weight_exception_path():
    """load_experts_weight_from_disk: unexpected exception branch."""
    loader = _make_loader()
    loader.old_model_ep_rank_to_expert_id_list = None
    loader.new_model_ep_rank_to_expert_id_list = np.array([[0, 1]])
    ok, msg = loader.load_experts_weight_from_disk()
    assert ok is False
    assert "Failed to load_experts_weight_from_disk" in msg


def test_bf16_from_disk(tmp_path):
    """load_weight_bf16_from_disk: success + exception."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(paddle.device, "get_device", lambda: "cpu")
        mp.setattr(paddle, "set_device", lambda *a: None)
        loader = _make_loader(model_dir=str(tmp_path), expert_per_rank=8, moe_layer_start_index=3)
        ok, _ = loader.load_weight_bf16_from_disk([(3, 0), (4, 1)])
        assert ok and len(loader.moe_file_names) == 4

    def _boom():
        raise RuntimeError("boom")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(paddle.device, "get_device", _boom)
        loader2 = _make_loader(model_dir=str(tmp_path))
        ok, msg = loader2.load_weight_bf16_from_disk([(3, 0)])
        assert not ok and "boom" in msg


def test_fp8_from_disk(tmp_path, monkeypatch):
    """load_safetensor_fp8_from_disk."""
    loader = _make_loader(safetensors=True, model_dir=str(tmp_path), expert_per_rank=8, moe_layer_start_index=3)
    fake_map = {}
    names = []
    for proj in ["up_gate_proj", "down_proj"]:
        for quant in ["quant_weight", "weight_scale"]:
            n = f"ernie.layers.3.mlp.experts.0.{proj}.{quant}"
            fake_map[n] = str(tmp_path / "shard.safetensors")
            names.append(n)
    tensors = {n: paddle.ones([4], dtype="float32") for n in names}
    stub_file = _StubSafeFile(tensors)
    monkeypatch.setattr(_ael_mod, "load_ep_checkpoint", lambda path: fake_map)
    monkeypatch.setattr("safetensors.safe_open", lambda *a, **kw: stub_file)
    monkeypatch.setattr(paddle.device, "get_device", lambda: "cpu")
    monkeypatch.setattr(paddle, "set_device", lambda *a: None)
    ok, _ = loader.load_safetensor_fp8_from_disk([(3, 0)])
    assert ok and len(loader.cached_weights) == 4


# -- create_mmap --


def test_create_mmap_errors():
    """create_mmap: mmap failure, cudart=None, cuda register failure."""
    # mmap failure
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_ael_mod, "cudart", _StubCudart())
        mp.setattr(_ael_mod, "libc", _StubLibc(mmap_ret=-1))
        mp.setattr(os.path, "isfile", lambda p: True)
        mp.setattr(os, "open", lambda *a: 5)
        mp.setattr(os, "ftruncate", lambda *a: None)
        with pytest.raises(OSError):
            create_mmap(["m"], 0, 1, "u", _eplb_config())
    # cudart=None
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_ael_mod, "cudart", None)
        mp.setattr(_ael_mod, "libc", _StubLibc(mmap_ret=12345))
        mp.setattr(os.path, "isfile", lambda p: False)
        mp.setattr("builtins.open", lambda *a, **kw: _DummyFileCtx())
        mp.setattr(os, "open", lambda *a: 5)
        mp.setattr(os, "ftruncate", lambda *a: None)
        with pytest.raises(ImportError):
            create_mmap(["m"], 0, 1, "u", _eplb_config())
    # cuda register failure
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_ael_mod, "cudart", _StubCudart(ok=False))
        mp.setattr(_ael_mod, "libc", _StubLibc(mmap_ret=12345))
        mp.setattr(os.path, "isfile", lambda p: False)
        mp.setattr("builtins.open", lambda *a, **kw: _DummyFileCtx())
        mp.setattr(os, "open", lambda *a: 5)
        mp.setattr(os, "ftruncate", lambda *a: None)
        mp.setattr(ctypes, "cast", lambda ptr, typ: _StubPtr())
        mp.setattr(ctypes, "addressof", lambda obj: 0x1000)
        with pytest.raises(RuntimeError):
            create_mmap(["m"], 0, 1, "u", _eplb_config())


def test_create_mmap_success():
    """create_mmap: success path."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_ael_mod, "cudart", _StubCudart())
        mp.setattr(_ael_mod, "libc", _StubLibc(mmap_ret=12345))
        mp.setattr(os.path, "isfile", lambda p: False)
        mp.setattr("builtins.open", lambda *a, **kw: _DummyFileCtx())
        mp.setattr(os, "open", lambda *a: 5)
        mp.setattr(os, "ftruncate", lambda *a: None)
        mp.setattr(ctypes, "cast", lambda ptr, typ: _StubPtr())
        mp.setattr(ctypes, "addressof", lambda obj: 0x1000)
        result = create_mmap(["m"], 0, 1, "u", _eplb_config(), _logger)
        assert "m" in result


# -- load_model_weights_process --


def _run_process(disk_ok=True):
    mg = _StubConn(
        [
            {
                "old_model_ep_rank_to_expert_id_list": np.array([[0, 1]]),
                "new_model_ep_rank_to_expert_id_list": np.array([[0, 1]]),
            }
        ]
    )
    data = _StubConn()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("setproctitle.setproctitle", lambda *a: None)
        mp.setattr("faulthandler.enable", lambda *a: None)
        mp.setattr(paddle, "set_device", lambda *a: None)
        mp.setattr("fastdeploy.utils.get_logger", lambda *a, **kw: _logger)
        mp.setattr(
            AsyncEPLoader,
            "load_experts_weight_from_disk",
            lambda self: (disk_ok, "ok" if disk_ok else "fail"),
        )
        if disk_ok:
            mp.setattr(
                _ael_mod,
                "save_tensor_to_shm_mem",
                lambda *a, **kw: [("w", 0, 4, [1], paddle.float32)],
            )
        try:
            load_model_weights_process(0, "/fake", 8, 3, "", "uuid", _eplb_config(), data, mg)
        except KeyboardInterrupt:
            pass
    return data


def test_load_model_weights_process():
    """load_model_weights_process: success + failure paths."""
    data = _run_process(disk_ok=True)
    assert len(data.sent) == 1
    assert data.sent[0]["result"] is True
    data2 = _run_process(disk_ok=False)
    assert len(data2.sent) == 1
    assert data2.sent[0]["result"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
