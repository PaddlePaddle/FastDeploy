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

import contextlib
import ctypes
import json
from unittest.mock import MagicMock, patch

import numpy as np
import paddle
import pytest

from fastdeploy.config import EPLBConfig
from fastdeploy.eplb.async_expert_loader import (
    AsyncEPLoader,
    create_mmap,
    load_ep_checkpoint,
    load_model_weights_process,
    load_tensor_from_shm_mem,
    save_tensor_to_shm_mem,
)


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
        logger=MagicMock(),
    )
    defaults.update(kw)
    return AsyncEPLoader(**defaults)


def _shm_buffer(data_bytes):
    buf = (ctypes.c_byte * len(data_bytes))(*data_bytes)
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


def test_load_experts_weight_paths():
    """load_experts_weight_from_disk: bf16 path, safetensor path, failure."""
    loader = _make_loader(safetensors=False)
    loader.old_model_ep_rank_to_expert_id_list = np.array([[0, 1], [0, 1]])
    loader.new_model_ep_rank_to_expert_id_list = np.array([[0, 1], [2, 3]])
    with patch.object(loader, "load_weight_bf16_from_disk", return_value=(True, "ok")):
        ok, _ = loader.load_experts_weight_from_disk()
        assert ok
    # safetensor path
    loader2 = _make_loader(safetensors=True)
    loader2.old_model_ep_rank_to_expert_id_list = np.array([[0, 1], [0, 1]])
    loader2.new_model_ep_rank_to_expert_id_list = np.array([[0, 1], [2, 3]])
    with patch.object(loader2, "load_safetensor_fp8_from_disk", return_value=(True, "ok")):
        assert loader2.load_experts_weight_from_disk()[0]
    # failure path
    loader3 = _make_loader()
    loader3.old_model_ep_rank_to_expert_id_list = np.array([[0, 1], [0, 1]])
    loader3.new_model_ep_rank_to_expert_id_list = np.array([[0, 1], [2, 3]])
    with patch.object(loader3, "load_weight_bf16_from_disk", return_value=(False, "err")):
        ok, msg = loader3.load_experts_weight_from_disk()
        assert not ok


def test_bf16_from_disk(tmp_path):
    """load_weight_bf16_from_disk: success + exception."""
    loader = _make_loader(model_dir=str(tmp_path), expert_per_rank=8, moe_layer_start_index=3)
    with patch("paddle.device.get_device", return_value="cpu"), patch("paddle.set_device"):
        ok, _ = loader.load_weight_bf16_from_disk([(3, 0), (4, 1)])
        assert ok and len(loader.moe_file_names) == 4
    loader2 = _make_loader(model_dir=str(tmp_path))
    with patch("paddle.device.get_device", side_effect=RuntimeError("boom")):
        ok, msg = loader2.load_weight_bf16_from_disk([(3, 0)])
        assert not ok and "boom" in msg


def test_fp8_from_disk(tmp_path):
    """load_safetensor_fp8_from_disk."""
    loader = _make_loader(safetensors=True, model_dir=str(tmp_path), expert_per_rank=8, moe_layer_start_index=3)
    fake_map = {}
    names = []
    for proj in ["up_gate_proj", "down_proj"]:
        for quant in ["quant_weight", "weight_scale"]:
            n = f"ernie.layers.3.mlp.experts.0.{proj}.{quant}"
            fake_map[n] = str(tmp_path / "shard.safetensors")
            names.append(n)
    mock_file = MagicMock()
    mock_file.__enter__ = MagicMock(return_value=mock_file)
    mock_file.__exit__ = MagicMock(return_value=False)
    mock_file.keys.return_value = names
    mock_file.get_tensor.return_value = paddle.ones([4], dtype="float32")
    with (
        patch("fastdeploy.eplb.async_expert_loader.load_ep_checkpoint", return_value=fake_map),
        patch("safetensors.safe_open", return_value=mock_file),
        patch("paddle.device.get_device", return_value="cpu"),
        patch("paddle.set_device"),
    ):
        ok, _ = loader.load_safetensor_fp8_from_disk([(3, 0)])
        assert ok and len(loader.cached_weights) == 4


# -- create_mmap --


def _mock_cudart(register_ok=True):
    m = MagicMock()

    class Err:
        cudaSuccess = 0
        cudaErrorInvalidValue = 1

    m.cudaError_t = Err
    ret = Err.cudaSuccess if register_ok else Err.cudaErrorInvalidValue
    m.cudaHostRegister.return_value = (ret,)
    m.cudaGetErrorString.return_value = (Err.cudaSuccess, b"err")
    return m


def test_create_mmap_errors():
    """create_mmap: mmap failure, cudart=None, cuda register failure."""
    with (
        patch("fastdeploy.eplb.async_expert_loader.cudart", _mock_cudart()),
        patch("fastdeploy.eplb.async_expert_loader.libc") as ml,
        patch("os.path.isfile", return_value=True),
        patch("os.open", return_value=5),
        patch("os.ftruncate"),
    ):
        ml.mmap.return_value = -1
        with pytest.raises(OSError):
            create_mmap(["m"], 0, 1, "u", _eplb_config())
    # cudart=None
    with (
        patch("fastdeploy.eplb.async_expert_loader.cudart", None),
        patch("fastdeploy.eplb.async_expert_loader.libc") as ml,
        patch("os.path.isfile", return_value=False),
        patch("builtins.open", MagicMock()),
        patch("os.open", return_value=5),
        patch("os.ftruncate"),
    ):
        ml.mmap.return_value = 12345
        with pytest.raises(ImportError):
            create_mmap(["m"], 0, 1, "u", _eplb_config())
    # cuda register failure
    with (
        patch("fastdeploy.eplb.async_expert_loader.cudart", _mock_cudart(register_ok=False)),
        patch("fastdeploy.eplb.async_expert_loader.libc") as ml,
        patch("os.path.isfile", return_value=False),
        patch("builtins.open", MagicMock()),
        patch("os.open", return_value=5),
        patch("os.ftruncate"),
        patch("ctypes.cast"),
        patch("ctypes.addressof", return_value=0x1000),
    ):
        ml.mmap.return_value = 12345
        with pytest.raises(RuntimeError):
            create_mmap(["m"], 0, 1, "u", _eplb_config())


def test_create_mmap_success():
    """create_mmap: success path + default shmem size."""
    with (
        patch("fastdeploy.eplb.async_expert_loader.cudart", _mock_cudart()),
        patch("fastdeploy.eplb.async_expert_loader.libc") as ml,
        patch("os.path.isfile", return_value=False),
        patch("builtins.open", MagicMock()),
        patch("os.open", return_value=5),
        patch("os.ftruncate"),
        patch("ctypes.cast"),
        patch("ctypes.addressof", return_value=0x1000),
    ):
        ml.mmap.return_value = 12345
        result = create_mmap(["m"], 0, 1, "u", _eplb_config(), MagicMock())
        assert "m" in result


# -- load_model_weights_process --


def _run_process(disk_ok=True, disk_exc=False):
    mg = MagicMock()
    data = MagicMock()
    mg.recv.side_effect = [
        {
            "old_model_ep_rank_to_expert_id_list": np.array([[0, 1]]),
            "new_model_ep_rank_to_expert_id_list": np.array([[0, 1]]),
        },
        KeyboardInterrupt,
    ]
    patches = [
        patch("setproctitle.setproctitle"),
        patch("faulthandler.enable"),
        patch("paddle.set_device"),
        patch("fastdeploy.utils.get_logger", return_value=MagicMock()),
    ]
    if disk_exc:
        patches.append(patch.object(AsyncEPLoader, "load_experts_weight_from_disk", side_effect=RuntimeError("boom")))
    else:
        patches.append(
            patch.object(
                AsyncEPLoader,
                "load_experts_weight_from_disk",
                return_value=(disk_ok, "ok" if disk_ok else "fail"),
            )
        )
    if disk_ok and not disk_exc:
        patches.append(
            patch(
                "fastdeploy.eplb.async_expert_loader.save_tensor_to_shm_mem",
                return_value=[("w", 0, 4, [1], paddle.float32)],
            )
        )
    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        try:
            load_model_weights_process(0, "/fake", 8, 3, "", "uuid", _eplb_config(), data, mg)
        except KeyboardInterrupt:
            pass
    return data


def test_load_model_weights_process():
    """load_model_weights_process: success + failure paths."""
    data = _run_process(disk_ok=True)
    data.send.assert_called_once()
    assert data.send.call_args[0][0]["result"] is True
    data2 = _run_process(disk_ok=False)
    data2.send.assert_called_once()
    assert data2.send.call_args[0][0]["result"] is False
