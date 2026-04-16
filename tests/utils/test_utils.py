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

import argparse
import asyncio
import logging
import pickle
import random
import socket
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import paddle
import pytest
import yaml
from fastapi import Request
from fastapi.exceptions import RequestValidationError

from fastdeploy import utils


def test_output_with_pager_fallback_print(capsys):
    with patch("fastdeploy.utils.subprocess.Popen", side_effect=OSError("no pager")):
        utils._output_with_pager("plain text")
    captured = capsys.readouterr()
    assert "plain text" in captured.out


def test_show_filtered_argument_listgroup(monkeypatch):
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group("ModelConfig", description="model group")
    group.add_argument("--foo", type=int)

    monkeypatch.setattr(utils, "_output_with_pager", Mock())
    monkeypatch.setattr(sys, "argv", ["fastdeploy", "serve", "--help=listgroup"])

    with pytest.raises(SystemExit) as excinfo:
        utils.show_filtered_argument_or_group_from_help(parser, ["serve"])

    assert excinfo.value.code == 0
    assert "ModelConfig" in utils._output_with_pager.call_args[0][0]


def test_show_filtered_argument_no_match(monkeypatch, capsys):
    parser = argparse.ArgumentParser()
    parser.add_argument("--bar", type=str)

    monkeypatch.setattr(sys, "argv", ["fastdeploy", "serve", "--help=unknown"])

    with pytest.raises(SystemExit) as excinfo:
        utils.show_filtered_argument_or_group_from_help(parser, ["serve"])

    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "No group or parameter" in captured.out


def test_daily_rotating_file_handler_cleanup(tmp_path):
    base_log = tmp_path / "app.log"
    base_log.write_text("current")

    (tmp_path / "app.log.2024-01-01").write_text("old")
    (tmp_path / "app.log.2024-01-02").write_text("new")

    handler = utils.DailyRotatingFileHandler(str(base_log), backupCount=1, delay=True)
    handler.delete_expired_files()

    assert not (tmp_path / "app.log.2024-01-01").exists()
    assert (tmp_path / "app.log.2024-01-02").exists()


def test_daily_rotating_file_handler_rollover(tmp_path, monkeypatch):
    base_log = tmp_path / "service.log"
    base_log.write_text("initial")
    handler = utils.DailyRotatingFileHandler(str(base_log), backupCount=0, delay=False)
    assert handler.shouldRollover(None) is False

    monkeypatch.setattr(handler, "_compute_fn", lambda: "service.log.2099-01-01")
    assert handler.shouldRollover(None) is True
    handler.doRollover()
    assert handler.current_filename == "service.log.2099-01-01"
    handler.close()


def test_colored_formatter_adds_color():
    formatter = utils.ColoredFormatter("%(levelname)s:%(message)s")
    record = logging.LogRecord(
        name="test", level=logging.WARNING, pathname="", lineno=1, msg="warn", args=(), exc_info=None
    )
    formatted = formatter.format(record)
    assert "\033[" in formatted


def test_datetime_diff_with_strings():
    start = "2024-01-01 00:00:00"
    end = "2024-01-01 00:00:03"
    assert utils.datetime_diff(start, end) == 3.0
    assert utils.datetime_diff(end, start) == 3.0


def test_get_limited_max_value_validator():
    validator = utils.get_limited_max_value(3)
    assert validator("2") == 2.0
    with pytest.raises(argparse.ArgumentTypeError):
        validator("4")


def test_flexible_argument_parser_yaml_and_correction(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"early_stop_config": {"enable_early_stop": True}}))

    parser = utils.FlexibleArgumentParser()
    parser.add_argument("--early-stop-config", dest="early_stop_config")
    parser.add_argument("--enable-early-stop", dest="enable_early_stop", action="store_true")

    args = parser.parse_args(["--config", str(config_path)])

    assert args.early_stop_config == {"enable_early_stop": True}
    assert args.enable_early_stop is True


def test_resolve_obj_from_strname():
    resolved = utils.resolve_obj_from_strname("fastdeploy.utils.ceil_div")
    assert resolved(5, 2) == 3


def test_check_unified_ckpt_variants(tmp_path):
    model_dir = tmp_path / "unified"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_text("ok")
    assert utils.check_unified_ckpt(str(model_dir)) is True

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    assert utils.check_unified_ckpt(str(empty_dir)) is False

    split_dir = tmp_path / "split"
    split_dir.mkdir()
    (split_dir / "model-00001-of-00002.safetensors").write_text("chunk1")
    (split_dir / "model-00002-of-00002.safetensors").write_text("chunk2")
    assert utils.check_unified_ckpt(str(split_dir)) is True

    broken_dir = tmp_path / "broken"
    broken_dir.mkdir()
    (broken_dir / "model-bad.safetensors").write_text("bad")

    with pytest.raises(Exception, match="Failed to check unified checkpoint"):
        utils.check_unified_ckpt(str(broken_dir))


def test_get_random_port_and_availability():
    port = utils.get_random_port()
    assert utils.is_port_available("127.0.0.1", port) is True

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", port))
    try:
        assert utils.is_port_available("127.0.0.1", port) is False
    finally:
        sock.close()


def test_singleton_decorator():
    @utils.singleton
    class Dummy:
        def __init__(self):
            self.value = 1

    first = Dummy()
    second = Dummy()
    assert first is second


def test_none_or_str():
    assert utils.none_or_str("None") is None
    assert utils.none_or_str("value") == "value"


def test_is_list_of_checks():
    assert utils.is_list_of([], int, check="first") is True
    assert utils.is_list_of(["a", "b"], str, check="first") is True
    assert utils.is_list_of(["a", "b"], str, check="all") is True
    assert utils.is_list_of([1, "b"], int, check="all") is False
    assert utils.is_list_of("not list", int, check="first") is False
    with pytest.raises(AssertionError):
        utils.is_list_of([1], int, check="unknown")


def test_import_from_path(tmp_path):
    module_path = tmp_path / "sample_module.py"
    module_path.write_text("VALUE = 42")

    module = utils.import_from_path("sample_module", module_path)
    assert module.VALUE == 42


def test_is_package_installed():
    assert utils.is_package_installed("pip") is True
    assert utils.is_package_installed("definitely-not-a-package") is False


def test_parse_quantization():
    assert utils.parse_quantization('{"bits": 4}') == {"bits": 4}
    assert utils.parse_quantization("int8") == {"quantization": "int8"}


def test_deprecated_warnings():
    with patch("fastdeploy.utils.console_logger.warning") as warning:
        utils.deprecated_kwargs_warning(enable_mm=True)

    warning.assert_called_once()

    parser = argparse.ArgumentParser()
    parser.add_argument("--deprecated", action=utils.DeprecatedOptionWarning)

    with patch("fastdeploy.utils.console_logger.warning") as action_warning:
        args = parser.parse_args(["--deprecated"])

    assert args.deprecated is True
    action_warning.assert_called_once()


def test_stateful_semaphore_status():
    semaphore = utils.StatefulSemaphore(2)

    async def run_ops():
        await semaphore.acquire()
        await semaphore.acquire()
        assert semaphore.locked() is True
        semaphore.release()
        return semaphore.status()

    status = asyncio.run(run_ops())

    assert status["available"] == 1
    assert status["acquired"] == 1
    assert status["max_value"] == 2
    assert isinstance(status["uptime"], float)


def test_parse_type_and_optional_type():
    parser_type = utils.parse_type(int)
    assert parser_type("5") == 5
    with pytest.raises(argparse.ArgumentTypeError):
        parser_type("bad")

    optional_int = utils.optional_type(int)
    assert optional_int("None") is None
    assert optional_int("") is None
    assert optional_int("7") == 7


def test_to_numpy_and_to_tensor_roundtrip():
    image_tensor = paddle.to_tensor(np.zeros((2, 2), dtype="float32"))
    feature_tensor = paddle.to_tensor(np.ones((1, 3), dtype="float32"))

    task = SimpleNamespace(
        multimodal_inputs={
            "images": image_tensor,
            "image_features": [feature_tensor],
        }
    )

    utils.to_numpy([task])

    assert isinstance(task.multimodal_inputs["images"], np.ndarray)
    assert isinstance(task.multimodal_inputs["image_features"][0], np.ndarray)

    utils.to_tensor([task])

    assert isinstance(task.multimodal_inputs["images"], paddle.Tensor)
    assert isinstance(task.multimodal_inputs["image_features"][0], paddle.Tensor)


def test_to_tensor_handles_missing_inputs():
    task = SimpleNamespace(multimodal_inputs=None)
    utils.to_tensor([task])


def test_chunk_list_and_str_to_datetime():
    items = list(utils.chunk_list([1, 2, 3, 4, 5], 2))
    assert items == [[1, 2], [3, 4], [5]]

    with_ms = utils.str_to_datetime("2024-01-01 12:00:00.123")
    without_ms = utils.str_to_datetime("2024-01-01 12:00:00")
    assert with_ms.microsecond == 123000
    assert without_ms.microsecond == 0


def test_download_file_success(tmp_path):
    data_chunks = [b"abc", b"def"]
    response = Mock()
    response.headers = {"content-length": "6"}
    response.iter_content.return_value = data_chunks
    response.raise_for_status.return_value = None

    with patch("fastdeploy.utils.requests.get", return_value=response):
        target = tmp_path / "file.bin"
        assert utils.download_file("http://example.com/file.bin", target) is True
        assert target.read_bytes() == b"abcdef"


def test_download_file_failure_cleans(tmp_path):
    response = Mock()
    response.raise_for_status.side_effect = RuntimeError("boom")
    with patch("fastdeploy.utils.requests.get", return_value=response):
        target = tmp_path / "file.bin"
        with pytest.raises(RuntimeError, match="Download failed"):
            utils.download_file("http://example.com/file.bin", target)
    assert not target.exists()


def test_extract_tar_success(tmp_path, capsys):
    tar_path = tmp_path / "data.tar"
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    sample_file = tmp_path / "sample.txt"
    sample_file.write_text("payload")
    with tarfile.open(tar_path, "w") as tar:
        tar.add(sample_file, arcname="sample.txt")

    utils.extract_tar(tar_path, output_dir)

    captured = capsys.readouterr()
    assert "Successfully extracted" in captured.out
    assert (output_dir / "sample.txt").read_text() == "payload"


def test_extract_tar_failure(tmp_path):
    tar_path = tmp_path / "bad.tar"
    tar_path.write_text("not a tar")
    with pytest.raises(RuntimeError, match="Extraction failed"):
        utils.extract_tar(tar_path, tmp_path / "out")


def test_download_model_success_and_cleanup(tmp_path):
    def fake_download(url, save_path):
        Path(save_path).write_text("data")
        return True

    def fake_extract(tar_path, output_dir):
        (Path(output_dir) / "model.txt").write_text("ok")

    with (
        patch("fastdeploy.utils.download_file", side_effect=fake_download),
        patch("fastdeploy.utils.extract_tar", side_effect=fake_extract),
    ):
        utils.download_model("http://example.com/model.tar", tmp_path, "temp.tar")
    assert (tmp_path / "model.txt").read_text() == "ok"
    assert not (tmp_path / "temp.tar").exists()


def test_download_model_failure_removes_temp(tmp_path):
    temp_tar = tmp_path / "temp.tar"
    temp_tar.write_text("stale")

    with patch("fastdeploy.utils.download_file", side_effect=RuntimeError("fail")):
        with pytest.raises(Exception, match="Failed to get model"):
            utils.download_model("http://example.com/model.tar", tmp_path, "temp.tar")
    assert not temp_tar.exists()


def test_flexible_argument_parser_conversion_error(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"number": "bad"}))

    parser = utils.FlexibleArgumentParser()
    parser.add_argument("--number", type=int)

    args = parser.parse_args(["--config", str(config_path)])
    assert args.number == "bad"


def test_check_download_links():
    response = SimpleNamespace(metadata=SimpleNamespace(content_length="5"))
    client = Mock()
    client.get_object_meta_data.return_value = response

    assert utils.check_download_links(client, ["bos://bucket/path/file.bin"]) is None

    client.get_object_meta_data.side_effect = RuntimeError("oops")
    assert "download error" in utils.check_download_links(client, ["bos://bucket/path/file.bin"])


def test_download_from_bos_success_and_retry():
    payload = pickle.dumps(np.array([1, 2, 3]))
    client = Mock()
    client.get_object_as_string.side_effect = [RuntimeError("request rate is too high"), payload]

    with patch("fastdeploy.utils.llm_logger.warning") as logger:
        results = list(utils.download_from_bos(client, "bos://bucket/path/data.pkl", retry=1))

    assert results[0][0] is True
    np.testing.assert_array_equal(results[0][1], np.array([1, 2, 3]))
    logger.assert_called_once()


def test_download_from_bos_failure_no_retry():
    client = Mock()
    client.get_object_as_string.side_effect = RuntimeError("other error")

    results = list(utils.download_from_bos(client, "bos://bucket/path/data.pkl", retry=1))
    assert results[0][0] is False


def test_clamp_prompt_logprobs_handles_none():
    assert utils.clamp_prompt_logprobs(None) is None


def test_import_from_path_missing_spec(tmp_path):
    fake_path = tmp_path / "missing.py"
    with patch("fastdeploy.utils.importlib.util.spec_from_file_location", return_value=None):
        with pytest.raises(ModuleNotFoundError):
            utils.import_from_path("missing", fake_path)


def test_exception_handler_default_and_validation():
    scope = {"type": "http", "method": "GET", "path": "/", "headers": [], "client": ("127.0.0.1", 12345)}
    request = Request(scope)

    response = asyncio.run(utils.ExceptionHandler.handle_exception(request, RuntimeError("boom")))
    assert response.status_code == 500
    assert response.body

    error = RequestValidationError([{"loc": ("body", "messages"), "msg": "required", "type": "value_error"}])
    response = asyncio.run(utils.ExceptionHandler.handle_request_validation_exception(request, error))
    assert response.status_code == 400

    empty_error = RequestValidationError([])
    response = asyncio.run(utils.ExceptionHandler.handle_request_validation_exception(request, empty_error))
    assert response.status_code == 400


def test_engine_and_parameter_error_fields():
    err = utils.EngineError("oops", error_code=503)
    assert err.error_code == 503

    param_error = utils.ParameterError("param", "bad")
    assert param_error.param == "param"
    assert param_error.message == "bad"


def test_set_random_seed_calls():
    utils.set_random_seed(123)
    python_val = random.random()
    numpy_val = np.random.rand()
    paddle_val = float(paddle.rand([1]).numpy()[0])

    utils.set_random_seed(123)
    assert random.random() == python_val
    assert np.random.rand() == numpy_val
    assert float(paddle.rand([1]).numpy()[0]) == paddle_val


def test_get_host_ip_returns_value():
    assert isinstance(utils.get_host_ip(), str)


def test_retrive_model_from_server_local_path(tmp_path):
    local = tmp_path / "model"
    local.mkdir()
    assert utils.retrive_model_from_server(str(local)) == str(local)


def test_retrive_model_from_server_invalid_source(monkeypatch):
    monkeypatch.setattr(utils.envs, "FD_MODEL_SOURCE", "INVALID")
    monkeypatch.setattr(utils.envs, "FD_MODEL_CACHE", None)
    with pytest.raises(ValueError, match="Unsupported model source"):
        utils.retrive_model_from_server("some-model")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
