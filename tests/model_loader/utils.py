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

import os
import shutil
import signal
import socket
import subprocess
import time
import traceback
from multiprocessing import Process, Queue

import pytest

TokensIdText = list[tuple[list[int], str]]
FD_CACHE_QUEUE_PORT = int(os.getenv("FD_CACHE_QUEUE_PORT", 8234))


def clear_logs():
    log_path = os.path.join(os.getcwd(), "log")
    if os.path.exists(log_path):
        try:
            shutil.rmtree(log_path)
            print(f"Deleted log directory: {log_path}")
        except Exception as e:
            print(f"Failed to delete log directory {log_path}: {e}")
    else:
        print(f"No log directory found at {log_path}")


def print_logs():
    log_dir = os.path.join(os.getcwd(), "log")
    log_file = os.path.join(log_dir, "workerlog.0")

    if not os.path.exists(log_file):
        print(f"Log file {log_file} does not exist.")
        return

    print(f"\n===== {log_file} start =====")
    with open(log_file, "r") as f:
        for line in f:
            print(line, end="")
    print(f"\n===== {log_file} end =====\n")


def run_with_timeout(target, args, timeout=60 * 5):
    clear_logs()
    result_queue = Queue()
    p = Process(target=target, args=(*args, result_queue))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        print_logs()
        raise RuntimeError("Worker process hung and was terminated")
    try:
        result = result_queue.get(timeout=60)
    except Exception as e:
        raise RuntimeError(f"Failed to get result from worker: {e}")
    finally:
        result_queue.close()
        result_queue.join_thread()
        clean_ports([FD_CACHE_QUEUE_PORT])

    return result


def form_model_get_output_topp0(
    fd_runner,
    model_path,
    tensor_parallel_size,
    max_num_seqs,
    max_model_len,
    max_tokens,
    quantization,
    load_choices,
    prompts,
    result_queue,
):
    try:
        with fd_runner(
            model_path,
            tensor_parallel_size=tensor_parallel_size,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            load_choices=load_choices,
            quantization=quantization,
        ) as fd_model:
            fd_outputs = fd_model.generate_topp0(prompts, max_tokens=max_tokens)
            result_queue.put(fd_outputs)
    except Exception:
        print(f"Failed using {load_choices} loader to load model from {model_path}.")
        traceback.print_exc()
        pytest.fail(f"Failed to initialize LLM model from {model_path}")


def kill_process_on_port(port: int):
    """
    Kill processes that are listening on the given port.
    Uses multiple methods to ensure thorough cleanup.
    """
    current_pid = os.getpid()
    parent_pid = os.getppid()

    # Method 1: Use lsof to find processes
    try:
        output = subprocess.check_output(f"lsof -i:{port} -t", shell=True).decode().strip()
        for pid in output.splitlines():
            pid = int(pid)
            if pid in (current_pid, parent_pid):
                print(f"Skip killing current process (pid={pid}) on port {port}")
                continue
            try:
                # First try SIGTERM for graceful shutdown
                os.kill(pid, signal.SIGTERM)
                time.sleep(1)
                # Then SIGKILL if still running
                os.kill(pid, signal.SIGKILL)
                print(f"Killed process on port {port}, pid={pid}")
            except ProcessLookupError:
                pass  # Process already terminated
    except subprocess.CalledProcessError:
        pass

    # Method 2: Use netstat and fuser as backup
    try:
        # Find processes using netstat and awk
        cmd = f"netstat -tulpn 2>/dev/null | grep :{port} | awk '{{print $7}}' | cut -d'/' -f1"
        output = subprocess.check_output(cmd, shell=True).decode().strip()
        for pid in output.splitlines():
            if pid and pid.isdigit():
                pid = int(pid)
                if pid in (current_pid, parent_pid):
                    continue
                try:
                    os.kill(pid, signal.SIGKILL)
                    print(f"Killed process (netstat) on port {port}, pid={pid}")
                except ProcessLookupError:
                    pass
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Method 3: Use fuser if available
    try:
        subprocess.run(f"fuser -k {port}/tcp", shell=True, timeout=5)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        pass


def clean_ports(ports_to_clean: list[int]):
    """
    Kill all processes occupying the ports listed in PORTS_TO_CLEAN.
    """
    print(f"Cleaning ports: {ports_to_clean}")
    for port in ports_to_clean:
        kill_process_on_port(port)

    # Double check and retry if ports are still in use
    time.sleep(2)
    for port in ports_to_clean:
        if is_port_open("127.0.0.1", port, timeout=0.1):
            print(f"Port {port} still in use, retrying cleanup...")
            kill_process_on_port(port)
            time.sleep(1)


def is_port_open(host: str, port: int, timeout=1.0):
    """
    Check if a TCP port is open on the given host.
    Returns True if connection succeeds, False otherwise.
    """
    try:
        with socket.create_connection((host, port), timeout):
            return True
    except Exception:
        return False


def get_paddle_model_path(base_model_name: str) -> str:
    fd_base_path = os.getenv("MODEL_PATH")
    if fd_base_path:
        fd_model_path = os.path.join(fd_base_path, base_model_name)
    else:
        fd_model_path = base_model_name
    return fd_model_path


def get_torch_model_path(base_model_name: str) -> str:
    """return (fastdeploy_path, huggingface_path)"""
    # FastDeploy model path
    fd_base_path = os.getenv("MODEL_PATH")
    # HuggingFace model path
    torch_model_path = os.path.join(
        fd_base_path,
        "torch",
        base_model_name,
    )

    return torch_model_path


def check_tokens_id_and_text_close(
    *,
    outputs_0_lst: TokensIdText,
    outputs_1_lst: TokensIdText,
    name_0: str,
    name_1: str,
    threshold: float = 0.0,
) -> None:
    assert len(outputs_0_lst) == len(outputs_1_lst)

    for prompt_idx, (outputs_0, outputs_1) in enumerate(zip(outputs_0_lst, outputs_1_lst)):
        assert len(outputs_0) == len(outputs_1)
        output_ids_0, output_str_0 = outputs_0
        output_ids_1, output_str_1 = outputs_1

        if threshold > 0:
            diff_rate = calculate_diff_rate(output_str_0, output_str_1)
            if diff_rate >= threshold:
                fail_msg = (
                    f"Test{prompt_idx}:"
                    f"\n{name_0}:\t{output_str_0!r}"
                    f"\n{name_1}:\t{output_str_1!r}"
                    f"\nDiff rate: {diff_rate:.4f} >= threshold: {threshold}"
                )
                raise AssertionError(fail_msg)
        else:
            # Loop through generated tokens.
            for idx, (output_id_0, output_id_1) in enumerate(zip(output_ids_0, output_ids_1)):
                is_tok_mismatch = output_id_0 != output_id_1
                if is_tok_mismatch:
                    fail_msg = (
                        f"Test{prompt_idx}:"
                        f"\nMatched tokens:\t{output_ids_0[:idx]}"
                        f"\n{name_0}:\t{output_str_0!r}"
                        f"\n{name_1}:\t{output_str_1!r}"
                    )
                    raise AssertionError(fail_msg)


def calculate_diff_rate(text1, text2):
    """
    Calculate the difference rate between two strings
    based on the normalized Levenshtein edit distance.
    Returns a float in [0,1], where 0 means identical.
    """
    if text1 == text2:
        return 0.0

    len1, len2 = len(text1), len(text2)
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    for i in range(len1 + 1):
        for j in range(len2 + 1):
            if i == 0 or j == 0:
                dp[i][j] = i + j
            elif text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    edit_distance = dp[len1][len2]
    max_len = max(len1, len2)
    return edit_distance / max_len if max_len > 0 else 0.0
