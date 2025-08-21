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
import signal
import socket
import subprocess
import warnings

TokensIdText = list[tuple[list[int], str]]


# (token_ids, text)
def kill_process_on_port(port: int):
    """
    Kill processes that are listening on the given port.
    Uses `lsof` to find process ids and sends SIGKILL.
    """
    try:
        output = subprocess.check_output(f"lsof -i:{port} -t", shell=True).decode().strip()
        for pid in output.splitlines():
            os.kill(int(pid), signal.SIGKILL)
            print(f"Killed process on port {port}, pid={pid}")
    except subprocess.CalledProcessError:
        pass


def clean_ports(ports_to_clean: list[int]):
    """
    Kill all processes occupying the ports listed in PORTS_TO_CLEAN.
    """
    for port in ports_to_clean:
        kill_process_on_port(port)


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


def check_tokens_id_and_text_close(
    *,
    outputs_0_lst: TokensIdText,
    outputs_1_lst: TokensIdText,
    name_0: str,
    name_1: str,
    warn_on_mismatch: bool = True,
) -> None:
    assert len(outputs_0_lst) == len(outputs_1_lst)

    for prompt_idx, (outputs_0, outputs_1) in enumerate(zip(outputs_0_lst, outputs_1_lst)):
        assert len(outputs_0) == len(outputs_1)
        output_ids_0, output_str_0 = outputs_0
        output_ids_1, output_str_1 = outputs_1

        # Loop through generated tokens.
        for idx, (output_id_0, output_id_1) in enumerate(zip(output_ids_0, output_ids_1)):
            is_tok_mismatch = output_id_0 != output_id_1
            if is_tok_mismatch and warn_on_mismatch:
                fail_msg = (
                    f"Test{prompt_idx}:"
                    f"\nMatched tokens:\t{output_ids_0[:idx]}"
                    f"\n{name_0}:\t{output_str_0!r}"
                    f"\n{name_1}:\t{output_str_1!r}"
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("always")
                    warnings.warn(fail_msg, stacklevel=2)
                break
    else:
        if output_str_0 != output_str_1 and warn_on_mismatch:
            fail_msg = f"Test{prompt_idx}:" f"\n{name_0}:\t{output_str_0!r}" f"\n{name_1}:\t{output_str_1!r}"
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn(fail_msg, stacklevel=2)
