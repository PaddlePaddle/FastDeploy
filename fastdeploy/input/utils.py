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

__all__ = [
    "IDS_TYPE_FLAG",
    "MAX_IMAGE_DIMENSION",
]

import os
import socket
from typing import Any, Callable, Dict, List, Tuple
from urllib.parse import urlparse

from fastdeploy.utils import console_logger

IDS_TYPE_FLAG = {"text": 0, "image": 1, "video": 2, "audio": 3}

MAX_IMAGE_DIMENSION = 9999999

# Hub endpoints for connectivity check, keyed by DOWNLOAD_SOURCE value
_HUB_ENDPOINTS = {
    "huggingface": ("huggingface.co", 443),
    "modelscope": ("modelscope.cn", 443),
}


def _get_hub_endpoint():
    """Return (host, port, hub_name) for the active download hub."""
    source = os.environ.get("DOWNLOAD_SOURCE", "huggingface")
    if source == "aistudio":
        url = os.environ.get("AISTUDIO_ENDPOINT", "http://git.aistudio.baidu.com")
        parsed = urlparse(url)
        host = parsed.hostname or "git.aistudio.baidu.com"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        return host, port, "aistudio"
    host, port = _HUB_ENDPOINTS.get(source, ("huggingface.co", 443))
    return host, port, source


def validate_model_path(model_name_or_path):
    """
    Validate model path before from_pretrained calls.
    Give immediate feedback instead of letting users wait 50s+ for timeout.
    """
    if os.path.isdir(model_name_or_path) or os.path.isfile(model_name_or_path):
        return  # Local path exists, no network needed

    host, port, hub_name = _get_hub_endpoint()

    console_logger.warning(
        f"Model path '{model_name_or_path}' is not a local directory or file, "
        f"will try to download from {hub_name} hub."
    )

    # Quick connectivity check — fail fast instead of waiting 50s
    try:
        sock = socket.create_connection((host, port), timeout=3)
        sock.close()
    except OSError:
        console_logger.warning(
            f"Cannot reach {host}. If the model is stored locally, "
            f"please check the path '{model_name_or_path}'. Otherwise check "
            f"network/proxy settings (DOWNLOAD_SOURCE={hub_name})."
        )


def process_stop_token_ids(
    request: Dict[str, Any],
    update_stop_seq_fn: Callable[[List[str]], Tuple[List[List[int]], List[int]]],
) -> None:
    stop_token_ids_final = []

    if request.get("stop_token_ids") is not None:
        stop_token_ids = request.get("stop_token_ids")
        if isinstance(stop_token_ids, list) and len(stop_token_ids) > 0:
            if isinstance(stop_token_ids[0], int):
                # List[int] -> List[List[int]]
                stop_token_ids_final.extend([[t] for t in stop_token_ids])
            elif isinstance(stop_token_ids[0], list):
                # Already List[List[int]]
                stop_token_ids_final.extend(stop_token_ids)

    stop_sequences = request.get("stop", [])
    if stop_sequences:
        stop_seqs, _ = update_stop_seq_fn(stop_sequences)
        stop_token_ids_final.extend(stop_seqs)

    # Update request
    if stop_token_ids_final:
        stop_seqs_len = [len(seq) for seq in stop_token_ids_final]
        request["stop_token_ids"] = stop_token_ids_final
        request["stop_seqs_len"] = stop_seqs_len
