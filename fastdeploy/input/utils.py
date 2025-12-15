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
]

IDS_TYPE_FLAG = {"text": 0, "image": 1, "video": 2, "audio": 3}


from typing import Any, Callable, Dict, List, Tuple


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
