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
    "check_mm_limits",
]

IDS_TYPE_FLAG = {"text": 0, "image": 1, "video": 2, "audio": 3}


from typing import Any, Callable, Dict, List, Tuple, Union


def check_mm_limits(
    item: Union[Dict[str, Any], List[Dict[str, Any]]],
    limit_mm_per_prompt: Dict[str, int],
    type_check_mode: str = "in",
) -> None:
    """
    Validate multimodal inputs against configured limits.

    Args:
        item: Input request item to validate. Can be either:
            - dict: Contains prompt and multi_modal_data
            - list: Contains messages with multimodal content
        limit_mm_per_prompt: Dictionary mapping modality names to their limits
        type_check_mode: How to check content types:
            - "in": Check if type is in ["image_url", "image"] (default)
            - "eq": Check if type equals "image" exactly

    Raises:
        ValueError: If input exceeds configured limits or if type_check_mode is invalid
    """
    if type_check_mode not in ("in", "eq"):
        raise ValueError(f"Invalid type_check_mode: {type_check_mode}. Must be 'in' or 'eq'")
    
    if isinstance(item, dict):
        # Request contains prompt and multi_modal_data
        mm_data = item
    else:
        # Request contains messages
        mm_data = {"image": [], "video": []}

        for message in item:
            if isinstance(message.get("content"), list):
                for part in message["content"]:
                    part_type = part.get("type")
                    if type_check_mode == "in":
                        if part_type in ["image_url", "image"]:
                            mm_data["image"].append(part)
                        elif part_type in ["video_url", "video"]:
                            mm_data["video"].append(part)
                    else:  # type_check_mode == "eq"
                        if part_type == "image":
                            mm_data["image"].append(part)
                        elif part_type == "video":
                            mm_data["video"].append(part)

    for modality, data in mm_data.items():
        if modality in limit_mm_per_prompt:
            limit = limit_mm_per_prompt[modality]
            if len(data) > limit:
                raise ValueError(f"Too many {modality} items in prompt, got {len(data)} but limit is {limit}")


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
