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

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import paddle
from paddleformers.utils.log import logger

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.logits_processor.base import LogitsProcessor


@dataclass
class _ThinkingState:
    started: bool = False
    ended: bool = False
    tokens_after_start: int = 0
    last_token_id: Optional[int] = None
    last_step_idx: Optional[int] = None
    current_step_idx: Optional[int] = None
    stop_sentence_token_ids: Optional[list[int]] = None
    stop_sentence_pos: int = 0
    prompt_checked: bool = False


class ThinkingBudgetLogitsProcessor(LogitsProcessor):
    """Limit the number of tokens generated in the thinking phase.

    The processor tracks per-request thinking state and forces a newline token
    when the budget is reached, followed by the thinking end token on the next step.
    Request-specific configuration is provided via logits_processors_args:

        {"thinking_budget": <int>}

    Requires model_config to provide think_start_id, think_end_id, and line_break_id.
    If any of these are missing or invalid (-1), the processor will be disabled.
    """

    def __init__(self, fd_config: FDConfig) -> None:
        self.device = paddle.device.get_device()
        self.dtype = fd_config.model_config.dtype
        think_start_id = getattr(fd_config.model_config, "think_start_id", -1)
        think_end_id = getattr(fd_config.model_config, "think_end_id", -1)
        line_break_id = getattr(fd_config.model_config, "line_break_id", -1)
        self.think_start_token_id = think_start_id if isinstance(think_start_id, int) and think_start_id >= 0 else -1
        self.think_end_token_id = think_end_id if isinstance(think_end_id, int) and think_end_id >= 0 else -1
        self.line_break_token_id = line_break_id if isinstance(line_break_id, int) and line_break_id >= 0 else -1
        self._enabled = (
            self.think_start_token_id >= 0 and self.think_end_token_id >= 0 and self.line_break_token_id >= 0
        )
        if not self._enabled:
            logger.warning(
                "ThinkingBudgetLogitsProcessor disabled: missing token ids "
                f"(think_start={think_start_id}, think_end={think_end_id}, line_break={line_break_id}). "
                "Ensure model vocab contains <think>, </think> tokens and line_break_id is configured."
            )
        self._states: Dict[str, _ThinkingState] = {}
        self._active_req_ids: list[str] = []
        self._active_budgets: list[int] = []
        self._active_slots: list[int] = []

    def update_state(self, share_inputs: dict) -> None:
        if not self._enabled:
            return
        stop_flags = share_inputs["stop_flags"]
        req_ids = share_inputs["req_ids"]
        logits_processors_args = share_inputs["logits_processors_args"]
        prompt_ids = share_inputs.get("prompt_ids")
        prompt_lens = share_inputs.get("prompt_lens")
        pre_ids = share_inputs.get("pre_ids")
        next_tokens = share_inputs.get("next_tokens")
        step_idx = share_inputs.get("step_idx")

        stop_flags_list = stop_flags.numpy().reshape(-1).tolist()

        prompt_lens_np = None
        if prompt_lens is not None and hasattr(prompt_lens, "numpy"):
            prompt_lens_np = prompt_lens.numpy().reshape(-1)

        self._active_req_ids = []
        self._active_budgets = []
        self._active_slots = []

        active_req_ids = []
        for req_id, stop_flag in zip(req_ids, stop_flags_list):
            if stop_flag:
                continue
            if req_id:
                active_req_ids.append(req_id)

        inactive_req_ids = set(self._states.keys()) - set(active_req_ids)
        for req_id in inactive_req_ids:
            self._states.pop(req_id, None)

        candidate_slots = []
        candidate_req_ids = []
        candidate_args = []
        candidate_budgets = []
        for slot_id, (req_id, stop_flag, logit_proc_args) in enumerate(
            zip(req_ids, stop_flags_list, logits_processors_args)
        ):
            if stop_flag or not req_id:
                continue

            thinking_budget = logit_proc_args.get("thinking_budget") if logit_proc_args else None
            if thinking_budget is None or not isinstance(thinking_budget, int) or thinking_budget < 0:
                continue

            candidate_slots.append(slot_id)
            candidate_req_ids.append(req_id)
            candidate_args.append(logit_proc_args)
            candidate_budgets.append(thinking_budget)

        if not candidate_slots:
            return

        place = None
        if step_idx is not None:
            place = step_idx.place
        elif next_tokens is not None:
            place = next_tokens.place
        slot_tensor = (
            paddle.to_tensor(candidate_slots, dtype="int64", place=place)
            if place
            else paddle.to_tensor(candidate_slots, dtype="int64")
        )

        step_idx_by_slot = {}
        if step_idx is not None:
            step_idx_sel = paddle.index_select(step_idx, slot_tensor, axis=0).numpy().reshape(-1)
            for idx, slot_id in enumerate(candidate_slots):
                step_idx_by_slot[slot_id] = int(step_idx_sel[idx])

        next_token_by_slot = {}
        if next_tokens is not None:
            next_sel = paddle.index_select(next_tokens, slot_tensor, axis=0).numpy().reshape(-1)
            for idx, slot_id in enumerate(candidate_slots):
                next_token_by_slot[slot_id] = int(next_sel[idx])

        for idx, slot_id in enumerate(candidate_slots):
            req_id = candidate_req_ids[idx]
            logit_proc_args = candidate_args[idx]
            thinking_budget = candidate_budgets[idx]

            state = self._states.setdefault(req_id, _ThinkingState())
            if logit_proc_args:
                stop_sentence_token_ids = logit_proc_args.get("think_stop_sentence_token_ids")
                if isinstance(stop_sentence_token_ids, list) and all(
                    isinstance(tid, int) and tid >= 0 for tid in stop_sentence_token_ids
                ):
                    if stop_sentence_token_ids != state.stop_sentence_token_ids:
                        state.stop_sentence_token_ids = stop_sentence_token_ids
                        state.stop_sentence_pos = 0
                else:
                    state.stop_sentence_token_ids = None
                    state.stop_sentence_pos = 0

                if logit_proc_args.get("think_prompt_checked") and not state.prompt_checked:
                    prompt_started = logit_proc_args.get("think_prompt_started")
                    prompt_ended = logit_proc_args.get("think_prompt_ended")
                    prompt_tokens_after_start = logit_proc_args.get("think_prompt_tokens_after_start")
                    prompt_last_token_id = logit_proc_args.get("think_prompt_last_token_id")
                    if isinstance(prompt_started, bool):
                        state.started = prompt_started
                    if isinstance(prompt_ended, bool):
                        state.ended = prompt_ended
                    if isinstance(prompt_tokens_after_start, int) and prompt_tokens_after_start >= 0:
                        state.tokens_after_start = prompt_tokens_after_start
                    if isinstance(prompt_last_token_id, int) and prompt_last_token_id >= 0:
                        state.last_token_id = prompt_last_token_id
                    state.prompt_checked = True

            current_step_idx = step_idx_by_slot.get(slot_id)
            state.current_step_idx = current_step_idx

            if not state.started and not state.prompt_checked:
                if prompt_ids is not None and prompt_lens is not None:
                    if prompt_lens_np is not None:
                        prompt_len = int(prompt_lens_np[slot_id])
                    else:
                        prompt_len = int(prompt_lens[slot_id])
                    prompt_slice = prompt_ids[slot_id, :prompt_len]
                    prompt_slice = prompt_slice.numpy().tolist()
                    if self.think_start_token_id in prompt_slice:
                        state.started = True
                        start_pos = prompt_slice.index(self.think_start_token_id)
                        tokens_after = prompt_slice[start_pos + 1 :]
                        if self.think_end_token_id in tokens_after:
                            end_pos = tokens_after.index(self.think_end_token_id)
                            state.tokens_after_start = end_pos + 1
                            state.ended = True
                        else:
                            state.tokens_after_start = len(tokens_after)
                        if prompt_slice:
                            state.last_token_id = int(prompt_slice[-1])
                        if current_step_idx is not None and state.last_step_idx is None:
                            state.last_step_idx = current_step_idx
                    state.prompt_checked = True

            last_token_id = next_token_by_slot.get(slot_id)

            if last_token_id is None or last_token_id < 0:
                if pre_ids is not None:
                    slot_pre_ids = pre_ids[slot_id]
                    if current_step_idx is not None:
                        step_pos = current_step_idx - 1
                        if 0 <= step_pos < slot_pre_ids.shape[0]:
                            last_token_id = int(slot_pre_ids[step_pos].item())
                        else:
                            last_token_id = int(slot_pre_ids[-1].item())
                    else:
                        last_token_id = int(slot_pre_ids[-1].item())

            if last_token_id is not None and last_token_id >= 0:
                if not state.started and last_token_id == self.think_start_token_id:
                    state.started = True
                    state.tokens_after_start = 0
                    state.last_token_id = last_token_id
                    if current_step_idx is not None and state.last_step_idx is None:
                        state.last_step_idx = current_step_idx
                    if state.started and not state.ended:
                        self._active_req_ids.append(req_id)
                        self._active_budgets.append(thinking_budget)
                        self._active_slots.append(slot_id)
                    continue
                if current_step_idx is None:
                    if last_token_id != state.last_token_id:
                        state.last_token_id = last_token_id
                        if state.started and not state.ended:
                            if last_token_id == self.think_end_token_id:
                                state.ended = True
                            elif last_token_id != self.think_start_token_id:
                                state.tokens_after_start += 1
                else:
                    if state.last_step_idx is None:
                        state.last_step_idx = current_step_idx
                        state.last_token_id = last_token_id
                    elif current_step_idx != state.last_step_idx:
                        # Count one token per decode step. step_idx can jump under
                        # certain schedulers, but we should not over-count.
                        state.last_step_idx = current_step_idx
                        state.last_token_id = last_token_id
                        if state.started and not state.ended:
                            if last_token_id == self.think_end_token_id:
                                state.ended = True
                            elif last_token_id != self.think_start_token_id:
                                state.tokens_after_start += 1

            if state.started and not state.ended:
                self._active_req_ids.append(req_id)
                self._active_budgets.append(thinking_budget)
                self._active_slots.append(slot_id)
                continue

    def apply(self, logits: paddle.Tensor) -> paddle.Tensor:
        if not self._enabled or not self._active_req_ids:
            return logits

        for active_idx, req_id in enumerate(self._active_req_ids):
            state = self._states.get(req_id)
            if state is None or state.ended:
                continue

            budget = self._active_budgets[active_idx]
            slot_id = self._active_slots[active_idx]
            stop_sentence_token_ids = state.stop_sentence_token_ids or []
            stop_sentence_len = len(stop_sentence_token_ids)
            if stop_sentence_len > 0:
                budget_threshold = budget - stop_sentence_len
                if budget_threshold < 0:
                    budget_threshold = 0
                if state.stop_sentence_pos > 0 or state.tokens_after_start >= budget_threshold:
                    if state.stop_sentence_pos < stop_sentence_len:
                        force_token_id = stop_sentence_token_ids[state.stop_sentence_pos]
                        logits[slot_id, :] = -float("inf")
                        logits[slot_id, force_token_id] = 0.0
                        state.last_token_id = force_token_id
                        state.last_step_idx = state.current_step_idx
                        state.stop_sentence_pos += 1
                        continue
                    logits[slot_id, :] = -float("inf")
                    logits[slot_id, self.think_end_token_id] = 0.0
                    state.last_token_id = self.think_end_token_id
                    state.last_step_idx = state.current_step_idx
                    state.ended = True
                    continue

            if state.tokens_after_start < budget:
                continue

            if state.last_token_id != self.line_break_token_id:
                logits[slot_id, :] = -float("inf")
                logits[slot_id, self.line_break_token_id] = 0.0
                state.last_token_id = self.line_break_token_id
                state.last_step_idx = state.current_step_idx
                continue

            logits[slot_id, :] = -float("inf")
            logits[slot_id, self.think_end_token_id] = 0.0
            state.last_token_id = self.think_end_token_id
            state.last_step_idx = state.current_step_idx
            state.ended = True

        return logits
