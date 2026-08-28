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

from dataclasses import dataclass, field
from typing import Dict, Optional

import paddle
from paddleformers.utils.log import logger

from fastdeploy.config import FDConfig, validate_thinking_token_sequences
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
    # Multi-token marker mode: partial-match buffers and forced-emission progress
    start_buffer: list[int] = field(default_factory=list)
    end_buffer: list[int] = field(default_factory=list)
    forcing: bool = False
    forcing_pos: int = 0


class ThinkingBudgetLogitsProcessor(LogitsProcessor):
    """Limit the number of tokens generated in the thinking phase.

    The processor tracks per-request thinking state and forces the thinking end
    token when the budget is reached. If a stop sentence is configured, the
    processor emits the stop sentence first and then the thinking end token.
    Request-specific configuration is provided via logits_processors_args:

        {"thinking_budget": <int>}

    Requires model_config to provide think_start_id and think_end_id. If any of
    these are missing or invalid (-1), the processor falls back to multi-token
    marker sequences from model_config.think_token_sequences (a dict with
    required "start"/"end" sequence lists and a "forced_end" id list), which
    supports tokenizers where <think>/</think> are not single vocab tokens.
    Without either form of markers, the processor is disabled.
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
        self._single_token_mode = self.think_start_token_id >= 0 and self.think_end_token_id >= 0
        sequences = validate_thinking_token_sequences(
            getattr(fd_config.model_config, "think_token_sequences", None), fd_config.model_config.vocab_size
        )
        self.think_start_sequences: tuple[tuple[int, ...], ...] = ()
        self.think_end_sequences: tuple[tuple[int, ...], ...] = ()
        self.think_forced_end_ids: list[int] = []
        if sequences is not None:
            if self._single_token_mode:
                raise ValueError("Configure either single-token thinking markers or think_token_sequences, not both")
            self.think_start_sequences = tuple(dict.fromkeys(tuple(sequence) for sequence in sequences["start"]))
            self.think_end_sequences = tuple(dict.fromkeys(tuple(sequence) for sequence in sequences["end"]))
            self.think_forced_end_ids = list(sequences["forced_end"])
        self._sequence_mode = sequences is not None
        self._enabled = self._single_token_mode or self._sequence_mode
        if not self._enabled:
            logger.warning(
                "ThinkingBudgetLogitsProcessor disabled: missing token ids "
                f"(think_start={think_start_id}, think_end={think_end_id}). "
                "Ensure model vocab contains <think> and </think> tokens."
            )
        self._states: Dict[str, _ThinkingState] = {}
        self._active_req_ids: list[str] = []
        self._active_budgets: list[int] = []
        self._active_slots: list[int] = []

    def _resolve_thinking_budget(self, logit_proc_args, slot_id: int, share_inputs: dict) -> Optional[int]:
        """Per-request thinking budget hook; model wrappers may override the source."""
        thinking_budget = logit_proc_args.get("thinking_budget") if logit_proc_args else None
        if thinking_budget is None or not isinstance(thinking_budget, int) or thinking_budget < 0:
            return None
        return thinking_budget

    @staticmethod
    def _buffer_match(buffer: list[int], sequences: tuple[tuple[int, ...], ...]) -> int:
        """2 = buffer exactly matches a sequence, 1 = proper prefix, 0 = mismatch."""
        is_prefix = False
        for sequence in sequences:
            if list(sequence[: len(buffer)]) == buffer:
                if len(sequence) == len(buffer):
                    return 2
                is_prefix = True
        return 1 if is_prefix else 0

    def _consume_token_sequences(self, state: _ThinkingState, token_id: int) -> None:
        """Advance the multi-token marker state machine by one token.

        Tokens held in a prefix buffer are only counted as thinking content once
        they are ruled out as marker candidates, mirroring single-token mode
        where marker tokens never consume the budget.
        """
        if state.forcing:
            forced_sequence = (state.stop_sentence_token_ids or []) + self.think_forced_end_ids
            expected_token_id = forced_sequence[state.forcing_pos]
            if token_id != expected_token_id:
                raise RuntimeError(
                    "Forced thinking-end token mismatch: " f"expected {expected_token_id}, received {token_id}"
                )
            state.forcing_pos += 1
            if state.forcing_pos == len(forced_sequence):
                state.forcing = False
                state.forcing_pos = 0
                state.ended = True
                state.start_buffer.clear()
                state.end_buffer.clear()
            return
        if state.started and not state.ended:
            state.end_buffer.append(token_id)
            while state.end_buffer:
                status = self._buffer_match(state.end_buffer, self.think_end_sequences)
                if status == 2:
                    state.ended = True
                    state.forcing = False
                    state.forcing_pos = 0
                    state.end_buffer.clear()
                    state.start_buffer.clear()
                    return
                if status == 1:
                    return
                state.end_buffer.pop(0)
                state.tokens_after_start += 1
            return
        state.start_buffer.append(token_id)
        while state.start_buffer:
            status = self._buffer_match(state.start_buffer, self.think_start_sequences)
            if status == 2:
                # Matches in both idle and ended phase, so a new thinking round
                # after a natural end restarts the budget.
                state.started = True
                state.ended = False
                state.tokens_after_start = 0
                state.stop_sentence_pos = 0
                state.forcing = False
                state.forcing_pos = 0
                state.start_buffer.clear()
                state.end_buffer.clear()
                return
            if status == 1:
                return
            state.start_buffer.pop(0)

    def _scan_prompt_state_sequences(self, state: _ThinkingState, prompt_slice: list) -> None:
        """Replay the prompt through the sequence matcher; prompt tokens never
        consume the budget, and a partial marker at the prompt tail stays buffered."""
        for token_id in prompt_slice:
            token_id = int(token_id)
            if token_id >= 0:
                self._consume_token_sequences(state, token_id)
        state.tokens_after_start = 0
        if prompt_slice:
            state.last_token_id = int(prompt_slice[-1])

    def _consume_decode_token_sequences(
        self, state: _ThinkingState, last_token_id: int, current_step_idx: Optional[int]
    ) -> None:
        # Without step indices, repeated identical tokens cannot be told apart from
        # duplicate update_state calls, so fall back to value-based dedup there.
        if current_step_idx is None:
            if state.forcing or last_token_id != state.last_token_id:
                state.last_token_id = last_token_id
                self._consume_token_sequences(state, last_token_id)
        elif current_step_idx != state.last_step_idx:
            state.last_step_idx = current_step_idx
            state.last_token_id = last_token_id
            self._consume_token_sequences(state, last_token_id)

    def _scan_prompt_state(self, prompt_slice: list[int]) -> tuple[bool, bool, int, Optional[int]]:
        started = False
        ended = False
        in_thinking = False
        for token_id in prompt_slice:
            if token_id == self.think_start_token_id:
                started = True
                ended = False
                in_thinking = True
            elif token_id == self.think_end_token_id and in_thinking:
                ended = True
                in_thinking = False
        last_token_id = int(prompt_slice[-1]) if started and prompt_slice else None
        return started, ended, 0, last_token_id

    def update_state(self, share_inputs: dict) -> None:
        if not self._enabled:
            return
        stop_flags = share_inputs["stop_flags"]
        req_ids = share_inputs["req_ids"]
        logits_processors_args = share_inputs["logits_processors_args"]
        prompt_ids = share_inputs.get("prompt_ids")
        token_ids_all = share_inputs.get("token_ids_all")
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

        candidate_slots = []
        candidate_req_ids = []
        candidate_args = []
        candidate_budgets = []
        for slot_id, (req_id, stop_flag, logit_proc_args) in enumerate(
            zip(req_ids, stop_flags_list, logits_processors_args)
        ):
            if stop_flag or not req_id:
                continue

            thinking_budget = self._resolve_thinking_budget(logit_proc_args, slot_id, share_inputs)
            if thinking_budget is None:
                continue

            candidate_slots.append(slot_id)
            candidate_req_ids.append(req_id)
            candidate_args.append(logit_proc_args)
            candidate_budgets.append(thinking_budget)

        inactive_req_ids = set(self._states) - set(candidate_req_ids)
        for req_id in inactive_req_ids:
            del self._states[req_id]

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

        prompt_source = prompt_ids if prompt_ids is not None else token_ids_all

        for idx, slot_id in enumerate(candidate_slots):
            req_id = candidate_req_ids[idx]
            logit_proc_args = candidate_args[idx]
            thinking_budget = candidate_budgets[idx]

            state = self._states.setdefault(req_id, _ThinkingState())
            prompt_initialized = False
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
                    prompt_initialized = True

            current_step_idx = step_idx_by_slot.get(slot_id)
            state.current_step_idx = current_step_idx

            if not state.started and not state.prompt_checked:
                if prompt_source is not None and prompt_lens is not None:
                    if prompt_lens_np is not None:
                        prompt_len = int(prompt_lens_np[slot_id])
                    else:
                        prompt_len = int(prompt_lens[slot_id])
                    prompt_slice = prompt_source[slot_id, :prompt_len]
                    if hasattr(prompt_slice, "numpy"):
                        prompt_slice = prompt_slice.numpy().tolist()
                    elif hasattr(prompt_slice, "tolist"):
                        prompt_slice = prompt_slice.tolist()
                    else:
                        prompt_slice = list(prompt_slice)
                    if prompt_ids is None:
                        prompt_slice = [int(token_id) for token_id in prompt_slice if int(token_id) >= 0]
                    if self._sequence_mode:
                        self._scan_prompt_state_sequences(state, prompt_slice)
                        if current_step_idx is not None and state.last_step_idx is None:
                            state.last_step_idx = current_step_idx
                    else:
                        prompt_started, prompt_ended, prompt_tokens_after_start, prompt_last_token_id = (
                            self._scan_prompt_state(prompt_slice)
                        )
                        if prompt_started:
                            state.started = True
                            state.ended = prompt_ended
                            state.tokens_after_start = prompt_tokens_after_start
                            state.last_token_id = prompt_last_token_id
                            if current_step_idx is not None and state.last_step_idx is None:
                                state.last_step_idx = current_step_idx
                    state.prompt_checked = True
                    prompt_initialized = True

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

            if last_token_id is not None and last_token_id >= 0 and not (self._sequence_mode and prompt_initialized):
                if self._sequence_mode:
                    self._consume_decode_token_sequences(state, last_token_id, current_step_idx)
                elif not state.started and last_token_id == self.think_start_token_id:
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
                elif current_step_idx is None:
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
            if self._sequence_mode:
                if not state.forcing:
                    if state.tokens_after_start < max(budget - stop_sentence_len, 0):
                        continue
                    state.forcing = True
                    state.forcing_pos = 0
                forced_sequence = stop_sentence_token_ids + self.think_forced_end_ids
                force_token_id = forced_sequence[state.forcing_pos]
                if force_token_id >= logits.shape[1]:
                    raise ValueError(
                        f"Forced thinking-end token {force_token_id} exceeds vocabulary size {logits.shape[1]}"
                    )
                logits[slot_id, :] = -float("inf")
                logits[slot_id, force_token_id] = 0.0
                continue
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

            logits[slot_id, :] = -float("inf")
            logits[slot_id, self.think_end_token_id] = 0.0
            state.last_token_id = self.think_end_token_id
            state.last_step_idx = state.current_step_idx
            state.ended = True

        return logits
