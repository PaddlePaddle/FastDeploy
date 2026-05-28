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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal, Optional


@dataclass
class OutputFallbackContext:
    request: Optional[Any]
    request_id: str
    choice_index: int
    stream: bool
    output: dict
    full_text: Optional[str] = None
    delta_text: Optional[str] = None


@dataclass
class StreamFallbackDecision:
    """Per-delta decision returned by streaming strategy hooks.

    ``action`` semantics (consumed by ``OutputFallbackManager``):

    - ``send``  : emit ``text`` downstream as the current delta.
    - ``hold``  : buffer this delta inside the strategy's ``state``; downstream
                  strategies still run, but the manager emits nothing this round.
    - ``drop``  : discard this delta after downstream strategies observe it.
    - ``flush`` : used by ``on_finish`` to emit any remaining buffered text
                  after the stream ends.
    - ``truncate``: send ``text`` as the final delta and stop further generation.
    """

    action: Literal["send", "hold", "drop", "flush", "truncate"]
    text: str = ""


class OutputFallbackStrategy(ABC):
    """Base class for output fallback strategies.

    Subclasses must implement the two primitives ``should_apply`` and ``apply``,
    which operate on the **full** text. ``on_delta`` / ``on_finish`` are optional
    hooks for streaming scenarios; the default implementations treat each delta
    as an independent piece of text and are suitable for stateless strategies.
    Strategies that need cross-chunk state (e.g. matching patterns spanning
    multiple deltas) should override these hooks and persist state via the
    per-request ``state`` dict managed by ``OutputFallbackManager``.

    The ``context`` argument carries request-level metadata; see
    :class:`OutputFallbackContext` for the available fields and how to use them.
    Built-in implementations may not read ``context`` today, but custom
    strategies are free to.
    """

    name: str = ""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}

    @abstractmethod
    def should_apply(self, text: str, context: OutputFallbackContext) -> bool:
        """Return True if ``apply`` should run on ``text`` for this request."""

    @abstractmethod
    def apply(self, text: str, context: OutputFallbackContext) -> str:
        """Transform the **full** text. Must be a pure function of its inputs;
        do not store cross-call state on ``self``."""

    def on_delta(self, delta_text: str, context: OutputFallbackContext, state: dict) -> StreamFallbackDecision:
        """Streaming hook called for each delta. Default: stateless per-delta apply.

        Override when a strategy must buffer / look across multiple deltas.
        Use the per-request ``state`` dict for any cross-delta state — never
        store it on ``self`` (instances are shared across requests).
        """
        if not self.should_apply(delta_text, context):
            return StreamFallbackDecision(action="send", text=delta_text)
        return StreamFallbackDecision(action="send", text=self.apply(delta_text, context))

    def on_finish(self, context: OutputFallbackContext, state: dict) -> StreamFallbackDecision:
        """Streaming hook called once after the last delta. Override to flush
        any text buffered in ``state`` during ``on_delta``."""
        return StreamFallbackDecision(action="flush")
