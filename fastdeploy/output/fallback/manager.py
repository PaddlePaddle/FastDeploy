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

import os
from typing import Callable, Optional, Union

from fastdeploy.output.fallback.base import (
    OutputFallbackContext,
    OutputFallbackStrategy,
    StreamFallbackDecision,
)
from fastdeploy.utils import data_processor_logger, import_from_path, is_list_of


class OutputFallbackManager:
    fallback_strategies: dict[str, type[OutputFallbackStrategy]] = {}

    @classmethod
    def get_strategy(cls, name: str) -> type[OutputFallbackStrategy]:
        name = name.replace("_", "-")
        if name in cls.fallback_strategies:
            return cls.fallback_strategies[name]
        raise KeyError(f"output fallback strategy: '{name}' not found")

    @classmethod
    def _register_strategy(
        cls,
        module: type[OutputFallbackStrategy],
        strategy_name: Optional[Union[str, list[str]]] = None,
        force: bool = True,
    ) -> None:
        if not issubclass(module, OutputFallbackStrategy):
            raise TypeError(f"module must be subclass of OutputFallbackStrategy, but got {type(module)}")
        if strategy_name is None:
            strategy_name = module.name or module.__name__
        if isinstance(strategy_name, str):
            strategy_name = [strategy_name]
        for name in strategy_name:
            name = name.replace("_", "-")
            if not force and name in cls.fallback_strategies:
                existed_module = cls.fallback_strategies[name]
                raise KeyError(f"{name} is already registered at {existed_module.__module__}")
            cls.fallback_strategies[name] = module

    @classmethod
    def register(
        cls, name: Optional[Union[str, list[str]]] = None, force: bool = True, module: Union[type, None] = None
    ) -> Union[type, Callable]:
        if not isinstance(force, bool):
            raise TypeError(f"force must be a boolean, but got {type(force)}")
        if not (name is None or isinstance(name, str) or is_list_of(name, str)):
            raise TypeError("name must be None, an instance of str, or a sequence of str, " f"but got {type(name)}")
        if module is not None:
            cls._register_strategy(module=module, strategy_name=name, force=force)
            return module

        def _register(module):
            cls._register_strategy(module=module, strategy_name=name, force=force)
            return module

        return _register

    @classmethod
    def import_fallback_plugin(cls, plugin_path: str) -> None:
        module_name = os.path.splitext(os.path.basename(plugin_path))[0]
        try:
            import_from_path(module_name, plugin_path)
        except Exception:
            data_processor_logger.exception(
                "Failed to load output fallback module '%s' from %s.", module_name, plugin_path
            )

    def __init__(self, strategies: list[str], config: Optional[dict] = None):
        self.strategies = [name.replace("_", "-") for name in strategies]
        self.config = {name.replace("_", "-"): value for name, value in (config or {}).items()}
        self.instances = [self.get_strategy(name)(self.config.get(name, {})) for name in self.strategies]
        self.states: dict[str, dict[tuple[int, str], dict]] = {}

    def apply(self, text: str, context: OutputFallbackContext) -> str:
        result_text = text
        for strategy in self.instances:
            try:
                if strategy.should_apply(result_text, context):
                    result_text = strategy.apply(result_text, context)
            except Exception:
                data_processor_logger.exception("Failed to apply output fallback strategy '%s'.", strategy.name)
        return result_text

    def on_delta(
        self, request_id: str, choice_index: int, delta_text: str, context: OutputFallbackContext
    ) -> StreamFallbackDecision:
        current_text = delta_text
        held = False
        truncated = False
        for strategy in self.instances:
            state = self._get_state(request_id, choice_index, strategy.name)
            try:
                decision = strategy.on_delta(current_text, context, state)
            except Exception:
                data_processor_logger.exception(
                    "Failed to apply streaming output fallback strategy '%s'.", strategy.name
                )
                continue
            if decision.action == "truncate":
                # Mark terminal but keep iterating so downstream strategies can
                # still post-process / buffer the final text.
                truncated = True
                current_text = decision.text or current_text
                continue
            if decision.action == "hold":
                held = True
                current_text = decision.text or current_text
                continue
            current_text = decision.text
        if truncated:
            return StreamFallbackDecision(action="truncate", text="" if held else current_text)
        if held:
            return StreamFallbackDecision(action="hold", text="")
        return StreamFallbackDecision(action="send", text=current_text)

    def on_finish(self, request_id: str, choice_index: int, context: OutputFallbackContext) -> StreamFallbackDecision:
        pending = ""
        for strategy in self.instances:
            state = self._get_state(request_id, choice_index, strategy.name)
            try:
                decision = strategy.on_finish(context, state)
            except Exception:
                data_processor_logger.exception(
                    "Failed to finish streaming output fallback strategy '%s'.", strategy.name
                )
                continue
            if decision.text:
                pending = decision.text
        return StreamFallbackDecision(action="flush", text=pending)

    def cleanup(self, request_id: str) -> None:
        self.states.pop(request_id, None)

    def _get_state(self, request_id: str, choice_index: int, strategy_name: str) -> dict:
        return self.states.setdefault(request_id, {}).setdefault((choice_index, strategy_name), {})
