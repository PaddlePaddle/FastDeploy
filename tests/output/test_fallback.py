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

from pathlib import Path

from fastdeploy.output.fallback import (
    OutputFallbackContext,
    OutputFallbackManager,
    OutputFallbackStrategy,
    StreamFallbackDecision,
)


@OutputFallbackManager.register("test-replace", force=True)
class ReplaceStrategy(OutputFallbackStrategy):
    name = "test-replace"

    def should_apply(self, text: str, context: OutputFallbackContext) -> bool:
        return "bad" in text

    def apply(self, text: str, context: OutputFallbackContext) -> str:
        return text.replace("bad", "good")


@OutputFallbackManager.register("test-suffix", force=True)
class SuffixStrategy(OutputFallbackStrategy):
    name = "test-suffix"

    def should_apply(self, text: str, context: OutputFallbackContext) -> bool:
        return True

    def apply(self, text: str, context: OutputFallbackContext) -> str:
        return text + "-suffix"

    def on_delta(self, delta_text: str, context: OutputFallbackContext, state: dict) -> StreamFallbackDecision:
        return StreamFallbackDecision(action="send", text=self.apply(delta_text, context))


@OutputFallbackManager.register("test-hold", force=True)
class HoldStrategy(OutputFallbackStrategy):
    name = "test-hold"

    def should_apply(self, text: str, context: OutputFallbackContext) -> bool:
        return True

    def apply(self, text: str, context: OutputFallbackContext) -> str:
        return text

    def on_delta(self, delta_text: str, context: OutputFallbackContext, state: dict) -> StreamFallbackDecision:
        state["held"] = state.get("held", "") + delta_text
        return StreamFallbackDecision(action="hold")

    def on_finish(self, context: OutputFallbackContext, state: dict) -> StreamFallbackDecision:
        return StreamFallbackDecision(action="flush", text=state.get("held", ""))


@OutputFallbackManager.register("test-truncate", force=True)
class TruncateStrategy(OutputFallbackStrategy):
    name = "test-truncate"

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self.trigger = self.config.get("trigger", "truncate")

    def should_apply(self, text: str, context: OutputFallbackContext) -> bool:
        return True

    def apply(self, text: str, context: OutputFallbackContext) -> str:
        return text

    def on_delta(self, delta_text: str, context: OutputFallbackContext, state: dict) -> StreamFallbackDecision:
        state["seen"] = delta_text
        if self.trigger in delta_text:
            return StreamFallbackDecision(action="truncate", text=delta_text)
        return StreamFallbackDecision(action="send", text=delta_text)


@OutputFallbackManager.register("test-token-observer", force=True)
class TokenObserverStrategy(OutputFallbackStrategy):
    name = "test-token-observer"

    def should_apply(self, text: str, context: OutputFallbackContext) -> bool:
        return True

    def apply(self, text: str, context: OutputFallbackContext) -> str:
        return text

    def on_delta(self, delta_text: str, context: OutputFallbackContext, state: dict) -> StreamFallbackDecision:
        state["token_count"] = state.get("token_count", 0) + len(context.output.get("token_ids") or [])
        return StreamFallbackDecision(action="send", text=delta_text)


def make_context(text: str, stream: bool = False, token_ids: list[int] | None = None) -> OutputFallbackContext:
    output = {"text": text}
    if token_ids is not None:
        output["token_ids"] = token_ids
    return OutputFallbackContext(
        request=None,
        request_id="test-request::n::0",
        choice_index=0,
        stream=stream,
        output=output,
        full_text=None if stream else text,
        delta_text=text if stream else None,
    )


class TestOutputFallbackStrategy:
    def test_default_on_delta_applies_full_text_strategy(self):
        strategy = ReplaceStrategy()
        decision = strategy.on_delta("bad output", make_context("bad output", stream=True), {})
        assert decision.action == "send"
        assert decision.text == "good output"

    def test_default_on_finish_flushes_nothing(self):
        strategy = ReplaceStrategy()
        decision = strategy.on_finish(make_context("", stream=True), {})
        assert decision.action == "flush"
        assert decision.text == ""


class TestOutputFallbackManager:
    def test_apply_runs_enabled_strategies_in_order(self):
        manager = OutputFallbackManager(strategies=["test-replace", "test-suffix"])
        assert manager.apply("bad output", make_context("bad output")) == "good output-suffix"

    def test_on_delta_send(self):
        manager = OutputFallbackManager(strategies=["test-suffix"])
        decision = manager.on_delta("request-1", 0, "hello", make_context("hello", stream=True))
        assert decision.action == "send"
        assert decision.text == "hello-suffix"

    def test_runs_later_strategy_after_hold(self):
        manager = OutputFallbackManager(strategies=["test-hold", "test-token-observer"])
        decision = manager.on_delta("request-1", 0, "hello", make_context("hello", stream=True, token_ids=[1, 2]))
        assert decision.action == "hold"
        assert decision.text == ""
        assert manager.states["request-1"][(0, "test-token-observer")]["token_count"] == 2

    def test_truncate_continues_later_strategy(self):
        manager = OutputFallbackManager(
            strategies=["test-truncate", "test-suffix"],
            config={"test-truncate": {"trigger": "stop"}},
        )
        decision = manager.on_delta("request-1", 0, "please stop", make_context("please stop", stream=True))
        assert decision.action == "truncate"
        assert decision.text == "please stop-suffix"

    def test_finish_feeds_flush_to_later_strategy_without_token_ids(self):
        manager = OutputFallbackManager(strategies=["test-hold", "test-suffix", "test-token-observer"])
        manager.on_delta("request-1", 0, "held", make_context("held", stream=True, token_ids=[7]))
        finish_decision = manager.on_finish("request-1", 0, make_context("", stream=True, token_ids=[8]))
        assert finish_decision.action == "flush"
        assert finish_decision.text == "held-suffix"
        assert manager.states["request-1"][(0, "test-token-observer")]["token_count"] == 1

    def test_normalizes_strategy_and_config_names(self):
        manager = OutputFallbackManager(
            strategies=["test_truncate"],
            config={"test_truncate": {"trigger": "stop"}},
        )
        decision = manager.on_delta("request-1", 0, "please stop", make_context("please stop", stream=True))
        assert decision.action == "truncate"
        assert decision.text == "please stop"

    def test_cleanup(self):
        manager = OutputFallbackManager(strategies=["test-suffix"])
        manager._get_state("request-1", 0, "test-suffix")["cache"] = "held"
        manager._get_state("request-2", 0, "test-suffix")["cache"] = "other"
        manager.cleanup("request-1")
        assert "request-1" not in manager.states
        assert manager.states["request-2"][(0, "test-suffix")]["cache"] == "other"


class TestOutputFallbackPlugin:
    def test_import(self, tmp_path: Path):
        plugin_path = tmp_path / "custom_fallback.py"
        plugin_path.write_text(
            "from fastdeploy.output.fallback import OutputFallbackManager, OutputFallbackStrategy\n"
            "@OutputFallbackManager.register('custom-plugin-test')\n"
            "class CustomPluginFallback(OutputFallbackStrategy):\n"
            "    name = 'custom-plugin-test'\n"
            "    def should_apply(self, text, context):\n"
            "        return 'bad' in text\n"
            "    def apply(self, text, context):\n"
            "        return text.replace('bad', 'good')\n"
        )
        OutputFallbackManager.import_fallback_plugin(str(plugin_path))
        manager = OutputFallbackManager(strategies=["custom-plugin-test"])
        result = manager.apply("bad output", make_context("bad output"))
        assert result == "good output"
