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


@OutputFallbackManager.register("test-hold-drop", force=True)
class HoldDropStrategy(OutputFallbackStrategy):
    name = "test-hold-drop"

    def should_apply(self, text: str, context: OutputFallbackContext) -> bool:
        return True

    def apply(self, text: str, context: OutputFallbackContext) -> str:
        return text

    def on_delta(self, delta_text: str, context: OutputFallbackContext, state: dict) -> StreamFallbackDecision:
        state["held"] = state.get("held", "") + delta_text
        return StreamFallbackDecision(action="hold")

    def on_finish(self, context: OutputFallbackContext, state: dict) -> StreamFallbackDecision:
        return StreamFallbackDecision(action="flush", text="")


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


@OutputFallbackManager.register("test-text-observer", force=True)
class TextObserverStrategy(OutputFallbackStrategy):
    name = "test-text-observer"

    def should_apply(self, text: str, context: OutputFallbackContext) -> bool:
        return True

    def apply(self, text: str, context: OutputFallbackContext) -> str:
        return text

    def on_delta(self, delta_text: str, context: OutputFallbackContext, state: dict) -> StreamFallbackDecision:
        state["seen"] = delta_text
        return StreamFallbackDecision(action="send", text=delta_text)


@OutputFallbackManager.register("test-mutable-state", force=True)
class MutableStateStrategy(OutputFallbackStrategy):
    name = "test-mutable-state"

    def should_apply(self, text: str, context: OutputFallbackContext) -> bool:
        return True

    def apply(self, text: str, context: OutputFallbackContext) -> str:
        return text

    def on_delta(self, delta_text: str, context: OutputFallbackContext, state: dict) -> StreamFallbackDecision:
        state.setdefault("items", []).append(delta_text)
        return StreamFallbackDecision(action="send", text=delta_text)


@OutputFallbackManager.register("test-raising-state", force=True)
class RaisingStateStrategy(OutputFallbackStrategy):
    name = "test-raising-state"

    def should_apply(self, text: str, context: OutputFallbackContext) -> bool:
        return True

    def apply(self, text: str, context: OutputFallbackContext) -> str:
        return text

    def on_delta(self, delta_text: str, context: OutputFallbackContext, state: dict) -> StreamFallbackDecision:
        state["phase"] = "partial"
        raise RuntimeError("boom")


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
        decision = manager.on_delta("request-1", "hello", make_context("hello", stream=True))
        assert decision.action == "send"
        assert decision.text == "hello-suffix"

    def test_runs_later_strategy_after_hold(self):
        manager = OutputFallbackManager(strategies=["test-hold", "test-token-observer"])
        decision = manager.on_delta("request-1", "hello", make_context("hello", stream=True, token_ids=[1, 2]))
        assert decision.action == "hold"
        assert decision.text == ""
        assert "test-token-observer" not in manager.states.get("request-1", {})

    def test_later_strategy_observes_text_when_hold_returns_empty_text(self):
        manager = OutputFallbackManager(strategies=["test-hold", "test-text-observer"])
        decision = manager.on_delta("request-1", "hello", make_context("hello", stream=True))
        assert decision.action == "hold"
        assert decision.text == ""
        assert "test-text-observer" not in manager.states.get("request-1", {})

    def test_hold_does_not_commit_mutable_trial_state(self):
        manager = OutputFallbackManager(strategies=["test-mutable-state", "test-hold"])
        manager._get_state("request-1", "test-mutable-state")["items"] = []
        decision = manager.on_delta("request-1", "hello", make_context("hello", stream=True))
        assert decision.action == "hold"
        assert manager.states["request-1"]["test-mutable-state"]["items"] == []

    def test_delta_exception_does_not_commit_partial_trial_state(self):
        manager = OutputFallbackManager(strategies=["test-raising-state", "test-suffix"])
        manager._get_state("request-1", "test-raising-state")["phase"] = "original"
        decision = manager.on_delta("request-1", "hello", make_context("hello", stream=True))
        assert decision.action == "send"
        assert decision.text == "hello-suffix"
        assert manager.states["request-1"]["test-raising-state"]["phase"] == "original"

    def test_hold_is_preserved_after_later_send_strategy(self):
        manager = OutputFallbackManager(strategies=["test-hold", "test-suffix"])
        decision = manager.on_delta("request-1", "hello", make_context("hello", stream=True))
        assert decision.action == "hold"
        assert decision.text == ""

    def test_truncate_continues_later_strategy(self):
        manager = OutputFallbackManager(
            strategies=["test-truncate", "test-suffix"],
            config={"test-truncate": {"trigger": "stop"}},
        )
        decision = manager.on_delta("request-1", "please stop", make_context("please stop", stream=True))
        assert decision.action == "truncate"
        assert decision.text == "please stop-suffix"

    def test_truncate_keeps_text_after_later_hold(self):
        manager = OutputFallbackManager(
            strategies=["test-truncate", "test-hold"],
            config={"test-truncate": {"trigger": "stop"}},
        )
        decision = manager.on_delta("request-1", "please stop", make_context("please stop", stream=True))
        assert decision.action == "truncate"
        assert decision.text == "please stop"

    def test_finish_flushes_pending_buffer_through_pipeline(self):
        manager = OutputFallbackManager(strategies=["test-hold", "test-suffix", "test-token-observer"])
        manager.on_delta("request-1", "held", make_context("held", stream=True, token_ids=[7]))
        finish_decision = manager.on_finish("request-1", make_context("", stream=True, token_ids=[8]))
        assert finish_decision.action == "flush"
        assert finish_decision.text == "held-suffix"
        assert manager.states["request-1"]["test-token-observer"]["token_count"] == 1

    def test_finish_honors_empty_flush_after_hold(self):
        manager = OutputFallbackManager(strategies=["test-hold-drop"])
        manager.on_delta("request-1", "held", make_context("held", stream=True))
        finish_decision = manager.on_finish("request-1", make_context("", stream=True))
        assert finish_decision.action == "flush"
        assert finish_decision.text == ""

    def test_normalizes_strategy_and_config_names(self):
        manager = OutputFallbackManager(
            strategies=["test_truncate"],
            config={"test_truncate": {"trigger": "stop"}},
        )
        decision = manager.on_delta("request-1", "please stop", make_context("please stop", stream=True))
        assert decision.action == "truncate"
        assert decision.text == "please stop"

    def test_cleanup(self):
        manager = OutputFallbackManager(strategies=["test-suffix"])
        manager._get_state("request-1", "test-suffix")["cache"] = "held"
        manager._get_state("request-2", "test-suffix")["cache"] = "other"
        manager.cleanup("request-1")
        assert "request-1" not in manager.states
        assert manager.states["request-2"]["test-suffix"]["cache"] == "other"


@OutputFallbackManager.register("test-raising-apply", force=True)
class RaisingApplyStrategy(OutputFallbackStrategy):
    name = "test-raising-apply"

    def should_apply(self, text: str, context: OutputFallbackContext) -> bool:
        return True

    def apply(self, text: str, context: OutputFallbackContext) -> str:
        raise RuntimeError("apply boom")


@OutputFallbackManager.register("test-raising-finish", force=True)
class RaisingFinishStrategy(OutputFallbackStrategy):
    name = "test-raising-finish"

    def should_apply(self, text: str, context: OutputFallbackContext) -> bool:
        return True

    def apply(self, text: str, context: OutputFallbackContext) -> str:
        return text

    def on_delta(self, delta_text: str, context: OutputFallbackContext, state: dict) -> StreamFallbackDecision:
        return StreamFallbackDecision(action="hold")

    def on_finish(self, context: OutputFallbackContext, state: dict) -> StreamFallbackDecision:
        raise RuntimeError("finish boom")


class TestGetStrategyNotFound:
    def test_raises_key_error(self):
        import pytest

        with pytest.raises(KeyError, match="not found"):
            OutputFallbackManager.get_strategy("nonexistent-strategy-xyz")


class TestRegisterValidation:
    def test_force_not_bool_raises(self):
        import pytest

        with pytest.raises(TypeError, match="force must be a boolean"):
            OutputFallbackManager.register(name="x", force="yes")

    def test_invalid_name_type_raises(self):
        import pytest

        with pytest.raises(TypeError, match="name must be None"):
            OutputFallbackManager.register(name=123)

    def test_register_with_module_param(self):
        result = OutputFallbackManager.register(name="test-direct-reg", module=ReplaceStrategy)
        assert result is ReplaceStrategy
        assert OutputFallbackManager.get_strategy("test-direct-reg") is ReplaceStrategy

    def test_not_subclass_raises(self):
        import pytest

        with pytest.raises(TypeError, match="must be subclass"):
            OutputFallbackManager._register_strategy(module=str, strategy_name="bad")

    def test_no_force_duplicate_raises(self):
        import pytest

        OutputFallbackManager._register_strategy(module=ReplaceStrategy, strategy_name="dup-test", force=True)
        with pytest.raises(KeyError, match="already registered"):
            OutputFallbackManager._register_strategy(module=SuffixStrategy, strategy_name="dup-test", force=False)

    def test_strategy_name_from_module(self):
        class MyStrat(OutputFallbackStrategy):
            name = "auto-name-test"

            def should_apply(self, text, context):
                return False

            def apply(self, text, context):
                return text

        OutputFallbackManager._register_strategy(module=MyStrat, strategy_name=None, force=True)
        assert OutputFallbackManager.get_strategy("auto-name-test") is MyStrat


class TestApplyException:
    def test_apply_exception_is_swallowed(self):
        manager = OutputFallbackManager(strategies=["test-raising-apply"])
        result = manager.apply("hello", make_context("hello"))
        assert result == "hello"


class TestOnFinishNoBufBranch:
    def test_on_finish_no_buffer_collects_strategy_finish(self):
        manager = OutputFallbackManager(strategies=["test-hold"])
        manager._get_state("r1", "test-hold")["held"] = "buffered"
        decision = manager.on_finish("r1", make_context("", stream=True))
        assert decision.action == "flush"
        assert decision.text == "buffered"

    def test_on_finish_no_buffer_exception_swallowed(self):
        manager = OutputFallbackManager(strategies=["test-raising-finish"])
        decision = manager.on_finish("r1", make_context("", stream=True))
        assert decision.action == "flush"
        assert decision.text == ""


class TestOnFinishBufferExceptions:
    def test_on_finish_buffer_on_delta_exception(self):
        manager = OutputFallbackManager(strategies=["test-raising-state", "test-suffix"])
        manager._buffers["r1"] = "hello"
        decision = manager.on_finish("r1", make_context("", stream=True))
        assert decision.action == "flush"
        assert decision.text == "hello-suffix"

    def test_on_finish_buffer_hold_then_finish_exception(self):
        manager = OutputFallbackManager(strategies=["test-raising-finish"])
        manager._buffers["r1"] = "data"
        decision = manager.on_finish("r1", make_context("", stream=True))
        assert decision.action == "flush"
        assert decision.text == "data"


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

    def test_import_invalid_path_swallowed(self):
        OutputFallbackManager.import_fallback_plugin("/nonexistent/path/plugin.py")
