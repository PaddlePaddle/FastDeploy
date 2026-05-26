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

from fastdeploy.output.fallback import OutputFallbackContext, OutputFallbackManager
from fastdeploy.output.fallback.markdown_bold_colon import (
    MarkdownBoldColonFallbackStrategy,
)


def make_context(text: str, stream: bool = False) -> OutputFallbackContext:
    return OutputFallbackContext(
        request=None,
        request_id="test-request::n::0",
        choice_index=0,
        stream=stream,
        output={"text": text},
        full_text=None if stream else text,
        delta_text=text if stream else None,
    )


def test_markdown_bold_colon_fallback_apply():
    strategy = MarkdownBoldColonFallbackStrategy()
    cases = {
        "**内容：**": "**内容**：",
        "**内容:**": "**内容**:",
        "**内容**：": "**内容**：",
        "普通文本": "普通文本",
        "**标题：** 内容 **小节:** 说明": "**标题**： 内容 **小节**: 说明",
    }
    for source, expected in cases.items():
        result = strategy.apply(source, make_context(source))
        assert result == expected


def test_output_fallback_manager_apply():
    manager = OutputFallbackManager(strategies=["markdown-bold-colon"])
    text = "**标题：** 这是内容"
    result = manager.apply(text, make_context(text))
    assert result == "**标题**： 这是内容"


def test_output_fallback_manager_on_delta_complete_bold():
    manager = OutputFallbackManager(strategies=["markdown-bold-colon"])
    decision = manager.on_delta("request-1", 0, "**标题：**", make_context("**标题：**", stream=True))
    assert decision.action == "send"
    assert decision.text == "**标题**："


def test_output_fallback_manager_on_delta_open_bold_does_not_hold_without_colon_suffix():
    manager = OutputFallbackManager(strategies=["markdown-bold-colon"])
    decision = manager.on_delta("request-1", 0, "**标题", make_context("**标题", stream=True))
    assert decision.action == "send"
    assert decision.text == "**标题"


def test_output_fallback_manager_on_delta_swaps_colon_before_closing_bold_across_delta():
    manager = OutputFallbackManager(strategies=["markdown-bold-colon"])
    first_decision = manager.on_delta("request-1", 0, "**标题", make_context("**标题", stream=True))
    assert first_decision.action == "send"
    assert first_decision.text == "**标题"

    second_decision = manager.on_delta("request-1", 0, "：** 后续", make_context("：** 后续", stream=True))
    assert second_decision.action == "send"
    assert second_decision.text == "**： 后续"


def test_output_fallback_manager_on_delta_holds_colon_suffix_inside_bold():
    manager = OutputFallbackManager(strategies=["markdown-bold-colon"])
    first_decision = manager.on_delta("request-1", 0, "**标题：", make_context("**标题：", stream=True))
    assert first_decision.action == "send"
    assert first_decision.text == "**"

    second_decision = manager.on_delta("request-1", 0, "** 后续", make_context("** 后续", stream=True))
    assert second_decision.action == "send"
    assert second_decision.text == "标题**： 后续"


def test_output_fallback_manager_on_delta_swaps_cached_colon_with_next_bold():
    manager = OutputFallbackManager(strategies=["markdown-bold-colon"])
    first_decision = manager.on_delta("request-1", 0, "前缀 **标题：", make_context("前缀 **标题：", stream=True))
    assert first_decision.action == "send"
    assert first_decision.text == "前缀 **"

    second_decision = manager.on_delta("request-1", 0, "**", make_context("**", stream=True))
    assert second_decision.action == "send"
    assert second_decision.text == "标题**："


def test_output_fallback_manager_on_delta_releases_cached_colon_when_next_delta_has_prefix_before_bold():
    manager = OutputFallbackManager(strategies=["markdown-bold-colon"])
    first_decision = manager.on_delta("request-1", 0, "**标题：", make_context("**标题：", stream=True))
    assert first_decision.action == "send"
    assert first_decision.text == "**"

    second_decision = manager.on_delta("request-1", 0, " 补充** 后续", make_context(" 补充** 后续", stream=True))
    assert second_decision.action == "send"
    assert second_decision.text == "标题： 补充** 后续"


def test_output_fallback_manager_on_delta_releases_cached_colon_when_next_delta_has_no_bold():
    manager = OutputFallbackManager(strategies=["markdown-bold-colon"])
    first_decision = manager.on_delta("request-1", 0, "**标题：", make_context("**标题：", stream=True))
    assert first_decision.action == "send"
    assert first_decision.text == "**"

    second_decision = manager.on_delta("request-1", 0, " 后续", make_context(" 后续", stream=True))
    assert second_decision.action == "send"
    assert second_decision.text == "标题： 后续"


def test_output_fallback_manager_on_finish_flushes_cached_colon():
    manager = OutputFallbackManager(strategies=["markdown-bold-colon"])
    first_decision = manager.on_delta("request-1", 0, "**标题：", make_context("**标题：", stream=True))
    assert first_decision.action == "send"
    assert first_decision.text == "**"

    finish_decision = manager.on_finish("request-1", 0, make_context("", stream=True))
    assert finish_decision.action == "flush"
    assert finish_decision.text == "标题："


def test_output_fallback_manager_cleanup():
    manager = OutputFallbackManager(strategies=["markdown-bold-colon"])
    manager._get_state("request-1", 0, "markdown-bold-colon")["gfm_cache"] = "held"
    manager._get_state("request-2", 0, "markdown-bold-colon")["gfm_cache"] = "other"
    manager.cleanup("request-1")
    assert "request-1" not in manager.states
    assert manager.states["request-2"][(0, "markdown-bold-colon")]["gfm_cache"] == "other"


def test_output_fallback_plugin_import(tmp_path: Path):
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
