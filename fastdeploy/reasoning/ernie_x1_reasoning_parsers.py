from collections.abc import Sequence
from typing import Tuple, Union

from fastdeploy.entrypoints.openai.protocol import ChatCompletionRequest, DeltaMessage
from fastdeploy.reasoning import ReasoningParser, ReasoningParserManager


@ReasoningParserManager.register_module("ernie-x1")
class ErnieX1ReasoningParser(ReasoningParser):
    """
    Reasoning parser for ernie-x1 model with stricter boundary checking.

    Unified rules:
    - Do not strip newline before </think>
    - Do not strip newline after <response>
    - Do not strip newline before </response>
    """

    def __init__(self, tokenizer):
        super().__init__(tokenizer)

        # 定义所有需要检查的token
        token_definitions = {
            "think_start_token": "<think>",
            "think_end_token": "</think>",
            "response_start_token": "<response>",
            "response_end_token": "</response>",
            "tool_call_start_token": "<tool_call>",
            "tool_call_end_token": "</tool_call>",
        }

        if not self.model_tokenizer:
            raise ValueError("The model tokenizer must be passed to the ReasoningParser constructor.")

        missing_tokens = []
        for name, token_value in token_definitions.items():
            setattr(self, name, token_value)
            token_id = self.vocab.get(token_value)
            setattr(self, f"{name}_id", token_id)
            if token_id is None:
                missing_tokens.append(token_value)

        if missing_tokens:
            raise RuntimeError(
                f"ernie x1 reasoning parser could not find the following token ids in tokenizer vocabulary: {', '.join(missing_tokens)}"
            )

        self.token_status_mapping = {
            self.think_start_token_id: "think_start",
            self.think_end_token_id: "think_end",
            self.response_start_token_id: "response_start",
            self.response_end_token_id: "response_end",
            self.tool_call_start_token_id: "tool_call_start",
            self.tool_call_end_token_id: "tool_call_end",
        }

    def find_last_special_token(self, prompt_token_ids: list[int]) -> int:
        for i in range(len(prompt_token_ids) - 1, -1, -1):
            if prompt_token_ids[i] in self.token_status_mapping:
                return prompt_token_ids[i]
        return -1

    def get_model_status(self, prompt_token_ids: list[int]):
        special_token_id = self.find_last_special_token(prompt_token_ids)

        if special_token_id == -1:
            return "think_start"

        return self.token_status_mapping[special_token_id]

    def extract_reasoning_content_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        model_status: str,
    ) -> Union[DeltaMessage, None]:

        if len(delta_token_ids) == 1 and delta_token_ids[0] in [
            self.think_end_token_id,
            self.response_start_token_id,
            self.response_end_token_id,
            self.tool_call_start_token_id,
            self.tool_call_end_token_id,
        ]:
            return None

        if model_status == "think_start":
            if self.think_end_token in delta_text:
                response_content = ""
                end_index = delta_text.find(self.think_end_token)
                reasoning_content = delta_text[:end_index]
                response_start_pos = delta_text.find(self.response_start_token)
                if response_start_pos != -1:
                    response_content = self._extract_response_content(
                        delta_text[response_start_pos + len(self.response_start_token) :]
                    )
                return DeltaMessage(reasoning_content=reasoning_content, content=response_content)
            elif self.think_end_token in previous_text:
                if self.response_start_token in previous_text and self.response_end_token not in previous_text:
                    return DeltaMessage(content=delta_text)
            else:
                return DeltaMessage(reasoning_content=delta_text)
        elif model_status == "think_end":
            if self.response_start_token in previous_text and self.response_end_token not in previous_text:
                return DeltaMessage(content=delta_text)
        elif model_status == "response_start":
            if self.response_end_token not in previous_text:
                return DeltaMessage(content=delta_text)

        return None

    def extract_reasoning_content(
        self, model_output: str, request: ChatCompletionRequest, model_status: str
    ) -> Tuple[str, str]:
        """
        优化版解析器。保留推理和响应内容中的换行符，
        仅删除闭合标签前的单个换行符。
        """
        reasoning_content = ""
        response_content = ""

        if model_status in ["think_start", "think_end"]:
            if model_status == "think_start":
                think_end_pos = model_output.find(self.think_end_token)
                if think_end_pos != -1:
                    reasoning_content = model_output[:think_end_pos]
                    remaining = model_output[think_end_pos + len(self.think_end_token) :].lstrip("\n")
                else:
                    reasoning_content = model_output
                    remaining = ""
            else:
                remaining = model_output.lstrip("\n")

            response_start_pos = remaining.find(self.response_start_token)
            if response_start_pos != -1:
                response_content = self._extract_response_content(
                    remaining[response_start_pos + len(self.response_start_token) :]
                )

        elif model_status == "response_start":
            response_content = self._extract_response_content(model_output)

        return reasoning_content, response_content

    def _extract_response_content(self, remaining: str) -> str:
        """
        Extracts response content, ensuring that the last newline before
        the </response> tag is removed.
        """
        response_end_pos = remaining.find(self.response_end_token)
        if response_end_pos != -1:
            return remaining[:response_end_pos]
        return remaining
