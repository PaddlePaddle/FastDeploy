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
        self.think_end_token = "</think>"
        self.response_start_token = "<response>"
        self.response_end_token = "</response>"
        self.tool_call_start_token = "<tool_call>"
        self.tool_call_end_token = "</tool_call>"

        if not self.model_tokenizer:
            raise ValueError("The model tokenizer must be passed to the ReasoningParser constructor.")

        self.think_end_token_id = self.vocab.get("</think>")
        if self.think_end_token_id is None:
            raise RuntimeError("Could not find think end token id in tokenizer vocabulary")
        self.tool_call_start_token_id = self.vocab.get("<tool_call>")

    def extract_reasoning_content_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> Union[DeltaMessage, None]:
        # Ignore the single </think> token
        if len(delta_token_ids) == 1 and delta_token_ids[0] == self.think_end_token_id:
            return None

        # --- Thinking stage handling ---
        if not previous_text.endswith(self.think_end_token) and self.think_end_token not in previous_text:
            # If delta is </think>, stop thinking, do not return
            if delta_text.startswith(self.think_end_token):
                return None
            # Otherwise, return thinking content (keep \n as-is)
            return DeltaMessage(reasoning_content=delta_text)

        # --- After thinking ends, check tool_call or response ---
        remaining_text = previous_text + delta_text
        after_think = remaining_text[remaining_text.find(self.think_end_token) + len(self.think_end_token) :]
        after_think = after_think.lstrip("\n")

        # Handle tool_call case: skip it
        if after_think.startswith(self.tool_call_start_token):
            return None

        # Handle response case
        if after_think.startswith(self.response_start_token) and self.response_end_token not in after_think:
            # Do not return when <response> tag itself appears
            if delta_text == self.response_start_token or delta_text == self.response_end_token:
                return None
            return DeltaMessage(content=delta_text)

        # Default case: return nothing
        return None

    def extract_reasoning_content(self, model_output: str, request: ChatCompletionRequest) -> Tuple[str, str]:
        reasoning_content = ""
        response_content = ""

        think_end_pos = model_output.find(self.think_end_token)
        if think_end_pos != -1:
            reasoning_content = model_output[:think_end_pos]

            remaining = model_output[think_end_pos + len(self.think_end_token) :]

            # find <response> or <tool>
            response_pos = remaining.find(self.response_start_token)
            tool_pos = remaining.find(self.tool_call_start_token)

            # <response> first
            if response_pos != -1 and (tool_pos == -1 or response_pos < tool_pos):
                # The content after the response_start position
                remaining_response = remaining[response_pos + len(self.response_start_token) :]
                response_end_pos = remaining_response.find(self.response_end_token)
                if response_end_pos != -1:
                    response_content = remaining_response[:response_end_pos]
                else:
                    response_content = remaining_response
            # The content after the response_start position is tool_call
        else:
            reasoning_content = model_output
            response_content = ""

        return reasoning_content, response_content
