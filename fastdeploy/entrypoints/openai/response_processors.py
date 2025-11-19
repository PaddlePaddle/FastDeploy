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

from typing import Any, List, Optional

from fastdeploy.entrypoints.openai.usage_calculator import count_tokens
from fastdeploy.input.tokenzier_client import AsyncTokenizerClient, ImageDecodeRequest
from fastdeploy.utils import api_server_logger


class ChatResponseProcessor:
    """
    A decoder class to build multimodal content (text/image) from token_ids.

    Attributes:
        eoi_token_id: Token ID indicating the end of an image (<eoi>).
    """

    def __init__(
        self,
        data_processor,
        enable_mm_output: Optional[bool] = False,
        eoi_token_id: Optional[int] = 101032,
        eos_token_id: Optional[int] = 2,
        decoder_base_url: Optional[str] = None,
    ):
        self.data_processor = data_processor
        self.enable_mm_output = enable_mm_output
        self.eoi_token_id = eoi_token_id
        self.eos_token_id = eos_token_id
        if decoder_base_url is not None:
            self.decoder_client = AsyncTokenizerClient(base_url=decoder_base_url)
        else:
            self.decoder_client = None
        self._mm_buffer: List[Any] = []  # Buffer for accumulating image token_ids
        self._end_image_code_request_output: Optional[Any] = None
        self._multipart_buffer = []

    def enable_multimodal_content(self):
        return self.enable_mm_output

    def accumulate_token_ids(self, request_output):
        decode_type = request_output["outputs"].get("decode_type", 0)

        if not self._multipart_buffer:
            self._multipart_buffer.append({"decode_type": decode_type, "request_output": request_output})
        else:
            last_part = self._multipart_buffer[-1]

            if last_part["decode_type"] == decode_type:
                last_token_ids = last_part["request_output"]["outputs"]["token_ids"]
                last_token_ids.extend(request_output["outputs"]["token_ids"])
                request_output["outputs"]["token_ids"] = last_token_ids
                last_part["request_output"] = request_output
            else:
                self._multipart_buffer.append({"decode_type": decode_type, "request_output": request_output})

    async def process_response_chat(self, request_outputs, stream, enable_thinking, include_stop_str_in_output):
        """
        Process a list of responses into a generator that yields each processed response as it's generated.
        Args:
            request_outputs: The list of outputs to be processed.
            stream: Whether or not to stream the output.
            enable_thinking: Whether or not to show thinking messages.
            include_stop_str_in_output: Whether or not to include stop strings in the output.
        """
        for request_output in request_outputs:
            api_server_logger.debug(f"request_output {request_output}")
            if not self.enable_mm_output:
                yield self.data_processor.process_response_dict(
                    response_dict=request_output,
                    stream=stream,
                    enable_thinking=enable_thinking,
                    include_stop_str_in_output=include_stop_str_in_output,
                )
            elif stream:
                decode_type = request_output["outputs"].get("decode_type", 0)
                token_ids = request_output["outputs"]["token_ids"]
                if decode_type == 0:
                    if self.eoi_token_id and self.eoi_token_id in token_ids:
                        if self._mm_buffer:
                            all_tokens = self._mm_buffer
                            self._mm_buffer = []
                            image = {"type": "image"}
                            if self.decoder_client:
                                req_id = request_output["request_id"]
                                image_ret = await self.decoder_client.decode_image(
                                    request=ImageDecodeRequest(req_id=req_id, data=all_tokens)
                                )
                                if image_ret is not None:
                                    image["url"] = image_ret["http_url"]
                            image_output = self._end_image_code_request_output
                            image_output["outputs"]["multipart"] = [image]
                            image_output["outputs"]["token_ids"] = all_tokens
                            image_output["outputs"]["num_image_tokens"] = count_tokens(all_tokens)
                            yield image_output

                    self.data_processor.process_response_dict(
                        response_dict=request_output,
                        stream=stream,
                        enable_thinking=enable_thinking,
                        include_stop_str_in_output=include_stop_str_in_output,
                    )
                    text = {"type": "text", "text": request_output["outputs"]["text"]}
                    request_output["outputs"]["multipart"] = [text]
                    yield request_output

                elif decode_type == 1:
                    self._mm_buffer.append(token_ids)
                    self._end_image_code_request_output = request_output
            else:
                self.accumulate_token_ids(request_output)
                token_ids = request_output["outputs"]["token_ids"]
                if token_ids[-1] == self.eos_token_id:
                    multipart = []
                    num_image_tokens = 0
                    for part in self._multipart_buffer:
                        if part["decode_type"] == 0:
                            self.data_processor.process_response_dict(
                                response_dict=part["request_output"],
                                stream=False,
                                enable_thinking=enable_thinking,
                                include_stop_str_in_output=include_stop_str_in_output,
                            )
                            text = {"type": "text", "text": part["request_output"]["outputs"]["text"]}
                            multipart.append(text)
                        elif part["decode_type"] == 1:
                            image = {"type": "image"}
                            if self.decoder_client:
                                req_id = part["request_output"]["request_id"]
                                all_tokens = part["request_output"]["outputs"]["token_ids"]
                                num_image_tokens += count_tokens(all_tokens)

                                image_ret = await self.decoder_client.decode_image(
                                    request=ImageDecodeRequest(req_id=req_id, data=all_tokens)
                                )

                                if image_ret is not None:
                                    image["url"] = image_ret["http_url"]
                            multipart.append(image)

                    lasrt_request_output = self._multipart_buffer[-1]["request_output"]
                    lasrt_request_output["outputs"]["multipart"] = multipart
                    lasrt_request_output["outputs"]["num_image_tokens"] = num_image_tokens
                    yield lasrt_request_output
