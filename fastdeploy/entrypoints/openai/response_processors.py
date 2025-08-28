from typing import Any, List, Optional

from fastdeploy.input.tokenzier_client import AsyncTokenizerClient


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
        eoi_token_id: Optional[int] = None,
        decoder_base_url: Optional[str] = None,
    ):
        self.data_processor = data_processor
        self.enable_mm_output = enable_mm_output
        self.eoi_token_id = eoi_token_id
        if decoder_base_url is not None:
            self.decoder_client = AsyncTokenizerClient(base_url=decoder_base_url)
        self._mm_buffer: List[Any] = []  # Buffer for accumulating image token_ids
        self._end_image_code_request_output: Optional[Any] = None

    def enable_multimodal_content(self):
        return self.enable_mm_output

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
            if not self.enable_mm_output:
                yield self.data_processor.process_response_dict(
                    response_dict=request_output,
                    stream=stream,
                    enable_thinking=enable_thinking,
                    include_stop_str_in_output=include_stop_str_in_output,
                )
            else:
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
                                    request={"req_id": req_id, "data": all_tokens}
                                )
                                image["url"] = image_ret["http_url"]
                            image_output = self._end_image_code_request_output
                            image_output["outputs"]["multipart"] = [image]
                            image_output["outputs"]["token_ids"] = all_tokens
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
                    self._mm_buffer.extend(token_ids)
                    self._end_image_code_request_output = request_output
