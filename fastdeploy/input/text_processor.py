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

from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Mapping

from paddleformers.generation import GenerationConfig

from fastdeploy import envs
from fastdeploy.input.base_processor import BaseTextProcessor
from fastdeploy.utils import data_processor_logger


class BaseDataProcessor(ABC):
    """base class for data processor"""

    def __init__(self):
        """
        Returns:
            None
        """
        self.tokenizer = self._load_tokenizer()
        self.tokenizer.bos_token_id = self.tokenizer._convert_token_to_id(self.tokenizer.bos_token)
        self.tokenizer.cls_token_id = self.tokenizer._convert_token_to_id(self.tokenizer.cls_token)
        self.tokenizer.sep_token_id = self.tokenizer._convert_token_to_id(self.tokenizer.sep_token)
        self.tokenizer.eos_token_id = self.tokenizer._convert_token_to_id(self.tokenizer.eos_token)
        self.tokenizer.mask_token_id = self.tokenizer._convert_token_to_id(self.tokenizer.mask_token)
        data_processor_logger.info(
            (
                f"tokenizer information: bos_token is {self.tokenizer.bos_token}, {self.tokenizer.bos_token_id}, ",
                f"cls_token is {self.tokenizer.cls_token}, {self.tokenizer.cls_token_id}, "
                f"sep_token is {self.tokenizer.sep_token}, {self.tokenizer.sep_token_id}, "
                f"eos_token is {self.tokenizer.eos_token}, {self.tokenizer.eos_token_id}, "
                f"mask_token is {self.tokenizer.mask_token}, {self.tokenizer.mask_token_id}",
            )
        )
        self._tokenize_cache = OrderedDict()
        self._tokenize_cache_capacity = 128

    def _apply_default_parameters(self, request):
        """
        Apply default value for parameters in request
        """

        def set_value(req, key, value):
            value = getattr(self.generation_config, key, value)
            if isinstance(req, dict):
                if key not in req or req[key] is None:
                    req[key] = value
            else:
                if req.get(key) is None:
                    req.set(key, value)

        set_value(request, "top_p", 0.7)
        set_value(request, "temperature", 1.0)
        set_value(request, "repetition_penalty", 1.0)
        set_value(request, "frequency_penalty", 0.0)
        set_value(request, "presence_penalty", 0.0)
        return request

    def text2ids(self, text, max_model_len=None):
        """
        text to token ids

        Args:
            text (str): text

        Returns:
            List[int]: token ids list
        """
        raise NotImplementedError

    def encode_with_cache(self, text, max_model_len=None, add_special_tokens=False):
        """
        Encode text into token ids with a small LRU cache.
        """
        if not hasattr(self, "_tokenize_cache"):
            self._tokenize_cache = OrderedDict()
            self._tokenize_cache_capacity = getattr(self, "_tokenize_cache_capacity", 128)
        key = (text, bool(add_special_tokens))
        cached = self._tokenize_cache.get(key)
        if cached is not None:
            self._tokenize_cache.move_to_end(key)
            return cached
        token_ids = self.text2ids(text, max_model_len, add_special_tokens=add_special_tokens)
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        elif not isinstance(token_ids, list):
            token_ids = list(token_ids)
        self._tokenize_cache[key] = token_ids
        if len(self._tokenize_cache) > self._tokenize_cache_capacity:
            self._tokenize_cache.popitem(last=False)
        return token_ids

    def _encode_literal_text_with_cache(self, text):
        if not hasattr(self, "_tokenize_cache"):
            self._tokenize_cache = OrderedDict()
            self._tokenize_cache_capacity = getattr(self, "_tokenize_cache_capacity", 128)
        key = ("literal_text", text)
        cached = self._tokenize_cache.get(key)
        if cached is not None:
            self._tokenize_cache.move_to_end(key)
            return cached
        token_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        elif not isinstance(token_ids, list):
            token_ids = list(token_ids)
        self._tokenize_cache[key] = token_ids
        if len(self._tokenize_cache) > self._tokenize_cache_capacity:
            self._tokenize_cache.popitem(last=False)
        return token_ids

    def messages2ids(self, messages):
        """
        Convert multi-turn messages into ID sequences.

        Args:
            messages (List[List[Dict[str, Any]]]): multi-turn messages.

        Returns:
            List[int]: ID sequences
        """
        raise NotImplementedError

    def _get_think_token_ids(self):
        think_token_ids = getattr(self, "_think_token_ids", None)
        if think_token_ids is not None:
            return think_token_ids
        tokenizer = getattr(self, "tokenizer", None)
        vocab = tokenizer.get_vocab() if tokenizer is not None else {}
        think_start_id = vocab.get("<think>", -1)
        think_end_id = vocab.get("</think>", -1)
        self._think_token_ids = (think_start_id, think_end_id)
        return self._think_token_ids

    def _prepare_think_stop_sentence(self, logits_processors_args, max_model_len=None):
        if not isinstance(logits_processors_args, dict):
            return logits_processors_args
        think_stop_sentence = logits_processors_args.get("think_stop_sentence")
        if isinstance(think_stop_sentence, str) and think_stop_sentence:
            sentence_token_ids = self._encode_literal_text_with_cache(think_stop_sentence)
            logits_processors_args["think_stop_sentence_token_ids"] = sentence_token_ids
            logits_processors_args.pop("think_stop_sentence", None)
        return logits_processors_args

    def _update_thinking_prompt_state(self, prompt_token_ids, logits_processors_args):
        if not isinstance(logits_processors_args, dict):
            return logits_processors_args
        thinking_budget = logits_processors_args.get("thinking_budget")
        if thinking_budget is None or not isinstance(thinking_budget, int) or thinking_budget < 0:
            return logits_processors_args
        if logits_processors_args.get("think_prompt_checked"):
            return logits_processors_args
        if prompt_token_ids is None:
            return logits_processors_args
        token_len = getattr(prompt_token_ids, "size", None) or len(prompt_token_ids)
        if token_len == 0:
            return logits_processors_args
        think_start_id, think_end_id = self._get_think_token_ids()
        if think_start_id < 0 or think_end_id < 0:
            return logits_processors_args

        if hasattr(prompt_token_ids, "tolist"):
            token_list = prompt_token_ids.tolist()
        else:
            token_list = list(prompt_token_ids)

        started = False
        ended = False
        tokens_after_start = 0
        last_token_id = None
        in_thinking = False
        for token_id in token_list:
            if token_id == think_start_id:
                started = True
                ended = False
                in_thinking = True
            elif token_id == think_end_id and in_thinking:
                ended = True
                in_thinking = False
        if started and token_list:
            # Align with operator-level reasoning_max_tokens: prompt-side tokens
            # inside <think> do not consume thinking budget.
            last_token_id = int(token_list[-1])

        logits_processors_args["think_prompt_checked"] = True
        logits_processors_args["think_prompt_started"] = started
        logits_processors_args["think_prompt_ended"] = ended
        logits_processors_args["think_prompt_tokens_after_start"] = tokens_after_start
        if last_token_id is not None:
            logits_processors_args["think_prompt_last_token_id"] = last_token_id
        else:
            logits_processors_args.pop("think_prompt_last_token_id", None)
        return logits_processors_args

    def ids2tokens(self, token_id, task_id=None):
        """
        token ids to strings

        Args:
            token_id (List[int]): token id
                        task_id (str): task id

        Returns:
            List[str]: strings
        """
        raise NotImplementedError

    @abstractmethod
    def _load_tokenizer(self):
        """
        load tokenizer

        Returns:
            tokenizer (AutoTokenizer)
        """
        raise NotImplementedError

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
    ) -> Mapping[str, int]:
        """
        Return the maximum number of tokens per item for each modality.
        """
        return None


class DataProcessor(BaseTextProcessor):
    def __init__(self, model_name_or_path, reasoning_parser_obj=None, tool_parser_obj=None):
        """
            Initializes the DecodeStatus object.

        Args:
            model_name_or_path (str): The name or path of the pre-trained model to be loaded.
                Can also be a path to a directory containing the pre-trained model file.

        Returns:
            None.

        Raises:
            None.
        """

        self.model_name_or_path = model_name_or_path

        # Generation config
        try:
            self.generation_config = GenerationConfig.from_pretrained(self.model_name_or_path)
        except Exception as e:
            data_processor_logger.warning(
                f"Can't find generation config: {e}, so it will not use generation_config field in the model config"
            )
            self.generation_config = None

        super().__init__()
        self.tokenizer = self._load_tokenizer()
        data_processor_logger.info(
            f"tokenizer information: bos_token is {self.tokenizer.bos_token}, {self.tokenizer.bos_token_id}, \
                                eos_token is {self.tokenizer.eos_token}, {self.tokenizer.eos_token_id} "
        )

        try:
            from paddleformers.trl.llm_utils import get_eos_token_id
        except Exception:
            from paddleformers.cli.utils.llm_utils import get_eos_token_id

        self.eos_token_ids = get_eos_token_id(self.tokenizer, self.generation_config)
        data_processor_logger.info(
            f"The eos_token_ids obtained by merging tokenizer and generation_config is {self.eos_token_ids}"
        )
        self.eos_token_id_len = len(self.eos_token_ids)
        self.pad_token_id = self.get_pad_id()
        self._init_parsers(reasoning_parser_obj, tool_parser_obj)
        self.tokenizer.pad_token_id = self.pad_token_id

    def process_logprob_response(self, token_ids, **kwargs):
        full_text = self.tokenizer.decode(token_ids, **kwargs)
        return full_text

    def text2ids(self, text, max_model_len, **kwargs):
        """
        text to token ids

        Args:
            text (str): text

        Returns:
            List[int]: token ids list
        """

        add_special_tokens = kwargs.get("add_special_tokens", False)
        if envs.FD_USE_HF_TOKENIZER:
            tokens = self.tokenizer(
                text,
                return_tensors="np",
                padding=True,
                truncation=True,
            )
        else:
            text = [text] if isinstance(text, str) else text

            tokens = self.tokenizer(
                text,
                return_tensors="np",
                padding=True,
                truncation=True,
                max_length=max_model_len,
                add_special_tokens=add_special_tokens,
            )

        return tokens["input_ids"][0]

    def _load_tokenizer(self):
        """
        load tokenizer

        Returns:
            tokenizer (AutoTokenizer)
        """
        if envs.FD_USE_HF_TOKENIZER:
            from transformers import AutoTokenizer

            return AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=False)
        else:
            from paddleformers.transformers import AutoTokenizer

            return AutoTokenizer.from_pretrained(self.model_name_or_path, padding_side="left", use_fast=True)


class TextProcessor(BaseTextProcessor):
    """Unified text processor for both auto and ernie4_5 tokenizer types.

    Replaces ``DataProcessor`` (tokenizer_type="auto") and
    ``Ernie4_5Processor`` (tokenizer_type="ernie4_5") with a single class.

    Args:
        model_name_or_path: Path or name of the pretrained model.
        tokenizer_type: ``"auto"`` (default) or ``"ernie4_5"``.
        reasoning_parser_obj: Optional reasoning-parser class.
        tool_parser_obj: Optional tool-parser class.
    """

    def __init__(
        self,
        model_name_or_path: str,
        tokenizer_type: str = "auto",
        reasoning_parser_obj=None,
        tool_parser_obj=None,
    ):
        self.model_name_or_path = model_name_or_path
        self.tokenizer_type = tokenizer_type

        # Generation config — both types use the same API
        try:
            self.generation_config = GenerationConfig.from_pretrained(self.model_name_or_path)
        except Exception as e:
            data_processor_logger.warning(
                f"Can't find generation config: {e}, so it will not use generation_config field in the model config"
            )
            self.generation_config = None

        # Response-handling state — delegated to BaseTextProcessor.__init__
        super().__init__()

        # Tokenizer
        self.tokenizer = self._load_tokenizer()
        data_processor_logger.info(
            f"tokenizer information: bos_token is {self.tokenizer.bos_token}, "
            f"{self.tokenizer.bos_token_id}, "
            f"eos_token is {self.tokenizer.eos_token}, {self.tokenizer.eos_token_id}"
        )

        # EOS tokens — both types use the same paddleformers utility
        try:
            from paddleformers.trl.llm_utils import get_eos_token_id
        except Exception:
            from paddleformers.cli.utils.llm_utils import get_eos_token_id

        self.eos_token_ids = get_eos_token_id(self.tokenizer, self.generation_config)
        data_processor_logger.info(
            f"The eos_token_ids obtained by merging tokenizer and generation_config is {self.eos_token_ids}"
        )
        self.eos_token_id_len = len(self.eos_token_ids)
        self.pad_token_id = self.get_pad_id()
        self.tokenizer.pad_token_id = self.pad_token_id
        self._init_parsers(reasoning_parser_obj, tool_parser_obj)

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def _load_tokenizer(self):
        if self.tokenizer_type == "ernie4_5":
            return self._load_ernie4_5_tokenizer()
        return self._load_auto_tokenizer()

    def _load_auto_tokenizer(self):
        if envs.FD_USE_HF_TOKENIZER:
            from transformers import AutoTokenizer

            return AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=False)
        else:
            from paddleformers.transformers import AutoTokenizer

            return AutoTokenizer.from_pretrained(self.model_name_or_path, padding_side="left", use_fast=True)

    def _load_ernie4_5_tokenizer(self):
        import os

        from fastdeploy.input.ernie4_5_tokenizer import Ernie4_5Tokenizer

        vocab_file_names = ["tokenizer.model", "spm.model", "ernie_token_100k.model"]
        for name in vocab_file_names:
            if os.path.exists(os.path.join(self.model_name_or_path, name)):
                Ernie4_5Tokenizer.resource_files_names["vocab_file"] = name
                break
        return Ernie4_5Tokenizer.from_pretrained(self.model_name_or_path)

    def text2ids(self, text, max_model_len=None, **kwargs):
        if self.tokenizer_type == "ernie4_5":
            return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        # auto path
        add_special_tokens = kwargs.get("add_special_tokens", False)
        if envs.FD_USE_HF_TOKENIZER:
            tokens = self.tokenizer(text, return_tensors="np", padding=True, truncation=True)
        else:
            text_input = [text] if isinstance(text, str) else text
            tokens = self.tokenizer(
                text_input,
                return_tensors="np",
                padding=True,
                truncation=True,
                max_length=max_model_len,
                add_special_tokens=add_special_tokens,
            )
        return tokens["input_ids"][0]

    def process_logprob_response(self, token_ids, **kwargs):
        return self.tokenizer.decode(token_ids, **kwargs)
