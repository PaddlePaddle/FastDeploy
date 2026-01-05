import os
from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
from paddleformers.generation import GenerationConfig
from paddleformers.transformers import Llama3Tokenizer, LlamaTokenizer
from paddleformers.trl.llm_utils import get_eos_token_id

from fastdeploy import envs
from fastdeploy.config import ErnieArchitectures
from fastdeploy.input.ernie4_5_tokenizer import Ernie4_5Tokenizer
from fastdeploy.utils import data_processor_logger


class BaseDataProcessor(ABC):
    def __init__(
        self,
        model_name_or_path: str,
        reasoning_parser_obj: Optional[Any] = None,
        tool_parser_obj: Optional[Any] = None,
        architecture: str = False,
    ):
        self.model_name_or_path = model_name_or_path
        self.architecture = architecture
        self.decode_status = dict()
        self.model_status_dict = dict()
        self.tool_parser_dict = dict()

        # 初始化核心组件
        # self._init_config()
        self._init_tokenizer()
        self._init_generation_config()

        # parser
        self.reasoning_parser = reasoning_parser_obj(self.tokenizer) if reasoning_parser_obj else None
        self.tool_parser_obj = tool_parser_obj
        self.eos_token_ids = get_eos_token_id(self.tokenizer, self.generation_config)
        self.eos_token_id_len = len(self.eos_token_ids)
        self.pad_token_id = self.get_pad_id()

    def _init_generation_config(self):
        try:
            self.generation_config = GenerationConfig.from_pretrained(self.model_name_or_path)
        except Exception as e:
            data_processor_logger.warning(
                f"Can't find generation config: {e}, so it will not use generation_config field in the model config"
            )
            self.generation_config = None

    def _init_tokenizer(self):
        """
        load tokenizer

        Returns:
            tokenizer (AutoTokenizer)
        """
        if not ErnieArchitectures.contains_ernie_arch(self.architecture):
            if envs.FD_USE_HF_TOKENIZER:
                from transformers import AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=False)
            else:
                from paddleformers.transformers import AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name_or_path, padding_side="left", use_fast=True
                )
        else:
            vocab_file_names = [
                "tokenizer.model",
                "spm.model",
                "ernie_token_100k.model",
            ]
            for i in range(len(vocab_file_names)):
                if os.path.exists(os.path.join(self.model_name_or_path, vocab_file_names[i])):
                    Ernie4_5Tokenizer.resource_files_names["vocab_file"] = vocab_file_names[i]
                    break
            self.tokenizer = Ernie4_5Tokenizer.from_pretrained(self.model_name_or_path)

    @abstractmethod
    def process_request(self, request, **kwargs):
        """
        Preprocess the request

        Args:
            request (Dict): may contain text and messages fields
            **kwargs: others

        Returns:
            bool: Whether preprocessing is successful
            str: error message
        """
        raise NotImplementedError

    @abstractmethod
    def process_request_dict(self, request, **kwargs):
        """
        Preprocess the request dict

        Args:
            request (Dict): may contain text and messages fields
            **kwargs: others

        Returns:
            bool: Whether preprocessing is successful
            str: error message
        """
        raise NotImplementedError

    @abstractmethod
    def process_response(self, response_dict):
        """
        Preprocess the response

        Args:
            response_dict (Dict): response for engine, contain ids fields

        Returns:
            Dict: response contain text fields
        """
        raise NotImplementedError

    @abstractmethod
    def process_response_dict(self, response_dict):
        """
        Preprocess the response dict

        Args:
            response_dict (Dict): response for engine, contain ids fields

        Returns:
            Dict: response contain text fields
        """
        raise NotImplementedError

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

        if ErnieArchitectures.contains_ernie_arch(self.architecture):
            tokens = self.tokenizer.tokenize(text)
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            return token_ids
        add_special_tokens = kwargs.get("add_special_tokens")
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

        return tokens["input_ids"][0].tolist()

    def messages2ids(self, request, **kwargs):
        """
        Convert multi-turn messages into ID sequences.

        Args:
            messages (List[List[Dict[str, Any]]]): multi-turn messages.

        Returns:
            List[int]: ID sequences
        """
        if self.tokenizer.chat_template is None:
            raise ValueError("This model does not support chat_template.")
        if "add_generation_prompt" not in kwargs:
            kwargs["add_generation_prompt"] = request.get("add_generation_prompt", True)

        spliced_message = self.tokenizer.apply_chat_template(
            request,
            tokenize=False,
            split_special_tokens=False,
            add_special_tokens=False,
            **kwargs,
        )
        request["prompt_tokens"] = spliced_message
        req_id = None
        tokens = self.tokenizer.tokenize(spliced_message)
        if isinstance(request, dict):
            req_id = request.get("request_id", None)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        data_processor_logger.info(f"req_id:{req_id}, tokens:{tokens}, token_ids: {token_ids}")
        return token_ids

    def ids2tokens(self, token_id, task_id):
        """
        token ids to strings

        Args:
            token_ids (List[int]): token ids
                        task_id (str): task id

        Returns:
            List[str]: strings
        """
        if envs.FD_USE_HF_TOKENIZER:
            if task_id not in self.decode_status:
                # history token ids & history token strings & befer decode str
                self.decode_status[task_id] = [[], [], ""]

            previous_token_ids = self.decode_status[task_id][0]
            decode_str = self.tokenizer.batch_decode(
                [previous_token_ids + token_id],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            if isinstance(decode_str, list) and len(decode_str):
                new_str = decode_str[0].replace(self.decode_status[task_id][2], "", 1)
                self.decode_status[task_id][1].append(new_str)
                self.decode_status[task_id][2] = decode_str[0]
            else:
                new_str = ""
            self.decode_status[task_id][0] += token_id
            return new_str
        else:
            if task_id not in self.decode_status:
                # prefix offset & read offset & history token ids & history token strings
                self.decode_status[task_id] = [0, 0, [], ""]

            prefix_offset = self.decode_status[task_id][0]
            read_offset = self.decode_status[task_id][1]
            previous_token_ids = self.decode_status[task_id][2]
            previous_texts = self.decode_status[task_id][3]
            decode_str, prefix_offset, read_offset = self.tokenizer.decode_token(
                previous_token_ids + token_id, prefix_offset, read_offset
            )
            self.decode_status[task_id][0] = prefix_offset
            self.decode_status[task_id][1] = read_offset
            self.decode_status[task_id][2] += token_id
            self.decode_status[task_id][3] += decode_str

            return decode_str, previous_token_ids, previous_texts

    def pad_batch_data(
        self,
        insts,
        pad_id=0,
        return_seq_len=False,
        return_array=True,
        pad_style="right",
    ):
        """Pad the instances to the max sequence length in batch."""
        if len(insts) == 0:
            padded_insts = np.array([[]], dtype=np.int64) if return_array else [[]]
            if return_seq_len:
                seq_len = np.array([], dtype=np.int64) if return_array else []
                return padded_insts, seq_len
            return padded_insts

        max_len = max(map(len, insts))
        if pad_style == "left":
            padded_insts = [[pad_id] * (max_len - len(inst)) + list(inst) for inst in insts]
        else:
            padded_insts = [list(inst) + [pad_id] * (max_len - len(inst)) for inst in insts]
        if return_array:
            padded_insts = np.array(padded_insts, dtype=np.int64).reshape([-1, max_len])

        if return_seq_len:
            seq_len = [len(inst) for inst in insts]
            if return_array:
                seq_len = np.array(seq_len, dtype=np.int64).reshape(-1, 1)
            return padded_insts, seq_len
        return padded_insts

    def update_stop_seq(self, stop_sequences):
        """
        Update stop sequences from request.
        """
        stop_seqs = []
        if isinstance(stop_sequences, str):
            stop_sequences = [stop_sequences]
        for seq in stop_sequences:
            if seq != self.tokenizer.eos_token_id:
                stop_seqs.append(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(seq)))
        stop_seqs, stop_seqs_len = self.pad_batch_data(stop_seqs, pad_id=-1, return_seq_len=True, return_array=False)
        data_processor_logger.debug(f"processed stop_seqs: {stop_seqs}, {stop_seqs_len}")
        return stop_seqs, stop_seqs_len

    def update_bad_words(self, bad_words, bad_words_token_ids):
        """Support bad words"""

        token_ids = bad_words_token_ids

        if token_ids is None:
            token_ids = []
        for bad_word in bad_words:
            # To prohibit words both at the beginning
            # and in the middle of text
            # (related to add_prefix_space tokenizer parameter)
            for add_prefix_space in [False, True]:
                prefix = " " if add_prefix_space else ""
                prompt = prefix + bad_word.lstrip()
                prompt_token_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(prompt))
                data_processor_logger.debug(f"processed bad_words: {prompt}, {prompt_token_ids}")

                if len(prompt_token_ids) != 1:
                    if not add_prefix_space:
                        data_processor_logger.warning(
                            f"Skip bad_words: <{prompt}>."
                            f"Bad words should be a single token."
                            f"Got tokens: {prompt_token_ids}."
                        )
                    continue

                if prompt_token_ids[0] > self.tokenizer.vocab_size:
                    if not add_prefix_space:
                        data_processor_logger.warning(
                            f"Skip bad_words: <{prompt}>."
                            f"All token id values should be satisfying:"
                            f" 0 <= token_id < {self.tokenizer.vocab_size}."
                            f"Got token: {prompt_token_ids}."
                        )
                    continue

                if prompt_token_ids not in token_ids:
                    token_ids.extend(prompt_token_ids)
        return token_ids

    def get_pad_id(self):
        """
        get pad_token_id, if not pad_token_id, use eos_token

        Returns:
            int: pad_token_id
        """
        if isinstance(self.tokenizer, (LlamaTokenizer, Llama3Tokenizer)) and not self.tokenizer.pad_token_id:
            return self.tokenizer.eos_token
        return self.tokenizer.pad_token_id

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
