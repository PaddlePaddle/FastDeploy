"""
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

# cipher_token=WjI1fQOvhN  # do not edit this line

import os
import re
from itertools import product
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import paddle
import sentencepiece as spm
from paddleformers.transformers import AddedToken, PretrainedTokenizer
from paddleformers.transformers.tokenizer_utils_base import PaddingStrategy, TextInput
from paddleformers.utils.log import logger


class ErnieBotTokenizer(PretrainedTokenizer):
    """
    一个更好用的 `ErnieBotToknizer`，
    能 encode 目前 sft/ppo 阶段的特殊token，也支持多模态。
    """

    resource_files_names = {"vocab_file": "spm.model"}
    pretrained_resource_files_map = {"vocab_file": {"ernie-bot-10b": None}}
    pretrained_init_configuration = {"ernie-bot-10b": {}}
    model_input_names = ["input_ids", "position_ids", "attention_mask", "labels"]
    padding_side = "right"

    def __init__(
        self,
        vocab_file,
        bos_token="<s>",
        cls_token="<cls>",
        eos_token="</s>",
        mask_token="<mask:0>",
        pad_token="<pad>",
        sep_token="<sep>",
        unk_token="<unk>",
        additional_special_tokens=None,
        verbose=False,
        **kwargs,
    ):
        """doc"""
        if additional_special_tokens is None:
            additional_special_tokens = ["<mask:1>", "<mask:7>"]
        super().__init__(
            bos_token=bos_token,
            cls_token=cls_token,
            eos_token=eos_token,
            mask_token=mask_token,
            pad_token=pad_token,
            sep_token=sep_token,
            unk_token=unk_token,
            additional_special_tokens=additional_special_tokens,
            verbose=False,
            **kwargs,
        )
        self.vocab_file = vocab_file
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(vocab_file)
        # pre-process map-type all spec token for decode accelerate.

    @property
    def space_token(self):
        """doc"""
        return "<mask:1>"

    @property
    def space_token_id(self):
        """doc"""
        return self.sp_model.piece_to_id("<mask:1>")

    @property
    def gend_token(self):
        """doc"""
        return "<mask:7>"

    @property
    def gend_token_id(self):
        """doc"""
        return self.sp_model.piece_to_id("<mask:7>")

    @property
    def im_start_id(self):
        """doc"""
        return self.sp_model.piece_to_id("<|im_start|>")

    @property
    def im_end_id(self):
        """doc"""
        return self.sp_model.piece_to_id("<|im_end|>")

    @property
    def vocab_size(self):
        """doc"""
        return self.sp_model.vocab_size()

    def get_vocab(self):
        """doc"""
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text):
        """doc"""
        return self.sp_model.encode_as_pieces(text)

    def _convert_token_to_id(self, token):
        """doc"""
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, id):
        """doc"""
        return self.sp_model.id_to_piece(id)

    def spec_init(self):
        """初始化special tokens"""
        if not hasattr(self, "all_spec_tok"):
            self.all_spec_tok = set(self.all_special_tokens)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        self.spec_init()
        current_sub_tokens = []
        out_string = ""
        # prev_is_special = False
        for token in tokens:
            # make sure that special tokens are not decoded using sentencepiece model
            if token in self.all_spec_tok:
                # if not prev_is_special:
                #     out_string += " "
                out_string += self.sp_model.decode(current_sub_tokens) + token
                # prev_is_special = True

                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
                # prev_is_special = False
        out_string += self.sp_model.decode(current_sub_tokens)
        return out_string  # .strip()

    def prepare_for_model(self, *args, **kwargs):
        """doc"""
        if "add_special_tokens" in kwargs:
            kwargs.pop("add_special_tokens")
            # logger.warning(f'ErnieBotTokenizer v2 does not support `add_special_tokens`')
        return super().prepare_for_model(*args, **kwargs)

    def save_vocabulary(self, save_directory, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the vocabulary and special tokens file to a directory.
        Args:
            save_directory (`str`):
                The directory in which to save the vocabulary.
        Returns:
            `Tuple(str)`: Paths to the files saved.
        """
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + self.resource_files_names["vocab_file"],
        )
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)
        return (out_vocab_file,)

    def tokenize(self, text: TextInput, **kwargs) -> List[str]:
        """
        Converts a string in a sequence of tokens, using the tokenizer.

        Split in words for word-based vocabulary or sub-words for sub-word-based vocabularies
        (BPE/SentencePieces/WordPieces). Takes care of added tokens.

        Args:
            text (`str`):
                The sequence to be encoded.
            **kwargs (additional keyword arguments):
                Passed along to the model-specific `prepare_for_tokenization` preprocessing method.

        Returns:
            `List[str]`: The list of tokens.
        """
        # Simple mapping string => AddedToken for special tokens with specific tokenization behaviors
        # all_special_tokens_extended = dict(
        #     (str(t), t)
        #     for t in self.all_special_tokens_extended
        #     if isinstance(t, AddedToken)
        # )

        self.spec_init()
        text, kwargs = self.prepare_for_tokenization(text, **kwargs)

        # TODO: should this be in the base class?
        if hasattr(self, "do_lower_case") and self.do_lower_case:
            # convert non-special tokens to lowercase
            escaped_special_toks = [re.escape(s_tok) for s_tok in (self.unique_no_split_tokens + self.all_spec_tok)]
            pattern = r"(" + r"|".join(escaped_special_toks) + r")|" + r"(.+?)"
            text = re.sub(pattern, lambda m: m.groups()[0] or m.groups()[1].lower(), text)

        no_split_token = set(self.unique_no_split_tokens)
        tokens = self.tokens_trie.split(text)

        # ["This is something", "<special_token_1>", "  else"]
        # for i, token in enumerate(tokens):
        #     if token in no_split_token:
        #         tok_extended = all_special_tokens_extended.get(token, None)
        #         print(f'>>>{token}|{tok_extended}|{all_special_tokens_extended}<<<')
        #         left = tokens[i - 1] if i > 0 else None
        #         right = tokens[i + 1] if i < len(tokens) - 1 else None
        #         if isinstance(tok_extended, AddedToken):
        #             if tok_extended.rstrip and right:
        #                 # A bit counter-intuitive but we strip the left of the string
        #                 # since tok_extended.rstrip means the special token is eating all white spaces on its right
        #                 tokens[i + 1] = right.lstrip()
        #             # Strip white spaces on the left
        #             if tok_extended.lstrip and left:
        #                 tokens[i - 1] = left.rstrip()  # Opposite here
        #         else:
        #             We strip left and right by default
        #             if right:
        #                 tokens[i + 1] = right.lstrip()
        #             if left:
        #                 tokens[i - 1] = left.rstrip()
        # ["This is something", "<special_token_1>", "else"]
        tokenized_text = []
        for token in tokens:
            # Need to skip eventual empty (fully stripped) tokens
            if not token:
                continue
            if token in no_split_token:
                tokenized_text.append(token)
            else:
                tokenized_text.extend(self._tokenize(token))
        # ["This", " is", " something", "<special_token_1>", "else"]
        return tokenized_text

    def _decode(self, *args, **kwargs):
        """doc"""
        kwargs.pop("clean_up_tokenization_spaces", None)
        kwargs.pop("spaces_between_special_tokens", None)
        return super()._decode(
            *args, **kwargs, clean_up_tokenization_spaces=False, spaces_between_special_tokens=False
        )

    def _pad(
        self,
        encoded_inputs: Dict,
        max_length: Optional[int] = None,
        padding_strategy=PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
    ) -> dict:
        """doc"""
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names
        if return_attention_mask:
            required_input = encoded_inputs[self.model_input_names[0]]
            if padding_strategy == PaddingStrategy.LONGEST:
                max_length = len(required_input)
            if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
                max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
            needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) != max_length
            if "attention_mask" in encoded_inputs and encoded_inputs["attention_mask"] is not None:
                attention_mask = encoded_inputs.pop("attention_mask")
                if isinstance(attention_mask, paddle.Tensor):
                    attention_mask = attention_mask.numpy()
                elif isinstance(attention_mask, list):
                    attention_mask = np.array(attention_mask)
                elif not isinstance(attention_mask, np.ndarray):
                    raise ValueError(f"Unexpected type {type(attention_mask)} of attention_mask, ")
            else:
                attention_mask = np.tril(np.ones((len(required_input), len(required_input)), dtype=np.int64))
                attention_mask = np.expand_dims(attention_mask, axis=0)
            if needs_to_be_padded:
                difference = max_length - len(required_input)
                if self.padding_side == "right":
                    if attention_mask.ndim == 1:
                        pad_width = [(0, difference)]
                    else:
                        pad_width = [(0, 0), (0, difference), (0, difference)]
                elif self.padding_side == "left":
                    if attention_mask.ndim == 1:
                        pad_width = [(difference, 0)]
                    else:
                        pad_width = [(0, 0), (difference, 0), (difference, 0)]
                else:
                    raise ValueError("Invalid padding strategy:" + str(self.padding_side))
                attention_mask = np.pad(attention_mask, pad_width=pad_width, mode="constant", constant_values=0)
        encoded_inputs = super()._pad(
            encoded_inputs,
            max_length,
            padding_strategy=padding_strategy,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=False,
        )
        if return_attention_mask:
            encoded_inputs["attention_mask"] = attention_mask.tolist()
        return encoded_inputs


def add_special_tokens(
    tokenizer,
    special_tokens_info,
    use_ocr_specialtoken=False,
    use_crop_specialtoken=False,
    special_token_ids_start=254208,
    special_token_ids_end=256256,
):
    """
    增加 special token

    placeholder [<|IMAGE_PLACEHOLDER|>, <|AUDIO_PLACEHOLDER|>, <|VIDEO_PLACEHOLDER|>] 共3个

    模态起始截止 special tokens [<|BOI|> <|EOI|> <|BOA|> <|EOA|> <|BOV|> <|EOV|>]

    ocr special tokens [<|LOC_0|> <|LOC_1|> ... <|LOC_1000|>] 共1001个

    crop special tokens [<|CROP_COL_SEP|>, <|CROP_ROW_SEP|>, <|CROP_IMAGE_SEP|>] 共3个
        <|CROP_COL_SEP|> for col 维度切 图片width（替换原明文逗号）
        <|CROP_ROW_SEP|> for row 维度切 图片height（替换原明文回车）
        <|CROP_IMAGE_SEP|> for 区分原图和crop图 图片width（替换原明文两个回车）

    共2048个 unsed token

    Args:
        tokenizer (ErnieTokenizer): tokenizer
        special_token_ids_start (int, optional): special token 起点 ids. Defaults to 254208.
        special_token_ids_end (int, optional): 词表最多支持大小. Defaults to 256256.
    """
    special_tokens = [special_tokens_info["image_placeholder"], special_tokens_info["audio_placeholder"]]

    if use_ocr_specialtoken:
        special_tokens.extend(special_tokens_info["ocr_coor"])
        special_tokens.extend(special_tokens_info["ocr_begin_end"])

    if use_crop_specialtoken:
        special_tokens.extend(special_tokens_info["crop"])

    # add special_tokens
    additional_special_tokens = {"additional_special_tokens": special_tokens}
    tokenizer.add_special_tokens(additional_special_tokens)

    # check
    first_special_tokens = tokenizer.encode(special_tokens[0])["input_ids"]

    assert first_special_tokens[0] == special_token_ids_start, f"[ERROR] first_special_tokens={first_special_tokens}"
    assert (
        len(tokenizer.get_vocab()) < special_token_ids_end
    ), f"[ERROR] vocab_size = {len(tokenizer.get_vocab())} >= {special_token_ids_end} 增加过多special token了!"


class Ernie45Tokenizer(PretrainedTokenizer):
    """
    一个更好用的 `ErnieBotToknizer`，
    能 encode 目前 sft/ppo 阶段的特殊token，也支持多模态。
    """

    resource_files_names = {"vocab_file": "tokenizer.model"}
    pretrained_resource_files_map = {"vocab_file": {"ernie-bot-10b": None}}
    pretrained_init_configuration = {"ernie-bot-10b": {}}
    model_input_names = ["input_ids", "position_ids", "attention_mask", "labels"]
    padding_side = "right"

    def __init__(
        self,
        vocab_file,
        bos_token="<s>",
        cls_token="<cls>",
        eos_token="</s>",
        mask_token="<mask:0>",
        pad_token="<pad>",
        sep_token="<sep>",
        unk_token="<unk>",
        additional_special_tokens=None,
        verbose=False,
        **kwargs,
    ):
        """doc"""
        if additional_special_tokens is None:
            additional_special_tokens = ["<mask:1>", "<mask:7>"]
        super().__init__(
            bos_token=bos_token,
            cls_token=cls_token,
            eos_token=eos_token,
            mask_token=mask_token,
            pad_token=pad_token,
            sep_token=sep_token,
            unk_token=unk_token,
            additional_special_tokens=additional_special_tokens,
            verbose=False,
            **kwargs,
        )
        self.vocab_file = vocab_file
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(vocab_file)
        # pre-process map-type all spec token for decode accelerate.

    @property
    def space_token(self):
        """doc"""
        return "<mask:1>"

    @property
    def space_token_id(self):
        """doc"""
        return self.sp_model.piece_to_id("<mask:1>")

    @property
    def gend_token(self):
        """doc"""
        return "<mask:7>"

    @property
    def gend_token_id(self):
        """doc"""
        return self.sp_model.piece_to_id("<mask:7>")

    @property
    def im_start_id(self):
        """doc"""
        return self.sp_model.piece_to_id("<|im_start|>")

    @property
    def im_end_id(self):
        """doc"""
        return self.sp_model.piece_to_id("<|im_end|>")

    @property
    def vocab_size(self):
        """doc"""
        return self.sp_model.vocab_size()

    def get_vocab(self):
        """doc"""
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text):
        """doc"""
        return self.sp_model.encode_as_pieces(text)

    def _convert_token_to_id(self, token):
        """doc"""
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, id):
        """doc"""
        return self.sp_model.id_to_piece(id)

    def spec_init(self):
        """初始化特殊token集合
        如果实例中不存在all_spec_tok属性，则使用all_special_tokens创建集合
        并赋值给all_spec_tok属性
        """
        if not hasattr(self, "all_spec_tok"):
            self.all_spec_tok = set(self.all_special_tokens)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        self.spec_init()
        current_sub_tokens = []
        out_string = ""
        # prev_is_special = False
        for token in tokens:
            # make sure that special tokens are not decoded using sentencepiece model
            if token in self.all_spec_tok:
                # if not prev_is_special:
                #     out_string += " "
                out_string += self.sp_model.decode(current_sub_tokens) + token
                # prev_is_special = True

                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
                # prev_is_special = False
        out_string += self.sp_model.decode(current_sub_tokens)
        return out_string  # .strip()

    def prepare_for_model(self, *args, **kwargs):
        """doc"""
        if "add_special_tokens" in kwargs:
            kwargs.pop("add_special_tokens")
            # logger.warning(f'Ernie45Tokenizer v2 does not support `add_special_tokens`')
        return super().prepare_for_model(*args, **kwargs)

    def save_vocabulary(self, save_directory, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the vocabulary and special tokens file to a directory.
        Args:
            save_directory (`str`):
                The directory in which to save the vocabulary.
        Returns:
            `Tuple(str)`: Paths to the files saved.
        """
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + self.resource_files_names["vocab_file"],
        )
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)
        return (out_vocab_file,)

    def tokenize(self, text: TextInput, **kwargs) -> List[str]:
        """
        Converts a string in a sequence of tokens, using the tokenizer.

        Split in words for word-based vocabulary or sub-words for sub-word-based vocabularies
        (BPE/SentencePieces/WordPieces). Takes care of added tokens.

        Args:
            text (`str`):
                The sequence to be encoded.
            **kwargs (additional keyword arguments):
                Passed along to the model-specific `prepare_for_tokenization` preprocessing method.

        Returns:
            `List[str]`: The list of tokens.
        """

        self.spec_init()
        text, kwargs = self.prepare_for_tokenization(text, **kwargs)

        # TODO: should this be in the base class?
        if hasattr(self, "do_lower_case") and self.do_lower_case:
            # convert non-special tokens to lowercase
            escaped_special_toks = [re.escape(s_tok) for s_tok in (self.unique_no_split_tokens + self.all_spec_tok)]
            pattern = r"(" + r"|".join(escaped_special_toks) + r")|" + r"(.+?)"
            text = re.sub(pattern, lambda m: m.groups()[0] or m.groups()[1].lower(), text)

        no_split_token = set(self.unique_no_split_tokens)
        tokens = self.tokens_trie.split(text)

        tokenized_text = []
        for token in tokens:
            # Need to skip eventual empty (fully stripped) tokens
            if not token:
                continue
            if token in no_split_token:
                tokenized_text.append(token)
            else:
                tokenized_text.extend(self._tokenize(token))
        # ["This", " is", " something", "<special_token_1>", "else"]
        return tokenized_text

    def _decode(self, *args, **kwargs):
        """doc"""
        kwargs.pop("clean_up_tokenization_spaces", None)
        kwargs.pop("spaces_between_special_tokens", None)
        return super()._decode(
            *args, **kwargs, clean_up_tokenization_spaces=False, spaces_between_special_tokens=False
        )

    def _pad(
        self,
        encoded_inputs: Dict,
        max_length: Optional[int] = None,
        padding_strategy=PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
    ) -> dict:
        """doc"""
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names
        if return_attention_mask:
            required_input = encoded_inputs[self.model_input_names[0]]
            if padding_strategy == PaddingStrategy.LONGEST:
                max_length = len(required_input)
            if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
                max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
            needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) != max_length
            if "attention_mask" in encoded_inputs and encoded_inputs["attention_mask"] is not None:
                attention_mask = encoded_inputs.pop("attention_mask")
                if isinstance(attention_mask, paddle.Tensor):
                    attention_mask = attention_mask.numpy()
                elif isinstance(attention_mask, list):
                    attention_mask = np.array(attention_mask)
                elif not isinstance(attention_mask, np.ndarray):
                    raise ValueError(f"Unexpected type {type(attention_mask)} of attention_mask, ")
            else:
                attention_mask = np.tril(np.ones((len(required_input), len(required_input)), dtype=np.int64))
                attention_mask = np.expand_dims(attention_mask, axis=0)
            if needs_to_be_padded:
                difference = max_length - len(required_input)
                if self.padding_side == "right":
                    if attention_mask.ndim == 1:
                        pad_width = [(0, difference)]
                    else:
                        pad_width = [(0, 0), (0, difference), (0, difference)]
                elif self.padding_side == "left":
                    if attention_mask.ndim == 1:
                        pad_width = [(difference, 0)]
                    else:
                        pad_width = [(0, 0), (difference, 0), (difference, 0)]
                else:
                    raise ValueError("Invalid padding strategy:" + str(self.padding_side))
                attention_mask = np.pad(attention_mask, pad_width=pad_width, mode="constant", constant_values=0)
        encoded_inputs = super()._pad(
            encoded_inputs,
            max_length,
            padding_strategy=padding_strategy,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=False,
        )
        if return_attention_mask:
            encoded_inputs["attention_mask"] = attention_mask.tolist()
        return encoded_inputs


hack_uft16_ascii = True
VOCAB_FILES_NAMES = {"vocab_file": "spm.model"}


class OOVProcess:
    """
    针对OOV词，做UTF-16-BE编码
    """

    def __init__(self, vocab):
        """
        Args:
        vocab (dict): dict {token:id}, token is the word in vocabulary, id is the index of this word in vocabulary
                      e.g., {'hello': 0, 'world': 1, ...}
        """
        self.vocab = vocab  # dict {token:id}
        self.b16_token_id_dict, self.b16_id_token_dict, self.bf16_tokens = self.get_b16_dict(self.vocab)
        self.bf16_tokens = set(self.bf16_tokens)

        self.PREFIX = "<0x"
        self.SUFFIX = ">"

    def encode_str(self, s, tgt_type):
        """输入s是字符串，tgt_type是要编码的类型，最终得到十六进制表示字节列表。如输入为s=“魍”, tgt_type=‘utf-16-be’,输出[‘<0x9B>’, ‘<0x4D>’]"""

        # 将字符串编码为指定类型的字节串
        encoded_bytes = s.encode(tgt_type)
        # 转换为十六进制表示的字节列表
        hex_list = [f"<0x{byte:02X}>" for byte in encoded_bytes]
        return hex_list

    def decode_str(self, byte_16_list, tgt_type="utf-16-be"):
        """
        功能正好相反，输入s是十六进制表示字节列表，tgt_type是编码的类型，输出字符串。
        如输出byte_16_list=[‘<0x9B>’, ‘<0x4D>’], tgt_type=‘utf-16-be’,输出“ 魍”
        """

        # 去除尖括号和'0x'前缀，并将其转换为字节数组
        byte_array = bytearray(int(byte[3:-1], 16) for byte in byte_16_list)
        # 将字节数组解码为字符串
        decoded_str = byte_array.decode(tgt_type)
        return decoded_str

    def tgt_type_convert(self, byte_16_list, src_type="utf-8", tgt_type="utf-16-be"):
        """
        输入是byte_16_list是十六进制的列表，src_type是byte_16_list的类型，tgt_type是要转换的类型。输出是类型为tgt_type的十六进制列表。
        例如输出byte_16_list=[‘<0xE9>’, ‘<0xAD>’, ‘<0x8D>’], src_type=“utf-8”,tgt_type=“utf-16-be”,输出是[‘<0x9B>’, ‘<0x4D>’]。
        """
        # =======编码类型转换=== src_type->tgt_type

        if tgt_type == "utf-8" and src_type == "utf-16-be":
            # hack: 针对OOV词被截断的bf16 ascii 字符，编码转化会失败，直接返回原字节列表
            try:
                # 使用 decode_str 将字节列表解码成字符串
                decoded_str = self.decode_str(byte_16_list, src_type)

                # 使用 encode_str 将字符串编码为目标类型的字节列表
                encoded_list = self.encode_str(decoded_str, tgt_type)
            except UnicodeDecodeError:
                logger.warning(
                    f"UnicodeDecodeError: 被截断的OOV词无法转码,decode明文:{byte_16_list}，src_type：{src_type}"
                )
                return byte_16_list
        else:
            # 使用 decode_str 将字节列表解码成字符串
            decoded_str = self.decode_str(byte_16_list, src_type)

            # 使用 encode_str 将字符串编码为目标类型的字节列表
            encoded_list = self.encode_str(decoded_str, tgt_type)

        # # 使用 decode_str 将字节列表解码成字符串
        # decoded_str = self.decode_str(byte_16_list, src_type)

        # # 使用 encode_str 将字符串编码为目标类型的字节列表
        # encoded_list = self.encode_str(decoded_str, tgt_type)

        return encoded_list

    def is_hex_string(self, token):
        """16进制判断"""
        return token in self.bf16_tokens

    def change_single_ascii_for_utf16be(self, tokens_list_part, byte16_flag):
        """TODO: hack 代码 LGH lgh 某个single ascii 字符没有添加到词表中，因此这个字符不转化,截断导致"""
        tokens_list_part_v1 = []
        byte16_flag_v1 = []

        for t, b in zip(tokens_list_part, byte16_flag):
            if b == 0:
                tokens_list_part_v1.append(t)
                byte16_flag_v1.append(b)
            else:
                if len(t) % 2 == 0:
                    tokens_list_part_v1.append(t)
                    byte16_flag_v1.append(b)
                else:
                    new_t = [t.pop()]
                    if t != []:
                        tokens_list_part_v1.append(t)
                        byte16_flag_v1.append(b)
                    tokens_list_part_v1.append(new_t)
                    byte16_flag_v1.append(0)

        tokens_list_part = tokens_list_part_v1
        byte16_flag = byte16_flag_v1

        # assert False
        return tokens_list_part, byte16_flag

    def oov_token_check(self, tokens_list):
        """
        检测 tokens_list 中的 16 进制片段，并进行分组。

        Args:
            tokens_list (list): 输入的 token 列表。

        Returns:
            tuple: 包含两个元素的元组：
                - tokens_list_part (list): 分组后的列表。
                - byte16_flag (list): 每组是否为全 16 进制元素的标志位列表（1 表示全为 16 进制，0 表示否）。
        """
        tokens_list_part = []
        byte16_flag = []

        current_group = []
        is_byte16 = None

        for token in tokens_list:
            # 使用正则表达式函数判断是否为十六进制格式的字符串
            if self.is_hex_string(token):
                # 当前元素是十六进制字符串
                if is_byte16 is None:
                    # 初始化当前组类型
                    is_byte16 = True
                if not is_byte16:
                    # 切换组，保存之前的组
                    tokens_list_part.append(current_group)
                    byte16_flag.append(0)
                    current_group = []
                    is_byte16 = True
            else:
                # 当前元素是普通字符串
                if is_byte16 is None:
                    # 初始化当前组类型
                    is_byte16 = False
                if is_byte16:
                    # 切换组，保存之前的组
                    tokens_list_part.append(current_group)
                    byte16_flag.append(1)
                    current_group = []
                    is_byte16 = False

            # 添加当前元素到当前组
            current_group.append(token)

        # 添加最后一个组
        if current_group:
            tokens_list_part.append(current_group)
            byte16_flag.append(1 if is_byte16 else 0)

        return tokens_list_part, byte16_flag

    def get_b16_dict(self, vocab):
        """从vocab中得到16进制的id"""
        hex_chars = "0123456789ABCDEF"
        bf16_tokens = [f"<0x{''.join(p)}>" for p in product(hex_chars, repeat=2)]
        assert len(bf16_tokens) == 256, bf16_tokens
        b16_token_id_dict = {}
        b16_id_token_dict = {}

        for bf16_t in bf16_tokens:
            idx = vocab[bf16_t]
            b16_token_id_dict[bf16_t] = idx
            b16_id_token_dict[idx] = bf16_t
        assert len(b16_token_id_dict) == len(b16_id_token_dict) == 256, f"{b16_token_id_dict}\n{b16_id_token_dict}"
        return b16_token_id_dict, b16_id_token_dict, bf16_tokens

    def encode_or_tokenize_convert_oov(self, tokens=None, token_ids=None, src_type="utf-8", tgt_type="utf-16-be"):
        """目的：tokenizer生成token或者token id过程中，对于oov词会拆成utf-8 的16进制表示,需要将这部分表示转化为utf-16-be 的16进制表示"""

        # token转化
        new_tokens = []
        if tokens is not None:
            # 筛选出ovv
            tokens_list_part, byte16_flag = self.oov_token_check(tokens)

            # TODO: hack 代码 LGH lgh 某个single ascii 字符没有添加到词表中，因此这个字符不转化。临时解决方案！！后续将这些字符加到词表中！！！ ====
            # if tgt_type=="utf-16-be":
            if tgt_type == "utf-8" and hack_uft16_ascii:
                tokens_list_part, byte16_flag = self.change_single_ascii_for_utf16be(tokens_list_part, byte16_flag)
            # ====

            # 将oov utf-8 的16进制表示转化为utf-16-be 16进制表示
            # ==================原始版本===============
            for token_part, byte16 in zip(tokens_list_part, byte16_flag):
                assert byte16 in [0, 1], byte16
                if byte16 == 1:
                    # utf-8 16进制 => utf-16-be 16进制
                    token_part = self.tgt_type_convert(token_part, src_type=src_type, tgt_type=tgt_type)
                new_tokens.extend(token_part)
            # ==========================
            # ## ==========加速版本2 ===========
            # new_tokens = [
            #     converted_token
            #     for token_part, byte16 in zip(tokens_list_part, byte16_flag)
            #     for converted_token in (
            #         self.tgt_type_convert(token_part, src_type=src_type, tgt_type=tgt_type)
            #         if byte16 == 1
            #         else token_part
            #     )
            # ]
            # # ==========================

        # ## token id转化
        new_token_ids = []
        if token_ids is not None:
            new_token_ids_b16 = []
            assert self.b16_id_token_dict != {}, token_ids
            # 将16进制的id转化为token

            token_ids_16 = [self.b16_id_token_dict.get(id_one, id_one) for id_one in token_ids]
            # 筛选出ovv
            tokens_list_part, byte16_flag = self.oov_token_check(token_ids_16)

            # TODO: hack 代码 LGH lgh 某个single ascii 字符没有添加到词表中，因此这个字符不转化。临时解决方案！！后续将这些字符加到词表中！！！ ====
            # if tgt_type=="utf-16-be":
            if tgt_type == "utf-8" and hack_uft16_ascii:
                tokens_list_part, byte16_flag = self.change_single_ascii_for_utf16be(tokens_list_part, byte16_flag)
            # ====

            # 将oov utf-8 的16进制表示转化为utf-16-be 16进制表示
            # ==================原始版本===============
            for token_part, byte16 in zip(tokens_list_part, byte16_flag):
                assert byte16 in [0, 1], byte16
                if byte16 == 1:
                    # utf-8 16进制 => utf-16-be 16进制
                    token_part = self.tgt_type_convert(token_part, src_type=src_type, tgt_type=tgt_type)
                new_token_ids_b16.extend(token_part)
            # ========================================
            # ### ==========加速版本2 ===========
            # new_token_ids_b16 = [
            #     converted_token
            #     for token_part, byte16 in zip(tokens_list_part, byte16_flag)
            #     for converted_token in (
            #         self.tgt_type_convert(token_part, src_type=src_type, tgt_type=tgt_type)
            #         if byte16 == 1
            #         else token_part
            #     )
            # ]
            # # ===========================
            new_token_ids = [self.b16_token_id_dict.get(id_one, id_one) for id_one in new_token_ids_b16]

        return new_tokens, new_token_ids

    def decode_convert_oov(self, tokens=None, token_ids=None, src_type="utf-16-be", tgt_type="utf-8"):
        """
        目的：sentencepiece中的sp.decode(ids)以及sp.decode_pieces(pieces)中的id或pieces中的16进制token都必须是"utf-8"格式，
        但是现在tokenizer.tokenize出来的16进制是“utf-16-be”，因此要转化为"utf-8"格式后，送入sentencepiece解码。
        """
        new_tokens, new_token_ids = self.encode_or_tokenize_convert_oov(
            tokens=tokens, token_ids=token_ids, src_type=src_type, tgt_type=tgt_type
        )
        return new_tokens, new_token_ids

    def encode_str_and_encode_str(self, s, tgt_type):
        """doc"""
        pass
        # encoded_list = self.encode_str(s, tgt_type)
        # decoded_str = self.decode_str(encoded_list, tgt_type)

    @staticmethod
    def get_vocab(model_file):
        """doc"""
        sp = spm.SentencePieceProcessor(model_file=model_file)
        vocab_size = sp.vocab_size()
        assert sp.vocab_size() == sp.get_piece_size()
        vocab = {sp.id_to_piece(i): i for i in range(vocab_size)}
        return vocab, sp


class Ernie5Tokenizer(PretrainedTokenizer):
    """
    Construct a ErnieBot tokenizer. Based on byte-level Byte-Pair-Encoding.
    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
    """

    vocab_files_names = {"vocab_file": "spm.model"}
    pretrained_vocab_files_map = {"vocab_file": {}, "tokenizer_file": {}}
    max_model_input_sizes = {}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        add_bos_token=True,
        add_eos_token=False,
        clean_up_tokenization_spaces=False,
        **kwargs,
    ):
        """
            Constructs a SentencePieceTokenizer.

        Args:
            vocab_file (str): The vocabulary file path (ends with .model) required to instantiate
                the SentencePiece processor.
            unk_token (str, optional): The unknown token. Defaults to " ".
            bos_token (Union[str, AddedToken], optional): The beginning of sentence token. Defaults to " ".
            eos_token (Union[str, AddedToken], optional): The end of sentence token. Defaults to " ".
            pad_token (Union[str, AddedToken], optional): The padding token. Defaults to "<pad>".
            sp_model_kwargs (Optional[Dict[str, Any]], optional): Keyword arguments passed to the SentencePiece
                constructor. Defaults to None.
            add_bos_token (bool, optional): Whether or not to add the bos token at the beginning of every
                encoded piece. Defaults to True.
            add_eos_token (bool, optional): Whether or not to add the eos token at the end of every encoded piece.
                Defaults to False.
            clean_up_tokenization_spaces (bool, optional): Whether or not to clean up the tokenization spaces.
                Defaults to False.
            **kwargs (Any, optional): Additional keyword arguments passed along to the `__init__` method of the
                parent `PretrainedTokenizer`.
        """
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            sp_model_kwargs=self.sp_model_kwargs,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )
        self.vocab_file = vocab_file
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(vocab_file)
        # for eb35 reader
        self.bos_id = self.bos_token_id
        self.eos_id = self.eos_token_id
        self.sep_id = self.sep_token_id
        self.pad_id = self.pad_token_id
        self.unk_id = self.unk_token_id

        vocab = self.get_vocab()  # oov
        self.oov_process = OOVProcess(vocab)  # oov
        self.use_oov_uft_16_be = True  # True # oov是否使用uft_16_be编码
        logger.info(f">>> UTF_16_BE: self.use_oov_uft_16_be:{self.use_oov_uft_16_be}")

    def set_oov_utf_16_be(self, use_oov_uft_16_be=True):
        """
        use_oov_uft_16_be 开关
        """
        self.use_oov_uft_16_be = use_oov_uft_16_be
        print(f"use_oov_uft_16_be:{self.use_oov_uft_16_be}")

    def __getstate__(self):
        """
            Override the default __getstate__ method to prevent pickling of spaCy models.

        Args:
            None

        Returns:
            dict (state): A dictionary containing all instance attributes except for "sp_model".
        """
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    def __setstate__(self, d):
        """
            Restore the state of the object from a dictionary.

        Args:
            d (dict): A dictionary containing the state of the object.
                It should contain the keys 'sp_model_kwargs' and 'vocab_file'.

        Returns:
            None. The object is updated in-place with the provided state.
        """
        self.__dict__ = d
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(self.vocab_file)

    @property
    def vocab_size(self):
        """Returns vocab size"""
        return self.sp_model.get_piece_size()

    def get_vocab(self):
        """Returns vocab as a dict"""
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def tokenize(self, text):
        """Returns a tokenized string."""
        return self._tokenize(text)

    def encode_oov_uft_16_be(self, tokens):
        """spm encode 或者 tokenizer生成tokens、token_ids后，使用此函数针对oov词转化为utf16be编码"""
        if isinstance(tokens, list):
            pass
        else:
            tokens = [tokens]

        if isinstance(tokens[0], str):
            tokens, _ = self.oov_process.encode_or_tokenize_convert_oov(tokens=tokens)
        else:
            assert isinstance(tokens[0], int)
            _, tokens = self.oov_process.encode_or_tokenize_convert_oov(token_ids=tokens)
        return tokens

    def is_empty(self, value):
        """检查是否为 None"""
        if value is None:
            return True

        # 检查是否为空字符串
        if isinstance(value, str) and value == "":
            return True

        # 检查是否为空列表
        if isinstance(value, list) and len(value) == 0:
            return True

        # 如果不是以上任何一种，返回 False
        return False

    def _tokenize(self, text):
        """Returns a tokenized string."""
        tokens = self.sp_model.encode(text, out_type=str)
        if not self.is_empty(tokens) and self.use_oov_uft_16_be:  # oov utf8转化为utf16be
            tokens = self.encode_oov_uft_16_be(tokens=tokens)
        return tokens

    def decode_oov_uft_16_be(self, tokens):
        """spm decode前，将tokens、token_ids形式中OOV词的utf16be编码转化为utf8编码"""
        if isinstance(tokens, list):
            pass
        else:
            tokens = [tokens]

        if isinstance(tokens[0], str):
            tokens, _ = self.oov_process.decode_convert_oov(tokens=tokens)
        else:
            assert isinstance(tokens[0], int)
            _, tokens = self.oov_process.decode_convert_oov(token_ids=tokens)
        return tokens

    def decode(self, tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        """Returns a tokenized string."""
        if not self.is_empty(tokens) and self.use_oov_uft_16_be:  # oov utf16be转化为utf8
            tokens = self.decode_oov_uft_16_be(tokens)
        return self.sp_model.decode(tokens)

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        token = self.sp_model.IdToPiece(index)
        return token

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        current_sub_tokens = []
        out_string = ""
        prev_is_special = False
        for i, token in enumerate(tokens):
            # make sure that special tokens are not decoded using sentencepiece model
            if token in self.all_special_tokens:
                if not prev_is_special and i != 0:
                    out_string += " "

                if not self.is_empty(current_sub_tokens) and self.use_oov_uft_16_be:  # oov utf16be转化为utf8
                    current_sub_tokens = self.decode_oov_uft_16_be(current_sub_tokens)

                out_string += self.sp_model.decode(current_sub_tokens) + token
                prev_is_special = True
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
                prev_is_special = False

        if not self.is_empty(current_sub_tokens) and self.use_oov_uft_16_be:  # oov utf16be转化为utf8
            current_sub_tokens = self.decode_oov_uft_16_be(current_sub_tokens)
        out_string += self.sp_model.decode(current_sub_tokens)
        return out_string

    def save_vocabulary(self, save_directory, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the vocabulary and special tokens file to a directory.
        Args:
            save_directory (`str`):
                The directory in which to save the vocabulary.
        Returns:
            `Tuple(str)`: Paths to the files saved.
        """
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"],
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        return (out_vocab_file,)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        build_inputs_with_special_tokens
        """
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        output = bos_token_id + token_ids_0 + eos_token_id

        if token_ids_1 is not None:
            output = output + bos_token_id + token_ids_1 + eos_token_id

        return output

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.
        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.
        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        bos_token_id = [1] if self.add_bos_token else []
        eos_token_id = [1] if self.add_eos_token else []

        if token_ids_1 is None:
            return bos_token_id + ([0] * len(token_ids_0)) + eos_token_id
        return (
            bos_token_id
            + ([0] * len(token_ids_0))
            + eos_token_id
            + bos_token_id
            + ([0] * len(token_ids_1))
            + eos_token_id
        )

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task. An ALBERT
        sequence pair mask has the following format:
        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```
        if token_ids_1 is None, only returns the first portion of the mask (0s).
        Args:
            token_ids_0 (`List[int]`):
                List of ids.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        output = [0] * len(bos_token_id + token_ids_0 + eos_token_id)

        if token_ids_1 is not None:
            output += [1] * len(bos_token_id + token_ids_1 + eos_token_id)

        return output
