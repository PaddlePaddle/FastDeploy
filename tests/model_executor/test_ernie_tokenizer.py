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

import os
from unittest.mock import MagicMock, patch

import pytest
from sentencepiece import SentencePieceProcessor

from fastdeploy.model_executor.guided_decoding.ernie_tokenizer import Ernie4_5Tokenizer

# ===== Pytest Fixtures =====


@pytest.fixture
def mock_sp_model():
    """Create a mock sentencepiece model"""
    model = MagicMock()
    model.get_piece_size.return_value = 10  # Smaller size for faster tests
    model.piece_to_id.return_value = 1
    model.IdToPiece.return_value = "test"
    model.encode.return_value = ["▁Hello", "▁World"]
    model.decode.return_value = "Hello World"
    model.serialized_model_proto.return_value = b"mock_model_proto"
    return model


@pytest.fixture
def vocab_file(tmp_path):
    """Create a temporary vocab file path"""
    return str(tmp_path / "test_spm.model")


@pytest.fixture
def tokenizer(mock_sp_model, vocab_file):
    """Create a tokenizer instance with mocked dependencies"""
    with patch.object(SentencePieceProcessor, "Load", return_value=None):
        with patch("sentencepiece.SentencePieceProcessor") as mock_spm:
            mock_spm.return_value = mock_sp_model
            return Ernie4_5Tokenizer(vocab_file=vocab_file)


# ===== Initialization Tests (Parametrized) =====


@pytest.mark.parametrize(
    "add_bos,add_eos,sp_kwargs,expected_bos,expected_eos",
    [
        # Default parameters
        (True, False, {}, True, False),
        # Custom parameters
        (False, True, {"add_extra_options": True}, False, True),
    ],
)
def test_tokenizer_initialization(mock_sp_model, vocab_file, add_bos, add_eos, sp_kwargs, expected_bos, expected_eos):
    """Test tokenizer initialization with default and custom parameters"""
    with patch.object(SentencePieceProcessor, "Load", return_value=None):
        with patch("sentencepiece.SentencePieceProcessor") as mock_spm:
            mock_spm.return_value = mock_sp_model

            tokenizer = Ernie4_5Tokenizer(
                vocab_file=vocab_file,
                sp_model_kwargs=sp_kwargs,
                add_bos_token=add_bos,
                add_eos_token=add_eos,
            )

            # Verify attributes
            assert tokenizer.vocab_file == vocab_file
            assert tokenizer.add_bos_token == expected_bos
            assert tokenizer.add_eos_token == expected_eos
            assert tokenizer.sp_model_kwargs == sp_kwargs

            # Verify special tokens (AddedToken objects have .token attribute, strings don't)
            bos_token = tokenizer.bos_token
            eos_token = tokenizer.eos_token
            unk_token = tokenizer.unk_token
            pad_token = tokenizer.pad_token

            # Handle both AddedToken objects and strings
            assert (bos_token.token if hasattr(bos_token, "token") else bos_token) == "<s>"
            assert (eos_token.token if hasattr(eos_token, "token") else eos_token) == "</s>"
            assert (unk_token.token if hasattr(unk_token, "token") else unk_token) == "<unk>"
            assert (pad_token.token if hasattr(pad_token, "token") else pad_token) == "<pad>"


# ===== Core Functionality Tests =====


def test_tokenize(tokenizer, mock_sp_model):
    """Test tokenize method - core functionality"""
    result = tokenizer.tokenize("Hello World")

    assert result == ["▁Hello", "▁World"]
    mock_sp_model.encode.assert_called_once_with("Hello World", out_type=str)


def test_decode(tokenizer, mock_sp_model):
    """Test decode method - core functionality"""
    result = tokenizer.decode([1, 2, 3])

    assert result == "Hello World"
    mock_sp_model.decode.assert_called_once_with([1, 2, 3])


def test_get_vocab(tokenizer, mock_sp_model):
    """Test get_vocab method"""

    # Mock convert_ids_to_tokens to return unique tokens
    def mock_convert_ids_to_tokens(token_id):
        return f"token_{token_id}"

    tokenizer.convert_ids_to_tokens = mock_convert_ids_to_tokens

    vocab = tokenizer.get_vocab()

    # Check vocab structure
    assert len(vocab) >= 10  # Matches the mock sp_model size
    # Vocabulary contains tokens from the mock sp_model
    assert isinstance(vocab, dict)
    assert "token_0" in vocab


def test_convert_token_to_id(mock_sp_model, vocab_file):
    """Test _convert_token_to_id method"""
    mock_sp_model.piece_to_id.return_value = 42
    with patch.object(SentencePieceProcessor, "Load", return_value=None):
        with patch("sentencepiece.SentencePieceProcessor") as mock_spm:
            mock_spm.return_value = mock_sp_model
            tokenizer = Ernie4_5Tokenizer(vocab_file=vocab_file)

            result = tokenizer._convert_token_to_id("test_token")

            assert result == 42
            mock_sp_model.piece_to_id.assert_called_once_with("test_token")


def test_convert_id_to_token(mock_sp_model, vocab_file):
    """Test _convert_id_to_token method"""
    mock_sp_model.IdToPiece.return_value = "test_piece"
    with patch.object(SentencePieceProcessor, "Load", return_value=None):
        with patch("sentencepiece.SentencePieceProcessor") as mock_spm:
            mock_spm.return_value = mock_sp_model
            tokenizer = Ernie4_5Tokenizer(vocab_file=vocab_file)

            # Reset mock to ignore calls from initialization
            mock_sp_model.IdToPiece.reset_mock()

            result = tokenizer._convert_id_to_token(42)

            assert result == "test_piece"
            mock_sp_model.IdToPiece.assert_called_once_with(42)


def test_convert_tokens_to_string(mock_sp_model, vocab_file):
    """Test convert_tokens_to_string with special tokens"""

    def mock_decode(tokens):
        if tokens:
            return "hello world"
        return ""

    mock_sp_model.decode.side_effect = mock_decode

    with patch.object(SentencePieceProcessor, "Load", return_value=None):
        with patch("sentencepiece.SentencePieceProcessor") as mock_spm:
            mock_spm.return_value = mock_sp_model
            tokenizer = Ernie4_5Tokenizer(vocab_file=vocab_file)

            # Test with special tokens mixed with regular tokens
            tokens = ["hello", "<s>", "world", "</s>"]
            result = tokenizer.convert_tokens_to_string(tokens)

            assert isinstance(result, str)


# ===== Special Tokens Tests (Parametrized) =====


@pytest.mark.parametrize(
    "token_ids_0,token_ids_1,description",
    [
        # Single sequence with BOS and EOS
        ([10, 20, 30], None, "single_sequence_with_special_tokens"),
        # Token pair with BOS and EOS
        ([10, 20, 30], [40, 50], "token_pair_with_special_tokens"),
    ],
)
def test_build_inputs_with_special_tokens(mock_sp_model, vocab_file, token_ids_0, token_ids_1, description):
    """Test build_inputs_with_special_tokens with various configurations"""
    # Mock token IDs
    mock_sp_model.piece_to_id.side_effect = lambda token: {"<s>": 1, "</s>": 2, "<pad>": 0, "<unk>": 3}.get(token, 10)

    with patch.object(SentencePieceProcessor, "Load", return_value=None):
        with patch("sentencepiece.SentencePieceProcessor") as mock_spm:
            mock_spm.return_value = mock_sp_model
            tokenizer = Ernie4_5Tokenizer(vocab_file=vocab_file, add_bos_token=True, add_eos_token=True)

            result = tokenizer.build_inputs_with_special_tokens(token_ids_0, token_ids_1)

            # Verify structure based on test case
            if token_ids_1 is None:
                # Single sequence
                expected = [1] + token_ids_0 + [2]
                assert result == expected
            else:
                # Token pair
                expected = (
                    [tokenizer.bos_token_id]
                    + token_ids_0
                    + [tokenizer.eos_token_id]
                    + [tokenizer.bos_token_id]
                    + token_ids_1
                    + [tokenizer.eos_token_id]
                )
                assert result == expected


# ===== Token Type IDs Tests (Parametrized) =====


@pytest.mark.parametrize(
    "add_bos,add_eos,token_ids_0,token_ids_1,description",
    [
        # With special tokens
        (True, True, [10, 20, 30], [40, 50], "with_special_tokens"),
        # Without special tokens
        (False, False, [10, 20, 30], [40, 50], "without_special_tokens"),
    ],
)
def test_create_token_type_ids_from_sequences(
    mock_sp_model, vocab_file, add_bos, add_eos, token_ids_0, token_ids_1, description
):
    """Test create_token_type_ids_from_sequences with and without special tokens"""
    with patch.object(SentencePieceProcessor, "Load", return_value=None):
        with patch("sentencepiece.SentencePieceProcessor") as mock_spm:
            mock_spm.return_value = mock_sp_model
            tokenizer = Ernie4_5Tokenizer(vocab_file=vocab_file, add_bos_token=add_bos, add_eos_token=add_eos)

            result = tokenizer.create_token_type_ids_from_sequences(token_ids_0, token_ids_1)

            # Verify result structure
            if add_bos and add_eos:
                # With special tokens
                first_seq_length = len([tokenizer.bos_token_id] + token_ids_0 + [tokenizer.eos_token_id])
                second_seq_length = len([tokenizer.bos_token_id] + token_ids_1 + [tokenizer.eos_token_id])
                expected_length = first_seq_length + second_seq_length
            else:
                # Without special tokens
                first_seq_length = len(token_ids_0)
                expected_length = len(token_ids_0) + len(token_ids_1)

            assert len(result) == expected_length

            # First sequence should be zeros
            assert all(x == 0 for x in result[:first_seq_length])

            # Second sequence should be ones
            assert all(x == 1 for x in result[first_seq_length:])


# ===== Save Vocabulary Tests =====


def test_save_vocabulary(mock_sp_model, tmp_path):
    """Test save_vocabulary with serialization"""
    with patch("sentencepiece.SentencePieceProcessor") as mock_spm:
        mock_spm.return_value = mock_sp_model
        mock_sp_model.serialized_model_proto.return_value = b"serialized_model"

        tokenizer = Ernie4_5Tokenizer(vocab_file="nonexistent_file.model")
        save_dir = tmp_path / "saved"
        save_dir.mkdir()

        result = tokenizer.save_vocabulary(str(save_dir))

        assert result is not None
        assert len(result) == 1
        assert os.path.exists(result[0])

        # Verify the file was created with serialized content
        with open(result[0], "rb") as f:
            content = f.read()
        assert content == b"serialized_model"
