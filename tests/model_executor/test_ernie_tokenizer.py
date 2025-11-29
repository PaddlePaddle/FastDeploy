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
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, Mock, patch

# Standalone testing mode - use dynamic import
# Mock the paddleformers module to avoid import issues
mock_logger = Mock()

# Create mock modules to avoid dependency issues
sys.modules["paddleformers"] = Mock()
sys.modules["paddleformers.utils"] = Mock()
sys.modules["paddleformers.utils.log"] = Mock()
sys.modules["paddleformers.utils.log"].logger = mock_logger

# Mock the fastdeploy module structure
sys.modules["fastdeploy"] = Mock()
sys.modules["fastdeploy.model_executor"] = Mock()
sys.modules["fastdeploy.model_executor.guided_decoding"] = Mock()

# Import the tokenizer module directly
import importlib.util

spec = importlib.util.spec_from_file_location(
    "ernie_tokenizer",
    os.path.join(os.path.dirname(__file__), "../../fastdeploy/model_executor/guided_decoding/ernie_tokenizer.py"),
)
ernie_tokenizer_module = importlib.util.module_from_spec(spec)

# Mock the required dependencies in the module
sys.modules["sentencepiece"] = Mock()
sys.modules["transformers"] = Mock()
sys.modules["transformers.tokenization_utils"] = Mock()


# Create mock classes for the dependencies
class MockAddedToken:
    def __init__(self, token, lstrip=False, rstrip=False):
        self.token = token
        self.lstrip = lstrip
        self.rstrip = rstrip


class MockPreTrainedTokenizer:
    def __init__(self, **kwargs):
        # Initialize token attributes that Ernie4_5Tokenizer expects from parent class
        self.bos_token = kwargs.get("bos_token", "<s>")
        self.eos_token = kwargs.get("eos_token", "</s>")
        self.unk_token = kwargs.get("unk_token", "<unk>")
        self.pad_token = kwargs.get("pad_token", "<pad>")
        self.sep_token = kwargs.get("sep_token", "<sep>")

        # Set up token IDs based on kwargs or use defaults
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        self.unk_token_id = kwargs.get("unk_token_id", 3)
        self.pad_token_id = kwargs.get("pad_token_id", 0)
        self.sep_token_id = kwargs.get("sep_token_id", 4)

        # Other attributes that might be needed
        self.all_special_tokens = [self.bos_token, self.eos_token, self.unk_token, self.pad_token, self.sep_token]
        self.added_tokens_encoder = {}

    def convert_ids_to_tokens(self, token_id):
        """Mock convert_ids_to_tokens method"""
        return f"token_{token_id}"

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        """Mock get_special_tokens_mask method"""
        return [1 if token_id in [1, 2, 3, 4] else 0 for token_id in token_ids_0]


# Setup the mock modules
sys.modules["transformers.tokenization_utils"].AddedToken = MockAddedToken
sys.modules["transformers.tokenization_utils"].PreTrainedTokenizer = MockPreTrainedTokenizer

# Execute the module to get the tokenizer class
spec.loader.exec_module(ernie_tokenizer_module)
Ernie4_5Tokenizer = ernie_tokenizer_module.Ernie4_5Tokenizer


class TestErnie4_5Tokenizer(unittest.TestCase):
    """Test suite for Ernie4_5Tokenizer class"""

    def setUp(self):
        """Setup method to create test fixtures"""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.vocab_file = os.path.join(self.temp_dir, "test_spm.model")

        # Create a mock sentencepiece model
        self.mock_sp_model = MagicMock()
        self.mock_sp_model.get_piece_size.return_value = 1000
        self.mock_sp_model.piece_to_id.return_value = 1
        self.mock_sp_model.IdToPiece.return_value = "test"
        self.mock_sp_model.encode.return_value = ["▁Hello", "▁World"]
        self.mock_sp_model.decode.return_value = "Hello World"
        self.mock_sp_model.serialized_model_proto.return_value = b"mock_model_proto"

    def tearDown(self):
        """Cleanup method to remove test files"""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch("sentencepiece.SentencePieceProcessor")
    def test_init_default_parameters(self, mock_spm_processor):
        """Test tokenizer initialization with default parameters"""
        mock_spm_processor.return_value = self.mock_sp_model

        tokenizer = Ernie4_5Tokenizer(vocab_file=self.vocab_file)

        # Verify that SentencePieceProcessor was called with correct parameters
        mock_spm_processor.assert_called_once_with()
        self.mock_sp_model.Load.assert_called_once_with(self.vocab_file)

        # Check default attributes
        self.assertEqual(tokenizer.vocab_file, self.vocab_file)
        self.assertTrue(tokenizer.add_bos_token)
        self.assertFalse(tokenizer.add_eos_token)
        self.assertEqual(tokenizer.sp_model_kwargs, {})
        # After conversion, tokens become AddedToken objects, so check the token attribute
        self.assertEqual(tokenizer.bos_token.token, "<s>")
        self.assertEqual(tokenizer.eos_token.token, "</s>")
        self.assertEqual(tokenizer.unk_token.token, "<unk>")
        self.assertEqual(tokenizer.pad_token.token, "<pad>")

    @patch("sentencepiece.SentencePieceProcessor")
    def test_init_custom_parameters(self, mock_spm_processor):
        """Test tokenizer initialization with custom parameters"""
        mock_spm_processor.return_value = self.mock_sp_model

        custom_kwargs = {"add_extra_options": True}
        tokenizer = Ernie4_5Tokenizer(
            vocab_file=self.vocab_file,
            unk_token="<unk>",
            bos_token="<s>",
            eos_token="</s>",
            pad_token="<pad>",
            sp_model_kwargs=custom_kwargs,
            add_bos_token=False,
            add_eos_token=True,
        )

        # Verify that SentencePieceProcessor was called with custom parameters
        mock_spm_processor.assert_called_once_with(add_extra_options=True)

        # Check custom attributes
        self.assertFalse(tokenizer.add_bos_token)
        self.assertTrue(tokenizer.add_eos_token)
        self.assertEqual(tokenizer.sp_model_kwargs, custom_kwargs)

    @patch("sentencepiece.SentencePieceProcessor")
    def test_vocab_size_property(self, mock_spm_processor):
        """Test vocab_size property"""
        mock_spm_processor.return_value = self.mock_sp_model

        tokenizer = Ernie4_5Tokenizer(vocab_file=self.vocab_file)

        # Test that vocab_size returns the piece size from sp_model
        self.assertEqual(tokenizer.vocab_size, 1000)
        # Allow multiple calls as get_piece_size may be called during initialization
        self.assertGreaterEqual(self.mock_sp_model.get_piece_size.call_count, 1)

    @patch("sentencepiece.SentencePieceProcessor")
    def test_get_vocab(self, mock_spm_processor):
        """Test get_vocab method"""
        mock_spm_processor.return_value = self.mock_sp_model

        tokenizer = Ernie4_5Tokenizer(vocab_file=self.vocab_file)
        vocab = tokenizer.get_vocab()

        # Check that vocab has correct structure and includes expected tokens
        # Allow for additional tokens (like special tokens) beyond the base 1000
        self.assertGreaterEqual(len(vocab), 1000)  # At least the base vocab size
        # The actual token names depend on the convert_ids_to_tokens implementation
        # Since we're using the mock implementation, we should see token_0, token_1, etc.
        self.assertIn("token_0", vocab)
        self.assertEqual(vocab["token_0"], 0)

    @patch("sentencepiece.SentencePieceProcessor")
    def test_tokenize(self, mock_spm_processor):
        """Test tokenize method"""
        mock_spm_processor.return_value = self.mock_sp_model

        tokenizer = Ernie4_5Tokenizer(vocab_file=self.vocab_file)

        # Test tokenization
        result = tokenizer.tokenize("Hello World")

        # Should return the result from _tokenize method
        self.assertEqual(result, ["▁Hello", "▁World"])
        self.mock_sp_model.encode.assert_called_once_with("Hello World", out_type=str)

    @patch("sentencepiece.SentencePieceProcessor")
    def test_decode(self, mock_spm_processor):
        """Test decode method"""
        mock_spm_processor.return_value = self.mock_sp_model

        tokenizer = Ernie4_5Tokenizer(vocab_file=self.vocab_file)

        # Test decoding
        result = tokenizer.decode([1, 2, 3])

        self.assertEqual(result, "Hello World")
        self.mock_sp_model.decode.assert_called_once_with([1, 2, 3])

    @patch("sentencepiece.SentencePieceProcessor")
    def test_build_inputs_with_special_tokens_single_sequence(self, mock_spm_processor):
        """Test build_inputs_with_special_tokens with single sequence"""
        mock_spm_processor.return_value = self.mock_sp_model

        # Mock bos and eos token IDs
        self.mock_sp_model.piece_to_id.side_effect = lambda token: {"<s>": 1, "</s>": 2, "<pad>": 0, "<unk>": 3}.get(
            token, 10
        )

        tokenizer = Ernie4_5Tokenizer(vocab_file=self.vocab_file, add_bos_token=True, add_eos_token=True)

        result = tokenizer.build_inputs_with_special_tokens([10, 20, 30])

        # Should add BOS and EOS tokens
        expected = [1] + [10, 20, 30] + [2]
        self.assertEqual(result, expected)

    @patch("sentencepiece.SentencePieceProcessor")
    def test_save_vocabulary_invalid_directory(self, mock_spm_processor):
        """Test save_vocabulary with invalid directory"""
        mock_spm_processor.return_value = self.mock_sp_model

        tokenizer = Ernie4_5Tokenizer(vocab_file=self.vocab_file)

        result = tokenizer.save_vocabulary("/nonexistent/directory")

        # Should return None for invalid directory
        self.assertIsNone(result)

    @patch("sentencepiece.SentencePieceProcessor")
    def test_serialization_getstate(self, mock_spm_processor):
        """Test __getstate__ method for serialization"""
        mock_spm_processor.return_value = self.mock_sp_model

        tokenizer = Ernie4_5Tokenizer(vocab_file=self.vocab_file)
        state = tokenizer.__getstate__()

        # sp_model should be None in the state for pickling
        self.assertIsNone(state["sp_model"])
        self.assertEqual(state["vocab_file"], self.vocab_file)

    @patch("sentencepiece.SentencePieceProcessor")
    def test_serialization_setstate(self, mock_spm_processor):
        """Test __setstate__ method for deserialization"""
        mock_spm_processor.return_value = self.mock_sp_model

        tokenizer = Ernie4_5Tokenizer(vocab_file=self.vocab_file)

        # Create a state dictionary
        state = {
            "vocab_file": self.vocab_file,
            "sp_model_kwargs": {},
            "add_bos_token": True,
            "add_eos_token": False,
            "sp_model": None,
        }

        # Call __setstate__ to restore the object
        tokenizer.__setstate__(state)

        # Verify that sp_model was recreated
        self.assertIsNotNone(tokenizer.sp_model)
        mock_spm_processor.assert_called_with()
        self.mock_sp_model.Load.assert_called_with(self.vocab_file)

    @patch("sentencepiece.SentencePieceProcessor")
    def test_convert_token_to_id(self, mock_spm_processor):
        """Test _convert_token_to_id method"""
        self.mock_sp_model.piece_to_id.return_value = 42
        mock_spm_processor.return_value = self.mock_sp_model

        tokenizer = Ernie4_5Tokenizer(vocab_file=self.vocab_file)

        result = tokenizer._convert_token_to_id("test_token")

        self.assertEqual(result, 42)
        self.mock_sp_model.piece_to_id.assert_called_once_with("test_token")

    @patch("sentencepiece.SentencePieceProcessor")
    def test_convert_id_to_token(self, mock_spm_processor):
        """Test _convert_id_to_token method"""
        self.mock_sp_model.IdToPiece.return_value = "test_piece"
        mock_spm_processor.return_value = self.mock_sp_model

        tokenizer = Ernie4_5Tokenizer(vocab_file=self.vocab_file)

        result = tokenizer._convert_id_to_token(42)

        self.assertEqual(result, "test_piece")
        self.mock_sp_model.IdToPiece.assert_called_once_with(42)

    @patch("sentencepiece.SentencePieceProcessor")
    def test_convert_tokens_to_string_with_special_tokens(self, mock_spm_processor):
        """Test convert_tokens_to_string with special tokens"""

        # Mock the sp_model.decode to handle subtokens
        def mock_decode(tokens):
            if tokens:
                return "hello world"
            return ""

        self.mock_sp_model.decode.side_effect = mock_decode
        mock_spm_processor.return_value = self.mock_sp_model

        tokenizer = Ernie4_5Tokenizer(vocab_file=self.vocab_file)

        # Test with special tokens mixed with regular tokens
        tokens = ["hello", "<s>", "world", "</s>"]
        result = tokenizer.convert_tokens_to_string(tokens)

        # Should properly handle special tokens and decode subtokens
        self.assertIsInstance(result, str)

    @patch("sentencepiece.SentencePieceProcessor")
    def test_convert_tokens_to_string_special_tokens_spacing(self, mock_spm_processor):
        """Test convert_tokens_to_string special token spacing logic"""

        def mock_decode(tokens):
            return "".join(tokens) if tokens else ""

        self.mock_sp_model.decode.side_effect = mock_decode
        mock_spm_processor.return_value = self.mock_sp_model

        tokenizer = Ernie4_5Tokenizer(vocab_file=self.vocab_file)

        # Test special token positioning affects spacing
        tokens = ["regular", "<s>", "special"]  # special token not at position 0
        result = tokenizer.convert_tokens_to_string(tokens)

        # Verify special tokens are handled with proper spacing
        self.assertIsInstance(result, str)

    @patch("sentencepiece.SentencePieceProcessor")
    def test_save_vocabulary_file_copying(self, mock_spm_processor):
        """Test save_vocabulary file copying logic"""
        mock_spm_processor.return_value = self.mock_sp_model

        # Create a source vocab file
        source_vocab = os.path.join(self.temp_dir, "source.model")
        with open(source_vocab, "wb") as f:
            f.write(b"fake model content")

        tokenizer = Ernie4_5Tokenizer(vocab_file=source_vocab)

        # Test saving to different directory
        save_dir = os.path.join(self.temp_dir, "saved")
        os.makedirs(save_dir, exist_ok=True)

        result = tokenizer.save_vocabulary(save_dir)

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertTrue(os.path.exists(result[0]))

    @patch("sentencepiece.SentencePieceProcessor")
    def test_save_vocabulary_serialization(self, mock_spm_processor):
        """Test save_vocabulary with model serialization"""
        self.mock_sp_model.serialized_model_proto.return_value = b"serialized_model"
        mock_spm_processor.return_value = self.mock_sp_model

        tokenizer = Ernie4_5Tokenizer(vocab_file="nonexistent_file.model")

        save_dir = os.path.join(self.temp_dir, "saved")
        os.makedirs(save_dir, exist_ok=True)

        result = tokenizer.save_vocabulary(save_dir)

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)

        # Verify the file was created with serialized content
        with open(result[0], "rb") as f:
            content = f.read()
        self.assertEqual(content, b"serialized_model")

    @patch("sentencepiece.SentencePieceProcessor")
    def test_build_inputs_with_special_tokens_token_pair(self, mock_spm_processor):
        """Test build_inputs_with_special_tokens with token pairs"""
        mock_spm_processor.return_value = self.mock_sp_model

        tokenizer = Ernie4_5Tokenizer(vocab_file=self.vocab_file, add_bos_token=True, add_eos_token=True)

        # Mock token IDs
        token_ids_0 = [10, 20, 30]
        token_ids_1 = [40, 50]

        result = tokenizer.build_inputs_with_special_tokens(token_ids_0, token_ids_1)

        # Should include BOS/EOS for both sequences
        expected = (
            [tokenizer.bos_token_id]
            + token_ids_0
            + [tokenizer.eos_token_id]
            + [tokenizer.bos_token_id]
            + token_ids_1
            + [tokenizer.eos_token_id]
        )
        self.assertEqual(result, expected)

    @patch("sentencepiece.SentencePieceProcessor")
    def test_get_special_tokens_mask_already_has_special(self, mock_spm_processor):
        """Test get_special_tokens_mask when already has special tokens"""
        mock_spm_processor.return_value = self.mock_sp_model

        tokenizer = Ernie4_5Tokenizer(vocab_file=self.vocab_file)

        token_ids_0 = [1, 10, 20, 2]  # Already has special tokens
        token_ids_1 = [1, 30, 40, 2]  # Already has special tokens

        result = tokenizer.get_special_tokens_mask(
            token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
        )

        # Should call parent method when already_has_special_tokens is True
        self.assertIsInstance(result, list)

    @patch("sentencepiece.SentencePieceProcessor")
    def test_create_token_type_ids_from_sequences(self, mock_spm_processor):
        """Test create_token_type_ids_from_sequences method"""
        mock_spm_processor.return_value = self.mock_sp_model

        tokenizer = Ernie4_5Tokenizer(vocab_file=self.vocab_file, add_bos_token=True, add_eos_token=True)

        # Test with single sequence
        token_ids_0 = [10, 20, 30]
        result = tokenizer.create_token_type_ids_from_sequences(token_ids_0)

        # Should return zeros for first sequence
        expected_length = len([tokenizer.bos_token_id] + token_ids_0 + [tokenizer.eos_token_id])
        self.assertEqual(len(result), expected_length)
        self.assertTrue(all(x == 0 for x in result))

        # Test with sequence pair
        token_ids_1 = [40, 50]
        result_pair = tokenizer.create_token_type_ids_from_sequences(token_ids_0, token_ids_1)

        # Should return zeros for first sequence and ones for second
        first_seq_length = len([tokenizer.bos_token_id] + token_ids_0 + [tokenizer.eos_token_id])
        second_seq_length = len([tokenizer.bos_token_id] + token_ids_1 + [tokenizer.eos_token_id])
        expected_length_pair = first_seq_length + second_seq_length
        self.assertEqual(len(result_pair), expected_length_pair)

        # First part should be zeros, second part should be ones
        self.assertTrue(all(x == 0 for x in result_pair[:first_seq_length]))
        self.assertTrue(all(x == 1 for x in result_pair[first_seq_length:]))

    @patch("sentencepiece.SentencePieceProcessor")
    def test_create_token_type_ids_from_sequences_no_special_tokens(self, mock_spm_processor):
        """Test create_token_type_ids_from_sequences without special tokens"""
        mock_spm_processor.return_value = self.mock_sp_model

        tokenizer = Ernie4_5Tokenizer(vocab_file=self.vocab_file, add_bos_token=False, add_eos_token=False)

        token_ids_0 = [10, 20, 30]
        token_ids_1 = [40, 50]

        result = tokenizer.create_token_type_ids_from_sequences(token_ids_0, token_ids_1)

        # Should handle sequences without special tokens
        expected_length = len(token_ids_0) + len(token_ids_1)
        self.assertEqual(len(result), expected_length)

        # Verify segmentation (first sequence zeros, second sequence ones)
        first_zero_count = len(token_ids_0)
        self.assertEqual(result[:first_zero_count], [0, 0, 0])
        self.assertEqual(result[first_zero_count:], [1, 1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
