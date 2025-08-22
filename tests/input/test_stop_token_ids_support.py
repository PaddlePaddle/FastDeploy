import unittest
from unittest.mock import MagicMock, patch
import numpy as np


class TestStopTokenIdsSupport(unittest.TestCase):
    """Test cases for stop_token_ids parameter support in data processors."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock tokenizer
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.eos_token_id = 1
        self.mock_tokenizer.convert_tokens_to_ids.side_effect = lambda tokens: list(range(len(tokens)))
        self.mock_tokenizer.tokenize.side_effect = lambda text: text.split()

    def _create_mock_processor(self, processor_class):
        """Create a mock processor with necessary attributes."""
        with patch.object(processor_class, "__init__", return_value=None):
            processor = processor_class("model_path")
            processor.tokenizer = self.mock_tokenizer
            processor.eos_token_ids = [self.mock_tokenizer.eos_token_id]
            
            # Mock the _apply_default_parameters method
            processor._apply_default_parameters = lambda x: x
            
            # Mock text2ids and messages2ids for text processing
            processor.text2ids = lambda text, max_len: np.array([1, 2, 3, 4])
            processor.messages2ids = lambda msgs: np.array([1, 2, 3, 4])
            
            return processor

    def test_text_processor_stop_token_ids_only(self):
        """Test TextProcessor with only stop_token_ids parameter."""
        from fastdeploy.input.text_processor import DataProcessor
        
        processor = self._create_mock_processor(DataProcessor)
        
        request = {
            "stop_token_ids": [100, 200, 300],
            "prompt": "Test prompt"
        }
        
        processor.process_request_dict(request, max_model_len=2048)
        
        # Check that stop_token_ids are properly processed
        self.assertIn("stop_token_ids", request)
        self.assertIn("stop_seqs_len", request)
        
        # Should have 3 stop token sequences (one for each token id)
        expected_structure = [[100], [200], [300]]
        self.assertEqual(request["stop_token_ids"], expected_structure)
        self.assertEqual(request["stop_seqs_len"], 3)

    def test_text_processor_stop_sequences_only(self):
        """Test TextProcessor with only stop sequences parameter."""
        from fastdeploy.input.text_processor import DataProcessor
        
        processor = self._create_mock_processor(DataProcessor)
        
        request = {
            "stop": ["</s>", "END"],
            "prompt": "Test prompt"
        }
        
        processor.process_request_dict(request, max_model_len=2048)
        
        # Check that stop sequences are converted to token ids
        self.assertIn("stop_token_ids", request)
        self.assertIn("stop_seqs_len", request)
        
        # Should have 2 stop token sequences (converted from strings)
        self.assertEqual(len(request["stop_token_ids"]), 2)
        self.assertEqual(request["stop_seqs_len"], 2)

    def test_text_processor_both_stop_types(self):
        """Test TextProcessor with both stop sequences and stop_token_ids."""
        from fastdeploy.input.text_processor import DataProcessor
        
        processor = self._create_mock_processor(DataProcessor)
        
        request = {
            "stop": ["</s>"],
            "stop_token_ids": [100, 200],
            "prompt": "Test prompt"
        }
        
        processor.process_request_dict(request, max_model_len=2048)
        
        # Check that both types are merged
        self.assertIn("stop_token_ids", request)
        self.assertIn("stop_seqs_len", request)
        
        # Should have 3 sequences: 1 from string conversion + 2 from token ids
        self.assertEqual(len(request["stop_token_ids"]), 3)
        self.assertEqual(request["stop_seqs_len"], 3)

    def test_ernie_processor_stop_token_ids_only(self):
        """Test ErnieProcessor with only stop_token_ids parameter."""
        from fastdeploy.input.ernie_processor import ErnieProcessor
        
        processor = self._create_mock_processor(ErnieProcessor)
        
        request = {
            "stop_token_ids": [100, 200, 300],
            "prompt": "Test prompt"
        }
        
        processor.process_request_dict(request, max_model_len=2048)
        
        # Check that stop_token_ids are properly processed
        self.assertIn("stop_token_ids", request)
        self.assertIn("stop_seqs_len", request)
        
        # Should have 3 stop token sequences
        expected_structure = [[100], [200], [300]]
        self.assertEqual(request["stop_token_ids"], expected_structure)
        self.assertEqual(request["stop_seqs_len"], 3)

    def test_ernie_processor_both_stop_types(self):
        """Test ErnieProcessor with both stop sequences and stop_token_ids."""
        from fastdeploy.input.ernie_processor import ErnieProcessor
        
        processor = self._create_mock_processor(ErnieProcessor)
        
        request = {
            "stop": ["</s>"],
            "stop_token_ids": [100, 200],
            "prompt": "Test prompt"
        }
        
        processor.process_request_dict(request, max_model_len=2048)
        
        # Check that both types are merged
        self.assertIn("stop_token_ids", request)
        self.assertIn("stop_seqs_len", request)
        
        # Should have 3 sequences: 1 from string conversion + 2 from token ids
        self.assertEqual(len(request["stop_token_ids"]), 3)
        self.assertEqual(request["stop_seqs_len"], 3)

    def test_stop_token_ids_as_lists(self):
        """Test handling of stop_token_ids provided as lists."""
        from fastdeploy.input.text_processor import DataProcessor
        
        processor = self._create_mock_processor(DataProcessor)
        
        request = {
            "stop_token_ids": [[100], [200, 201]],
            "prompt": "Test prompt"
        }
        
        processor.process_request_dict(request, max_model_len=2048)
        
        # Check that list format is preserved and padded correctly
        self.assertIn("stop_token_ids", request)
        self.assertIn("stop_seqs_len", request)
        
        # Should have 2 sequences, padded to same length
        self.assertEqual(len(request["stop_token_ids"]), 2)
        self.assertEqual(request["stop_seqs_len"], 2)
        
        # Check padding with -1
        result = request["stop_token_ids"]
        self.assertEqual(result[0], [100, -1])  # Padded to length 2
        self.assertEqual(result[1], [200, 201])  # Original length 2

    def test_no_stop_parameters(self):
        """Test that processing works when no stop parameters are provided."""
        from fastdeploy.input.text_processor import DataProcessor
        
        processor = self._create_mock_processor(DataProcessor)
        
        request = {
            "prompt": "Test prompt"
        }
        
        processor.process_request_dict(request, max_model_len=2048)
        
        # Should not have stop_token_ids or stop_seqs_len when no stop parameters provided
        self.assertNotIn("stop_token_ids", request)
        self.assertNotIn("stop_seqs_len", request)


if __name__ == "__main__":
    unittest.main()