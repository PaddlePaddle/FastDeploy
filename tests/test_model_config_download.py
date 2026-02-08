import unittest
from unittest.mock import patch
from fastdeploy.config import ModelConfig

class TestModelConfigDownload(unittest.TestCase):
    @patch('fastdeploy.config.retrive_model_from_server')
    def test_get_download_model(self, mock_retrieve):
        # Patch __init__ to avoid side effects during instantiation
        with patch.object(ModelConfig, '__init__', return_value=None):
            config = ModelConfig({})

            model_name = "test/model"
            mock_retrieve.return_value = "/path/to/test/model"

            result = config._get_download_model(model_name)

            mock_retrieve.assert_called_once_with(model_name)
            self.assertEqual(result, "/path/to/test/model")

if __name__ == '__main__':
    unittest.main()
