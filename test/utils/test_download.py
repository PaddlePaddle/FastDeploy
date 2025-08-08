import os
import unittest

from fastdeploy.utils import retrive_model_from_server


class TestAistudioDownload(unittest.TestCase):
    """
    Test cases for downloading models from different sources using FastDeploy utilities.
    """

    def test_retrive_model_from_server_unsupported_source(self):
        """
        Test case for retrieving a model from an unsupported source.
        """
        os.environ["FD_MODEL_SOURCE"] = "UNSUPPORTED_SOURCE"
        os.environ["FD_MODEL_CACHE"] = "./models"

        model_name_or_path = "baidu/ERNIE-4.5-0.3B-PT"
        with self.assertRaises(ValueError):
            retrive_model_from_server(model_name_or_path)

        os.environ.clear()

    def test_retrive_model_from_modelscope_server_model_not_exist(self):
        """
        Test case for retrieving a model from ModelScope server when it doesn't exist.
        """
        os.environ["FD_MODEL_SOURCE"] = "MODELSCOPE"
        os.environ["FD_MODEL_CACHE"] = "./model"

        model_name_or_path = "non_existing_model_modelscope"

        with self.assertRaises(Exception):
            retrive_model_from_server(model_name_or_path)

        os.environ.clear()

    def test_retrive_model_from_huggingface_server_model_not_exist(self):
        """
        Test case for retrieving a model from Hugging Face server when it doesn't exist.
        """
        os.environ["FD_MODEL_SOURCE"] = "HUGGINGFACE"
        os.environ["FD_MODEL_CACHE"] = "./models"

        model_name_or_path = "non_existing_model_hf"

        with self.assertRaises(Exception):
            retrive_model_from_server(model_name_or_path)

        os.environ.clear()


if __name__ == "__main__":
    unittest.main()
