import os
import unittest

from fastdeploy.utils import retrive_model_from_server


class TestAistudioDownload(unittest.TestCase):
    def test_retrive_model_from_server_MODELSCOPE(self):
        os.environ["FD_MODEL_SOURCE"] = "MODELSCOPE"
        os.environ["FD_MODEL_CACHE"] = "./models"

        model_name_or_path = "baidu/ERNIE-4.5-0.3B-PT"
        revision = "master"
        expected_path = f"./models/PaddlePaddle/ERNIE-4.5-0.3B-PT/{revision}"
        result = retrive_model_from_server(model_name_or_path, revision)
        self.assertEqual(expected_path, result)

        os.environ.clear()

    def test_retrive_model_from_server_unsupported_source(self):
        os.environ["FD_MODEL_SOURCE"] = "UNSUPPORTED_SOURCE"
        os.environ["FD_MODEL_CACHE"] = "./models"

        model_name_or_path = "baidu/ERNIE-4.5-0.3B-PT"
        with self.assertRaises(ValueError):
            retrive_model_from_server(model_name_or_path)

        os.environ.clear()

    def test_retrive_model_from_server_model_not_exist(self):
        os.environ["FD_MODEL_SOURCE"] = "MODELSCOPE"
        os.environ["FD_MODEL_CACHE"] = "./models"

        model_name_or_path = "non_existing_model"

        with self.assertRaises(Exception):
            retrive_model_from_server(model_name_or_path)

        os.environ.clear()


if __name__ == "__main__":
    unittest.main()
