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
import unittest

from fastdeploy.utils import init_bos_client, retrive_model_from_server


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


class TestInitBosClient(unittest.TestCase):
    """
    Test cases for initializing Baidu Object Storage (BOS) client using FastDeploy utilities.
    """

    def test_init_bos_client_success(self):
        """
        Test case for successful initialization of BOS client with valid environment variables.
        """
        from unittest.mock import MagicMock, patch

        # Mock BosClient and related dependencies to skip validation
        with patch("baidubce.auth.bce_credentials.BceCredentials") as mock_credentials:
            with patch("baidubce.bce_client_configuration.BceClientConfiguration") as mock_config:
                with patch("baidubce.services.bos.bos_client.BosClient") as mock_bos_client:
                    # Additional mock to make list_buckets call succeed
                    mock_bos_client_instance = MagicMock()
                    # Mock the list_buckets call to return empty buckets list
                    mock_bos_client_instance.list_buckets = MagicMock(return_value=MagicMock())
                    mock_bos_client.return_value = mock_bos_client_instance

                    # Mock the credentials and config
                    mock_credentials_instance = MagicMock()
                    mock_credentials.return_value = mock_credentials_instance

                    mock_config_instance = MagicMock()
                    mock_config.return_value = mock_config_instance

                    # Set up valid environment variables
                    os.environ["ENCODE_FEATURE_BOS_AK"] = "test_access_key"
                    os.environ["ENCODE_FEATURE_BOS_SK"] = "test_secret_key"
                    os.environ["ENCODE_FEATURE_ENDPOINT"] = "http://test.endpoint.com"

                    try:
                        # Call the function
                        client = init_bos_client()

                        # Verify that BosClient was created with correct arguments
                        mock_credentials.assert_called_once_with("test_access_key", "test_secret_key")
                        mock_config.assert_called_once_with(
                            credentials=mock_credentials_instance, endpoint="http://test.endpoint.com"
                        )
                        mock_bos_client.assert_called_once_with(mock_config_instance)

                        # Verify that the returned client is the mock instance
                        self.assertEqual(client, mock_bos_client_instance)
                    finally:
                        os.environ.clear()

    def test_init_bos_client_missing_envs(self):
        """
        Test case for initializing BOS client when necessary environment variables are missing.
        """
        # Test with empty environment variables
        os.environ["ENCODE_FEATURE_BOS_AK"] = ""
        os.environ["ENCODE_FEATURE_BOS_SK"] = ""
        os.environ["ENCODE_FEATURE_ENDPOINT"] = ""

        with self.assertRaises(Exception) as context:
            init_bos_client()
        self.assertIn("Create BOSClient Error, Please check your ENV", str(context.exception))
        os.environ.clear()


if __name__ == "__main__":
    unittest.main()
