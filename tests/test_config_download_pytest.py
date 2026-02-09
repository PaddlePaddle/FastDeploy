from unittest.mock import MagicMock, patch

import fastdeploy.config


def test_get_download_model_default_revision():
    """
    Test that _get_download_model calls retrive_model_from_server with revision="master" when model_type="default".
    """
    # Mock dependencies of ModelConfig.__init__ to allow instantiation
    with (
        patch("fastdeploy.config.PretrainedConfig.get_config_dict", return_value=({}, None)),
        patch("fastdeploy.config.PretrainedConfig.from_dict", return_value=MagicMock()),
        patch.object(fastdeploy.config.ModelConfig, "_post_init"),
    ):
        # 'model' is required by __init__ assertion
        config = fastdeploy.config.ModelConfig({"model": "fake-model-name"})

        with patch("fastdeploy.config.retrive_model_from_server") as mock_retrieve:
            expected_path = "/tmp/fake/model/path"
            mock_retrieve.return_value = expected_path

            model_name = "fake-model-name"
            # Default model_type is "default"
            result = config._get_download_model(model_name)

            assert result == expected_path
            mock_retrieve.assert_called_once_with(model_name, revision="master")


def test_get_download_model_custom_revision():
    """
    Test that _get_download_model calls retrive_model_from_server with custom revision when model_type is provided.
    """
    # Mock dependencies of ModelConfig.__init__ to allow instantiation
    with (
        patch("fastdeploy.config.PretrainedConfig.get_config_dict", return_value=({}, None)),
        patch("fastdeploy.config.PretrainedConfig.from_dict", return_value=MagicMock()),
        patch.object(fastdeploy.config.ModelConfig, "_post_init"),
    ):
        config = fastdeploy.config.ModelConfig({"model": "fake-model-name"})

        with patch("fastdeploy.config.retrive_model_from_server") as mock_retrieve:
            expected_path = "/tmp/fake/model/v2"
            mock_retrieve.return_value = expected_path

            model_name = "fake-model-name"
            model_type = "v2.0"
            result = config._get_download_model(model_name, model_type=model_type)

            assert result == expected_path
            mock_retrieve.assert_called_once_with(model_name, revision="v2.0")
