import json
from pathlib import Path
from typing import Dict, Optional, Union

from fastdeploy.utils import get_logger

logger = get_logger("transformer_config", "transformer_config.log")


def file_or_path_exists(model, config_name):
    if (local_path := Path(model)).exists():
        return (local_path / config_name).is_file()

    return False


def get_pooling_config_name(pooling_name: str):

    if "pooling_mode_" in pooling_name:
        pooling_name = pooling_name.replace("pooling_mode_", "")

    if "_" in pooling_name:
        pooling_name = pooling_name.split("_")[0]
    print("pooling_name", pooling_name)

    if "lasttoken" in pooling_name:
        pooling_name = "last"

    supported_pooling_types = ["LAST", "ALL", "CLS", "STEP", "MEAN"]
    pooling_type_name = pooling_name.upper()

    if pooling_type_name in supported_pooling_types:
        return pooling_type_name

    raise NotImplementedError(f"Pooling type {pooling_type_name} not supported")


def get_hf_file_to_dict(file_name: str, model: Union[str, Path]) -> Optional[Dict]:
    """
    Load a file from model directory and return its contents as a dictionary.

    Args:
        file_name (str): Name of the file to load
        model (Union[str, Path]): Model path or identifier
        revision (str, optional): Model revision. Defaults to 'main'.

    Returns:
        Optional[Dict]: File contents as dictionary, None if not found
    """
    model_path = Path(model)

    # Check if it's a local path
    if model_path.exists():
        file_path = model_path / file_name
        if file_path.is_file():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load {file_name}: {e}")
                return None

    # TODO: Add remote model file downloading logic here
    # This would depend on your model repository system
    return None


def get_pooling_config(model: str, revision: Optional[str] = "main"):
    """
    This function gets the pooling and normalize
    config from the model - only applies to
    sentence-transformers models.

    Args:
        model (str): The name of the Hugging Face model.
        revision (str, optional): The specific version
        of the model to use. Defaults to 'main'.

    Returns:
        dict: A dictionary containing the pooling
        type and whether normalization is used.
    """

    modules_file_name = "modules.json"
    modules_dict = None
    if file_or_path_exists(model, config_name=modules_file_name):
        modules_dict = get_hf_file_to_dict(modules_file_name, model)

    if modules_dict is None:
        return None

    logger.info("Found sentence-transformers modules configuration.")

    pooling = next((item for item in modules_dict if item["type"] == "sentence_transformers.models.Pooling"), None)

    normalize = bool(
        next((item for item in modules_dict if item["type"] == "sentence_transformers.models.Normalize"), False)
    )

    if pooling:
        pooling_file_name = "{}/config.json".format(pooling["path"])
        pooling_dict = get_hf_file_to_dict(pooling_file_name, model)
        logger.info(f"pooling_dict:{pooling_dict}")
        pooling_type_name = next((item for item, val in pooling_dict.items() if val is True), None)

        if pooling_type_name is not None:
            pooling_type_name = get_pooling_config_name(pooling_type_name)

        logger.info("Found pooling configuration.")
        return {"pooling_type": pooling_type_name, "normalize": normalize}

    return None
