import json
from pathlib import Path
from typing import Optional, Union

import huggingface_hub
from huggingface_hub import hf_hub_download, try_to_load_from_cache
from huggingface_hub.utils import (
    EntryNotFoundError,
    HfHubHTTPError,
    LocalEntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

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

    if "lasttoken" in pooling_name:
        pooling_name = "last"

    supported_pooling_types = ["LAST", "ALL", "CLS", "STEP", "MEAN"]
    pooling_type_name = pooling_name.upper()

    if pooling_type_name in supported_pooling_types:
        return pooling_type_name

    raise NotImplementedError(f"Pooling type {pooling_type_name} not supported")


def try_get_local_file(model: Union[str, Path], file_name: str, revision: Optional[str] = "main") -> Optional[Path]:
    file_path = Path(model) / file_name
    if file_path.is_file():
        return file_path
    else:
        try:
            cached_filepath = try_to_load_from_cache(repo_id=model, filename=file_name, revision=revision)
            if isinstance(cached_filepath, str):
                return Path(cached_filepath)
        except ValueError:
            ...
    return None


def get_hf_file_to_dict(file_name: str, model: Union[str, Path], revision: Optional[str] = "main"):
    """
    Downloads a file from the Hugging Face Hub and returns
    its contents as a dictionary.

    Parameters:
    - file_name (str): The name of the file to download.
    - model (str): The name of the model on the Hugging Face Hub.
    - revision (str): The specific version of the model.

    Returns:
    - config_dict (dict): A dictionary containing
    the contents of the downloaded file.
    """
    file_path = try_get_local_file(model=model, file_name=file_name, revision=revision)

    if file_path is None:
        try:
            hf_hub_file = hf_hub_download(model, file_name, revision=revision)
        except huggingface_hub.errors.OfflineModeIsEnabled:
            return None
        except (RepositoryNotFoundError, RevisionNotFoundError, EntryNotFoundError, LocalEntryNotFoundError) as e:
            logger.debug("File or repository not found in hf_hub_download", e)
            return None
        except HfHubHTTPError as e:
            logger.warning(
                "Cannot connect to Hugging Face Hub. Skipping file " "download for '%s':", file_name, exc_info=e
            )
            return None
        file_path = Path(hf_hub_file)

    if file_path is not None and file_path.is_file():
        with open(file_path) as file:
            return json.load(file)

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

    pooling = next((item for item in modules_dict if item["type"] == "sentence_transformers.models.Pooling"), None)

    normalize = bool(
        next((item for item in modules_dict if item["type"] == "sentence_transformers.models.Normalize"), False)
    )

    if pooling:
        pooling_file_name = "{}/config.json".format(pooling["path"])
        pooling_dict = get_hf_file_to_dict(pooling_file_name, model)
        pooling_type_name = next((item for item, val in pooling_dict.items() if val is True), None)

        if pooling_type_name is not None:
            pooling_type_name = get_pooling_config_name(pooling_type_name)

        return {"pooling_type": pooling_type_name, "normalize": normalize}

    return None
