from paddleformers.transformers import PretrainedModel

from fastdeploy import ModelRegistry
from fastdeploy.model_executor.models.model_base import ModelForCasualLM


class MyPretrainedModel(PretrainedModel):
    @classmethod
    def arch_names(cls):
        return "MyModelForCasualLM"


class MyModelForCasualLM(ModelForCasualLM):

    def __init__(self, fd_config):
        """
        Args:
            fd_config : Configurations for the LLM model.
        """
        super().__init__(fd_config)
        print("init done")

    @classmethod
    def name(cls):
        return "MyModelForCasualLM"

    def compute_logits(self, logits):
        logits[:, 0] += 1.0
        return logits


def register():
    if "MyModelForCasualLM" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model_class(MyModelForCasualLM)
        ModelRegistry.register_pretrained_model(MyPretrainedModel)
