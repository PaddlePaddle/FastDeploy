from ..quant_base import QuantConfigBase
from ...linear import LinearBase, UnquantizedLinearMethod
from ...moe import FusedMoE
from .compressed_tensors_moe import CompressedTensorsMoEMethod

class CompressedTensorsConfig(QuantConfigBase):
    def __init__(self, is_checkpoint_bf16: bool = False):
        self.quant_max_bound = 0
        self.quant_min_bound = 0
        self.quant_round_type = 0
        self.is_checkpoint_bf16 = is_checkpoint_bf16
    
    def name(self) -> str:
        return "compressed-tenosrs"
    
    @classmethod
    def from_config(cls, config:dict) -> "CompressedTensorsConfig":
        return cls()
    
    def get_quant_method(self, layer):
        if isinstance(layer, LinearBase):
            return UnquantizedLinearMethod()
        elif isinstance(layer, FusedMoE):
            return CompressedTensorsMoEMethod()
        else:
            return None