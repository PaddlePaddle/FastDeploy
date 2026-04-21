from ..quant_base import QuantConfigBase
from ...linear import LinearBase, UnquantizedLinearMethod
from ...moe import FusedMoE
from .compressed_tensors_moe import CompreesedTensorsMoEMethod

class CompressedTensorsConfig(QuantConfigBase):
    def __init__(self):
        self.quant_max_bound = 0
        self.quant_min_bound = 0
        self.quant_round_type = 0
    
    def name(self) -> str:
        return "compressed-tenosrs"
    
    @classmethod
    def from_config(cls, config:dict) -> "CompressedTensorsConfig":
        return cls()
    
    def get_quant_method(self, layer):
        if isinstance(layer, LinearBase):
            return UnquantizedLinearMethod()
        elif isinstance(layer, FusedMoE):
            return CompreesedTensorsMoEMethod()
        else:
            return None