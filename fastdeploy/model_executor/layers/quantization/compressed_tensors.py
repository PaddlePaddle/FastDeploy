from .quant_base import QuantConfigBase

class CompressedTensorsConfig(QuantConfigBase):
    def __init__(self):
        pass
    
    def name():
        pass
    
    @classmethod
    def from_config(cls, config:dict) -> "CompressedTensorsConfig":
        return cls()
    
    def get_quant_method():
        pass