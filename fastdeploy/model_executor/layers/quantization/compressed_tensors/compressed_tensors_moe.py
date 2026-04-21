from ...linear import QuantMethodBase
import paddle

class CompressedTensorsMoEMethod(QuantMethodBase):
    
    def __init__(self):
        pass
    
    def name(self) -> str:
        return "compressed-tensors"
    
    def create_weights(self, layer, **extra_weight_attrs):
        self.default_dtype = layer._helper.get_default_dtype()
        self.weight_dtype = "int32"
        
        self.w13_shape = [layer.num_local_experts, layer.moe_intermediate_size, layer.hidden_size // 8]
        self.s13_shape = [layer.num_local_experts, layer.moe_intermediate_size, layer.hidden_size // 32]
        
        self.w2_shape = [layer.num_local_experts, layer.hidden_size, layer.moe_intermediate_size // 8]
        self.s2_shape = [layer.num_local_experts, layer.hidden_size, layer.moe_intermediate_size // 32]

        setattr(layer, "gate_proj_weight_packed",  layer.create_parameter(shape=self.w13_shape, dtype=self.weight_dtype,    default_initializer=paddle.nn.initializer.Constant(0)))
        setattr(layer, "gate_proj_weight_scale",   layer.create_parameter(shape=self.s13_shape, dtype=self.default_dtype,   default_initializer=paddle.nn.initializer.Constant(0)))
        setattr(layer, "up_proj_weight_packed",    layer.create_parameter(shape=self.w13_shape, dtype=self.weight_dtype,    default_initializer=paddle.nn.initializer.Constant(0)))
        setattr(layer, "up_proj_weight_scale",     layer.create_parameter(shape=self.s13_shape, dtype=self.default_dtype,   default_initializer=paddle.nn.initializer.Constant(0)))
    
        setattr(layer, "down_proj_weight_packed",  layer.create_parameter(shape=self.w2_shape,  dtype=self.weight_dtype,    default_initializer=paddle.nn.initializer.Constant(0)))
        setattr(layer, "down_proj_weight_scale",   layer.create_parameter(shape=self.s2_shape,  dtype=self.default_dtype,   default_initializer=paddle.nn.initializer.Constant(0)))
       
    
    def apply():
        pass