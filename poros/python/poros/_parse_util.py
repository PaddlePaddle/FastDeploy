"""
 parse util for some python settings to poros c++ setting.
"""
import typing #  List & Dict & Any
import poros._C

def _parse_device(device):
    # type: Any -> poros._C.Device
    """
    converter device info to Device struct that can handle in poros
    """
    if isinstance(device, poros._C.Device):
        return device
    elif isinstance(device, str):
        if device == "GPU" or device == "gpu":
            return poros._C.Device.GPU
        elif device == "CPU" or device == "cpu":
            return poros._C.Device.CPU
        elif device == "XPU" or device == "xpu":
            return poros._C.Device.XPU
        else:
            ValueError("Got a device type unknown (type: " + str(device) + ")")
    else:
        raise TypeError("Device specification must be of type string or poros.Device, but got: " +
                str(type(device)))


def _parse_dynamic_shape(dynamic_shape):
    # type: Dict(str, List) -> poros._C.DynamicShapeOptions
    """
    converter key-value dynamic_shape to DynamicShapeOptions that can handle in poros
    """
    
    shape_option = poros._C.DynamicShapeOptions()
    if "is_dynamic" in dynamic_shape:
        assert isinstance(dynamic_shape["is_dynamic"], bool)
        shape_option.is_dynamic = dynamic_shape["is_dynamic"]
    
    if "max_shapes" in dynamic_shape:
        assert isinstance(dynamic_shape["max_shapes"], list)
        shape_option.max_shapes = dynamic_shape["max_shapes"]

    if "min_shapes" in dynamic_shape:
        assert isinstance(dynamic_shape["min_shapes"], list)
        shape_option.min_shapes = dynamic_shape["min_shapes"]

    if "opt_shapes" in dynamic_shape:
        assert isinstance(dynamic_shape["opt_shapes"], list)
        shape_option.opt_shapes = dynamic_shape["opt_shapes"]

    return shape_option