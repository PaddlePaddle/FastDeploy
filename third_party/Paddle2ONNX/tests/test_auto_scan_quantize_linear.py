from quantize_ops import quantize_linear
import os
import paddle
import numpy as np
from onnxbase import APIOnnx
from onnxbase import randtool
from auto_scan_test import OPConvertAutoScanTest, BaseNet
from onnxbase import randtool
from hypothesis import reproduce_failure
import hypothesis.strategies as st
import numpy as np
import unittest
import paddle


class Net(BaseNet):
    """
    simple Net
    """

    def __init__(self, config=None):
        super(Net, self).__init__(config)
        self.scale = paddle.to_tensor(
            randtool("float", -8, 8, self.config["quant_shape"]),
            dtype='float32')
        self.zero_points = paddle.to_tensor(
            np.zeros(self.config["quant_shape"]), dtype='float32')

    def forward(self, x):
        """
        forward
        """
        x = quantize_linear(
            x,
            self.scale,
            self.zero_points,
            bit_length=8,
            quant_axis=self.config["quant_axis"])
        return x.astype('float32')


class TestQuantizeLinearConvert(OPConvertAutoScanTest):
    """
    api: quantize_linear
    OPset version: 7, 9, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=3, max_value=10), min_size=2, max_size=4))

        if draw(st.booleans()):
            quant_axis = 0

        else:
            quant_axis = 1

        scale_shape = input_shape[quant_axis]
        zero_shape = input_shape[quant_axis]

        def generator_data():
            input_data = randtool("float", -10, 10, input_shape)
            floor_data = np.floor(input_data)
            diff = abs(input_data - floor_data)
            res = abs(diff - 0.5)
            input_data[res < 1e-5] = 1
            return input_data

        config = {
            "op_names": ["quantize_linear"],
            "test_data_shapes": [generator_data],
            "test_data_types": [['float32']],
            "opset_version": [13, 15],
            "input_spec_shape": [],
            "quant_axis": quant_axis,
            "quant_shape": scale_shape,
            "delta":
            1e1,  #TODO(yeliang) Can be remove after the quantize method of paddle updated
            "rtol": 1e1
        }

        models = Net(config)

        if not os.path.exists("calibration_table.txt"):
            with open("calibration_table.txt", 'w') as txt_file:
                txt_file.write("Fake_Quantize_Demo.")

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)


if __name__ == "__main__":
    unittest.main()
