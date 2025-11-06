import unittest
from unittest.mock import patch

import numpy as np
import paddle

from fastdeploy.platforms.utils import convert_to_npu_dequant_scale


class TestConvertToNpuDequantScale(unittest.TestCase):

    def test_npu_not_available(self):
        with patch("paddle.is_compiled_with_custom_device", return_value=False):
            x = paddle.to_tensor([1.0, 2.0, 3.0], dtype=paddle.float32)
            out = convert_to_npu_dequant_scale(x)
            self.assertTrue((out.numpy() == x.numpy()).all())

    def test_npu_available(self):
        with patch("paddle.is_compiled_with_custom_device", return_value=True):
            x = paddle.to_tensor([1, 2, 3], dtype=paddle.float32)
            out = convert_to_npu_dequant_scale(x)
            self.assertEqual(out.dtype, paddle.int64)
            # Verify scaled output matches expected NPU dequantization format
            arr = x.numpy()
            new_deq_scale = np.stack([arr.reshape(-1, 1), np.zeros_like(arr).reshape(-1, 1)], axis=-1).reshape(-1)
            expected = np.frombuffer(new_deq_scale.tobytes(), dtype=np.int64)
            self.assertTrue((out.numpy() == expected).all())


if __name__ == "__main__":
    unittest.main()
