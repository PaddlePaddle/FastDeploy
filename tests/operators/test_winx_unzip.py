import unittest

import numpy as np
import paddle

from fastdeploy.model_executor.ops.gpu import winx_unzip


class TestWinxUnzip(unittest.TestCase):
    def setUp(self):
        paddle.set_device("gpu")
        np.random.seed(42)
        self.batch = 1
        self.num_rows = 128
        self.num_columns = 128

    def test_wint25_unzip(self):
        zipped_rows = self.num_rows * 10 // 64
        zipped_weight = np.random.randint(
            -32768, 32767, size=(self.batch, zipped_rows, self.num_columns), dtype=np.int32
        ).astype(np.int16)

        super_scale = np.random.rand(self.batch, 1, self.num_columns).astype("float16")

        zipped_weight = paddle.to_tensor(zipped_weight, dtype=paddle.int16)
        super_scale = paddle.to_tensor(super_scale, dtype=paddle.float16)

        out = winx_unzip(
            zipped_weight,
            local_scale=None,
            code_scale=None,
            code_zp=None,
            super_scale=super_scale,
            quant_method="weight_only_int2.5",
        )[0]

        # Check output shape and non-trivial values
        expected_shape = [self.batch, self.num_rows, self.num_columns]
        if list(out.shape) == expected_shape[1:]:
            out = out.unsqueeze(0)

        self.assertEqual(list(out.shape), expected_shape)
        self.assertTrue(np.any(out.numpy() != 0))

    def test_wint2_unzip(self):
        zipped_rows = self.num_rows // 4
        zipped_weight = np.random.randint(0, 256, size=(self.batch, zipped_rows, self.num_columns), dtype=np.uint8)
        zipped_weight = paddle.to_tensor(zipped_weight, dtype=paddle.uint8)

        local_scale = np.random.randint(
            0, 256, size=(self.batch, self.num_rows // 128, self.num_columns), dtype=np.uint8
        )
        local_scale = paddle.to_tensor(local_scale, dtype=paddle.uint8)

        code_scale = np.random.rand(self.batch, 1, self.num_columns).astype("float32")
        code_zp = np.random.rand(self.batch, 1, self.num_columns).astype("float32")

        code_scale = paddle.to_tensor(code_scale, dtype=paddle.float32)
        code_zp = paddle.to_tensor(code_zp, dtype=paddle.float32)

        dummy_super_scale = paddle.ones([self.batch, 1, self.num_columns], dtype="float16")

        out = winx_unzip(
            zipped_weight,
            local_scale=local_scale,
            code_scale=code_scale,
            code_zp=code_zp,
            super_scale=dummy_super_scale,
            quant_method="weight_only_int2",
        )[0]

        out = paddle.cast(out, dtype="float16")

        # Check output shape and non-trivial values
        expected_shape = [self.batch, self.num_rows, self.num_columns]
        if list(out.shape) == expected_shape[1:]:
            out = out.unsqueeze(0)

        self.assertEqual(list(out.shape), expected_shape)
        self.assertTrue(np.any(out.numpy() != 0))


if __name__ == "__main__":
    unittest.main()
