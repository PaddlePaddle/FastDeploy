# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

from fastdeploy.model_executor.utils import (
    BitMaskTracker,
    TensorTracker,
    WeightsMapper,
    remap_weight_keys,
    set_weight_attrs,
    slice_fn,
)


class TestBitMaskTracker(unittest.TestCase):
    def test_empty_is_not_full(self):
        t = BitMaskTracker(8)
        self.assertFalse(t.is_full())

    def test_mark_all(self):
        t = BitMaskTracker(4)
        t.mark(0, 4)
        self.assertTrue(t.is_full())

    def test_mark_in_parts(self):
        t = BitMaskTracker(8)
        t.mark(0, 4)
        self.assertFalse(t.is_full())
        t.mark(4, 8)
        self.assertTrue(t.is_full())

    def test_overlapping_marks(self):
        t = BitMaskTracker(4)
        t.mark(0, 3)
        t.mark(2, 4)
        self.assertTrue(t.is_full())

    def test_single_element(self):
        t = BitMaskTracker(1)
        self.assertFalse(t.is_full())
        t.mark(0, 1)
        self.assertTrue(t.is_full())

    def test_invalid_range_raises(self):
        t = BitMaskTracker(4)
        with self.assertRaises(ValueError):
            t.mark(-1, 2)
        with self.assertRaises(ValueError):
            t.mark(0, 5)
        with self.assertRaises(ValueError):
            t.mark(3, 2)


class TestTensorTracker2D(unittest.TestCase):
    def test_track_columns(self):
        tt = TensorTracker((4, 8), output_dim=1)
        tt.mark(start=0, end=4)
        self.assertFalse(tt.is_fully_copied())
        tt.mark(start=4, end=8)
        self.assertTrue(tt.is_fully_copied())

    def test_track_rows(self):
        tt = TensorTracker((4, 8), output_dim=0)
        tt.mark(start=0, end=4)
        self.assertTrue(tt.is_fully_copied())

    def test_partial_fill(self):
        tt = TensorTracker((4, 8), output_dim=1)
        tt.mark(start=0, end=3)
        self.assertFalse(tt.is_fully_copied())


class TestTensorTracker3D(unittest.TestCase):
    def test_track_all_batches(self):
        tt = TensorTracker((2, 4, 8), output_dim=1)
        # Must fill both batches
        tt.mark(start=0, end=8, batch_id=0)
        self.assertFalse(tt.is_fully_copied())
        tt.mark(start=0, end=8, batch_id=1)
        self.assertTrue(tt.is_fully_copied())

    def test_missing_batch_id_raises(self):
        tt = TensorTracker((2, 4, 8), output_dim=1)
        with self.assertRaises(ValueError):
            tt.mark(start=0, end=8)


class TestTensorTrackerInvalidDim(unittest.TestCase):
    def test_1d_raises(self):
        with self.assertRaises(ValueError):
            TensorTracker((8,), output_dim=0)


class TestWeightsMapper(unittest.TestCase):
    def test_prefix_mapping(self):
        mapper = WeightsMapper(orig_to_new_prefix={"old.": "new."})
        self.assertEqual(mapper.apply("old.layer1.weight"), "new.layer1.weight")

    def test_no_match(self):
        mapper = WeightsMapper(orig_to_new_prefix={"old.": "new."})
        self.assertEqual(mapper.apply("other.layer1.weight"), "other.layer1.weight")

    def test_multiple_prefixes(self):
        mapper = WeightsMapper(orig_to_new_prefix={"a.": "x.", "b.": "y."})
        self.assertEqual(mapper.apply("a.foo"), "x.foo")
        self.assertEqual(mapper.apply("b.bar"), "y.bar")


class TestRemapWeightKeys(unittest.TestCase):
    def test_basic_remap(self):
        weights = [("model.layer.weight", 1), ("model.layer.bias", 2)]
        mapper = {"model.": "new_model."}
        result = list(remap_weight_keys(iter(weights), mapper))
        self.assertEqual(result[0][0], "new_model.layer.weight")
        self.assertEqual(result[1][0], "new_model.layer.bias")

    def test_include_keys_filter(self):
        weights = [("model.a.weight", 1), ("model.b.weight", 2), ("model.c.bias", 3)]
        mapper = {}
        result = list(remap_weight_keys(iter(weights), mapper, include_keys=["weight"]))
        self.assertEqual(len(result), 2)

    def test_no_match_passthrough(self):
        weights = [("layer.weight", 1)]
        mapper = {"other.": "new."}
        result = list(remap_weight_keys(iter(weights), mapper))
        self.assertEqual(result[0][0], "layer.weight")


class TestSetWeightAttrs(unittest.TestCase):
    def test_sets_attrs(self):
        class Param:
            pass

        p = Param()
        set_weight_attrs(p, {"output_dim": 1, "tp_row_bias": True})
        self.assertEqual(p.output_dim, 1)
        self.assertTrue(p.tp_row_bias)

    def test_none_map_noop(self):
        class Param:
            pass

        p = Param()
        set_weight_attrs(p, None)  # should not raise


class TestSliceFn(unittest.TestCase):
    def test_1d_slice(self):
        import numpy as np

        w = np.arange(10)
        result = slice_fn(w, output_dim=False, start=2, end=5)
        self.assertEqual(list(result), [2, 3, 4])

    def test_2d_output_dim_true(self):
        import numpy as np

        w = np.ones((4, 8))
        result = slice_fn(w, output_dim=True, start=0, end=4)
        self.assertEqual(result.shape, (4, 4))

    def test_2d_output_dim_false(self):
        import numpy as np

        w = np.ones((4, 8))
        result = slice_fn(w, output_dim=False, start=1, end=3)
        self.assertEqual(result.shape, (2, 8))


if __name__ == "__main__":
    unittest.main()
