"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
"""

import numpy as np

from fastdeploy.entrypoints.openai.usage_calculator import count_tokens


class TestCountTokens:
    """Test cases for count_tokens function"""

    def test_empty_list(self):
        """Test counting tokens in an empty list"""
        tokens = []
        result = count_tokens(tokens)
        assert result == 0

    def test_flat_list_of_integers(self):
        """Test counting tokens in a flat list of integers"""
        tokens = [1, 2, 3, 4, 5]
        result = count_tokens(tokens)
        assert result == 5

    def test_flat_list_of_strings(self):
        """Test counting tokens in a flat list of strings"""
        tokens = ["hello", "world", "test"]
        result = count_tokens(tokens)
        assert result == 3

    def test_flat_numpy_array(self):
        """Test counting tokens in a flat numpy array"""
        tokens = np.array([1, 2, 3, 4, 5])
        result = count_tokens(tokens)
        assert result == 5

    def test_nested_list_one_level(self):
        """Test counting tokens in a nested list with one level of nesting"""
        tokens = [[1, 2], [3, 4], [5]]
        result = count_tokens(tokens)
        assert result == 5

    def test_nested_list_multiple_levels(self):
        """Test counting tokens in a deeply nested list"""
        tokens = [[1, [2, 3]], [4, [5, [6]]], 7]
        result = count_tokens(tokens)
        assert result == 7

    def test_nested_tuple(self):
        """Test counting tokens in nested tuples"""
        tokens = ((1, 2), (3, (4, 5)), 6)
        result = count_tokens(tokens)
        assert result == 6

    def test_mixed_nested_structures(self):
        """Test counting tokens in mixed nested structures (list, tuple, numpy array)"""
        tokens = [1, (2, 3), np.array([4, 5]), [6, [7, 8]]]
        result = count_tokens(tokens)
        assert result == 8

    def test_single_element_list(self):
        """Test counting tokens in a list with single element"""
        tokens = [42]
        result = count_tokens(tokens)
        assert result == 1

    def test_single_element_tuple(self):
        """Test counting tokens in a tuple with single element"""
        tokens = (42,)
        result = count_tokens(tokens)
        assert result == 1

    def test_single_element_numpy_array(self):
        """Test counting tokens in a numpy array with single element"""
        tokens = np.array([42])
        result = count_tokens(tokens)
        assert result == 1

    def test_nested_empty_lists(self):
        """Test counting tokens in nested empty lists"""
        tokens = [[], [[]], [[[]]]]
        result = count_tokens(tokens)
        assert result == 0

    def test_complex_mixed_structure(self):
        """Test counting tokens in a complex mixed structure"""
        tokens = [
            1,
            [2, 3, (4, np.array([5, 6]))],
            [7, [8, 9, [10]]],
            (11, [12, 13]),
            np.array([14, 15]),  # Note: numpy arrays can't contain lists directly
        ]
        # Flatten the structure manually for expected count
        result = count_tokens(tokens)
        assert result == 15

    def test_large_flat_list(self):
        """Test counting tokens in a large flat list"""
        tokens = list(range(1000))
        result = count_tokens(tokens)
        assert result == 1000

    def test_none_values(self):
        """Test counting tokens when list contains None values"""
        tokens = [1, None, 2, [None, 3], None]
        result = count_tokens(tokens)
        assert result == 6

    def test_boolean_values(self):
        """Test counting tokens with boolean values"""
        tokens = [True, False, [True, False]]
        result = count_tokens(tokens)
        assert result == 4

    def test_float_values(self):
        """Test counting tokens with float values"""
        tokens = [1.5, 2.7, [3.14, 4.2]]
        result = count_tokens(tokens)
        assert result == 4

    def test_mixed_data_types(self):
        """Test counting tokens with mixed data types"""
        tokens = [1, "hello", 2.5, True, None, [1, "world"]]
        result = count_tokens(tokens)
        assert result == 7

    def test_deeply_nested_structure(self):
        """Test counting tokens in a very deeply nested structure"""
        tokens = 1
        for _ in range(100):
            tokens = [tokens]
        result = count_tokens(tokens)
        assert result == 1

    def test_numpy_array_2d(self):
        """Test counting tokens in a 2D numpy array"""
        tokens = np.array([[1, 2], [3, 4], [5, 6]])
        result = count_tokens(tokens)
        assert result == 6

    def test_numpy_array_3d(self):
        """Test counting tokens in a 3D numpy array"""
        tokens = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        result = count_tokens(tokens)
        assert result == 8
