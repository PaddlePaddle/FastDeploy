import numpy as np
from dlpack import from_numpy, to_numpy


def test_to_from_numpy_zero_copy():
    """Test the copy free conversion of numpy array via DLPack."""
    np_ary = np.random.normal(size=[10, 10])
    np_ary_big = np.random.normal(size=[12, 10])
    dlpack_capsule = from_numpy(np_ary_big)
    reconstructed_ary = to_numpy(dlpack_capsule)
    del dlpack_capsule
    np_ary_big[1:11] = np_ary
    del np_ary_big
    np.testing.assert_equal(actual=reconstructed_ary[1:11], desired=np_ary)


def test_to_from_numpy_memory():
    """Test that DLPack capsule keeps the source array alive"""
    source_array = np.random.normal(size=[10, 10])
    np_array_ref = source_array.copy()
    dlpack_capsule = from_numpy(source_array)
    del source_array
    reconstructed_array = to_numpy(dlpack_capsule)
    del dlpack_capsule
    np.testing.assert_equal(actual=reconstructed_array, desired=np_array_ref)


if __name__ == "__main__":
    """
    Run both tests a bunch of times to make
    sure the conversions and memory management are stable.
    """
    print("### Running `test_to_from_numpy_zero_copy`")
    for i in range(10000):
        test_to_from_numpy_zero_copy()
    print("### Running `test_to_from_numpy_memory`")
    for i in range(10000):
        test_to_from_numpy_memory()
