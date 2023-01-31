import tvm
import numpy as np
from dlpack import from_numpy, to_numpy


def test_from_numpy():
    """Test the copy free conversion of numpy to a tvm ndarray."""
    np_array = np.random.normal(size=[10, 10])
    np_array_ref = np_array.copy()
    tvm_array = tvm.nd.from_dlpack(from_numpy(np_array))
    del np_array
    np.testing.assert_equal(actual=tvm_array.numpy(), desired=np_array_ref)
    del tvm_array


def test_to_numpy():
    """Test the copy free conversion of a tvm ndarray to a numpy array"""
    tvm_array = tvm.nd.array(np.random.normal(size=[10, 10]))
    np_array_ref = tvm_array.numpy()
    np_array = to_numpy(tvm_array.__dlpack__())
    del tvm_array
    np.testing.assert_equal(actual=np_array, desired=np_array_ref)
    del np_array


if __name__ == "__main__":
    """
    Run both tests a bunch of times to make
    sure the conversions and memory management are stable.
    """
    print("### Testing from_numpy")
    for i in range(10000):
        test_from_numpy()
    print("### Testing to_numpy")
    for i in range(10000):
        test_to_numpy()
