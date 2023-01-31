import ctypes
import numpy as np
from .dlpack import _c_str_dltensor, DLManagedTensor, DLTensor

ctypes.pythonapi.PyCapsule_IsValid.restype = ctypes.c_int
ctypes.pythonapi.PyCapsule_IsValid.argtypes = [
    ctypes.py_object, ctypes.c_char_p
]

ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [
    ctypes.py_object, ctypes.c_char_p
]


def _array_interface_from_dl_tensor(dlt):
    """Constructs NumPy's array_interface dictionary
    from `dlpack.DLTensor` descriptor."""
    assert isinstance(dlt, DLTensor)
    shape = tuple(dlt.shape[dim] for dim in range(dlt.ndim))
    itemsize = dlt.dtype.lanes * dlt.dtype.bits // 8
    if dlt.strides:
        strides = tuple(dlt.strides[dim] * itemsize for dim in range(dlt.ndim))
    else:
        # Array is compact, make it numpy compatible.
        strides = []
        for i, s in enumerate(shape):
            cumulative = 1
            for e in range(i + 1, dlt.ndim):
                cumulative *= shape[e]
            strides.append(cumulative * itemsize)
        strides = tuple(strides)
    typestr = "|" + str(dlt.dtype.type_code)[0] + str(itemsize)
    return dict(
        version=3,
        shape=shape,
        strides=strides,
        data=(dlt.data, True),
        offset=dlt.byte_offset,
        typestr=typestr, )


class _Holder:
    """A wrapper that combines a pycapsule and array_interface for consumption by  numpy.

    Parameters
    ----------
    array_interface : dict
        A description of the underlying memory.

    pycapsule : PyCapsule
        A wrapper around the dlpack tensor that will be converted to numpy.
    """

    def __init__(self, array_interface, pycapsule) -> None:
        self.__array_interface__ = array_interface
        self._pycapsule = pycapsule


def to_numpy(pycapsule) -> np.ndarray:
    """Convert a dlpack tensor into a numpy array without copying.

    Parameters
    ----------
    pycapsule : PyCapsule
        A pycapsule wrapping a dlpack tensor that will be converted.

    Returns
    -------
    np_array : np.ndarray
        A new numpy array that uses the same underlying memory as the input
        pycapsule.
    """
    assert ctypes.pythonapi.PyCapsule_IsValid(pycapsule, _c_str_dltensor)
    dl_managed_tensor = ctypes.pythonapi.PyCapsule_GetPointer(pycapsule,
                                                              _c_str_dltensor)
    dl_managed_tensor_ptr = ctypes.cast(dl_managed_tensor,
                                        ctypes.POINTER(DLManagedTensor))
    dl_managed_tensor = dl_managed_tensor_ptr.contents
    holder = _Holder(
        _array_interface_from_dl_tensor(dl_managed_tensor.dl_tensor),
        pycapsule)
    return np.ctypeslib.as_array(holder)
