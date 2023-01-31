# Numpy DLPack Array Conversion Example

This example demonstrates how a underlying array memory can be handed off between
two DLPack compatible frameworks without requiring any copies. In this case,
we demonstrate how to convert numpy to TVM's NDArray and vice-versa with proper
memory handling. We hope that not only is this directly useful for TVM users, but
also a solid example for how similar efficient copies can be implemented in other
array frameworks.

## File Breakdown

[dlpack.py](dlpack/dlpack.py): Contains the definition of common DLPack structures shared between frameworks. Mirrors the official C++ definitions.

[from_numpy.py](dlpack/from_numpy.py): Demonstrates how to convert a numpy array into a PyCapsule containing a DLPack Tensor.

[to_numpy.py](dlpack/to_numpy.py): Demonstrates how to take a PyCapsule with a DLPack Tensor and convert it into a numpy array.

[test.py](dlpack/test.py): Shows how to_numpy and from_numpy can be used to convert tensor formats without copies.

## Authors
[Josh Fromm](https://github.com/jwfromm)

[Junru Shao](https://github.com/junrushao1994)
