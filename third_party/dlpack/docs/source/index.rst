Welcome to DLPack's documentation!
==================================


Purpose
~~~~~~~

In order for an ndarray system to interact with a variety of frameworks, a
stable in-memory data structure is needed.

DLPack is one such data structure that allows exchange between major
frameworks. It is developed with inputs from many deep learning system core
developers. Highlights include:

* Minimum and stable: :ref:`simple header <c_api>`
* Designed for cross hardware: CPU, CUDA, OpenCL, Vulkan, Metal, VPI, ROCm,
  WebGPU, Hexagon
* Already a standard with wide community adoption and support:

  * `NumPy <https://numpy.org/doc/stable/release/1.22.0-notes.html#add-nep-47-compatible-dlpack-support>`_
  * `CuPy <https://docs.cupy.dev/en/stable/reference/generated/cupy.fromDlpack.html>`_
  * `PyTorch <https://pytorch.org/docs/stable/dlpack.html>`_
  * `Tensorflow <https://www.tensorflow.org/api_docs/python/tf/experimental/dlpack/from_dlpack>`_
  * `MXNet <https://mxnet.apache.org/versions/master/api/python/docs/_modules/mxnet/dlpack.html>`_
  * `TVM <https://tvm.apache.org/docs/reference/api/python/contrib.html#module-tvm.contrib.dlpack>`_
  * `mpi4py <https://mpi4py.readthedocs.io/en/stable/overview.html#support-for-gpu-aware-mpi>`_

* Clean C ABI compatible.

  * Means you can create and access it from any language.
  * It is also essential for building JIT and AOT compilers to support these
    data types.


Scope
~~~~~

The main design rationale of DLPack is the minimalism. DLPack drops the
consideration of allocator, device API and focus on the minimum data
structure. While still considering the need for cross hardware support
(e.g. the data field is opaque for platforms that does not support normal
addressing).

It also simplifies some of the design to remove legacy issues (e.g. everything
assumes to be row major, strides can be used to support other case, and avoid
the complexity to consider more layouts).


Roadmap
~~~~~~~

* C API that could be exposed as a new Python attribute ``__dlpack_info__``
  for returning API and ABI versions. (see `#34 <https://github.com/dmlc/dlpack/issues/34>`_,
  `#72 <https://github.com/dmlc/dlpack/pull/72>`_)
* Clarify alignment requirements. (see
  `data-apis/array-api#293 <https://github.com/data-apis/array-api/issues/293>`_,
  `numpy/numpy#20338 <https://github.com/numpy/numpy/issues/20338>`_,
  `data-apis/array-api#293 (comment) <https://github.com/data-apis/array-api/issues/293#issuecomment-964434449>`_)
* Adding support for boolean data type (see `#75 <https://github.com/dmlc/dlpack/issues/75>`_)
* Adding a read-only flag (ABI break) or making it a hard requirement in the spec that
  imported arrays should be treated as read-only. (see
  `data-apis/consortium-feedback#1 (comment) <https://github.com/data-apis/consortium-feedback/issues/1#issuecomment-675857753>`_,
  `data-apis/array-api#191 <https://github.com/data-apis/array-api/issues/191>`_)
* Standardize C interface for stream exchange. (see `#74 <https://github.com/dmlc/dlpack/issues/74>`_,
  `#65 <https://github.com/dmlc/dlpack/issues/65>`_)


DLPack Documentation
~~~~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 2

   c_api
   python_spec


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
