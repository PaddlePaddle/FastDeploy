DLPack Change Log
=================

This file records the changes in DLPack in reverse chronological order.

## v0.7

- Add kDLHexagon
- Add kDLOneAPI
- Add DLPACK_VERSION and DLPACK_ABI_VERSION


## v0.6

- Add kDLROCMHost
- Add kDLCUDAManaged


## v0.5

Rename enum values
- kDLGPU -> kDLCUDA
- kDLCPUPinned -> kDLCUDAHost
The ABI is backward compatible, as it is only change of constant name,
exchange can still happen between the new version and old version.


## v0.4

- OpaqueHandle type
- Complex support
- Rename DLContext -> DLDevice
  - This requires dependent frameworks to upgrade the type name.
  - The ABI is backward compatible, as it is only change of constant name.

## v0.3

- Add bfloat16
- Vulkan support


## v0.2
- New device types
  - kDLMetal for Apple Metal device
  - kDLVPI for verilog simulator memory
  - kDLROCM for AMD GPUs
- Add prefix DL to all enum constant values
  - This requires dependent frameworks to upgrade their reference to these constant
  - The ABI is compatible, as it is only change of constant name.
- Add DLManagedTensor structure for borrowing tensors

## v0.1
- Finalize DLTensor structure
