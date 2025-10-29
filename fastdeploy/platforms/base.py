# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
"""
platform interface file
"""

import enum

import paddle


class _Backend(enum.Enum):
    NATIVE_ATTN = enum.auto()
    APPEND_ATTN = enum.auto()
    MLA_ATTN = enum.auto()
    FLASH_ATTN = enum.auto()
    BLOCK_ATTN = enum.auto()
    PLAS_ATTN = enum.auto()
    HPU_ATTN = enum.auto()


class Platform:
    """
    Platform base class, all device class will be derived from it
    """

    device_name: str

    def is_cuda(self) -> bool:
        """
        whether platform is cuda
        """
        return paddle.is_compiled_with_cuda() and not paddle.is_compiled_with_rocm()

    def is_npu(self) -> bool:
        """
        whether platform is npu
        """
        return paddle.is_compiled_with_custom_device("npu")

    def is_xpu(self) -> bool:
        """
        whether platform is xpu
        """
        return paddle.is_compiled_with_xpu()

    def is_intel_hpu(self) -> bool:
        """
        whether platform is intel_hpu
        """
        return paddle.is_compiled_with_custom_device("intel_hpu")

    def is_cpu(self) -> bool:
        """
        whether platform is cpu
        """
        return paddle.device.get_device().lower() == "cpu"

    def is_dcu(self) -> bool:
        """
        whether platform is dcu
        """
        return paddle.is_compiled_with_rocm()

    def is_iluvatar(self) -> bool:
        """
        whether platform is iluvatar gpu
        """
        return paddle.is_compiled_with_custom_device("iluvatar_gpu")

    def is_gcu(self) -> bool:
        """
        whether platform is gcu
        """
        return paddle.is_compiled_with_custom_device("gcu")

    def is_maca(self) -> bool:
        """
        whether platform is metax gpu
        """
        return paddle.is_compiled_with_custom_device("metax_gpu")

    @classmethod
    def get_attention_backend_cls(self, selected_backend):
        """Get the attention backend"""
        return ""

    @classmethod
    def verify_quant(self, quant):
        """
        Verify whether the quantization is supported by the current platform.
        """
        if self.supported_quantization and quant not in self.supported_quantization:
            raise ValueError(f"{quant} quantization is currently not supported in " f"{self.device_name}.")

    @classmethod
    def available(self):
        """Return whether the device is available"""
        return True

    @classmethod
    def supports_fp8(self) -> bool:
        """
        Returns whether the current platform supports FP8 types.
        """
        return False
