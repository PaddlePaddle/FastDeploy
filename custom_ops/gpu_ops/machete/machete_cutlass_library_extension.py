# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum
from typing import Union

from cutlass_library import (
    DataType,
    DataTypeNames,
    DataTypeSize,
    DataTypeTag,
    KernelScheduleTag,
    KernelScheduleType,
    enum_auto,
)

#
#   Extend cutlass library with custom types, and missing values
#


class MACHETEDataType(enum.Enum):
    u4b8 = enum_auto()
    u8b128 = enum_auto()


class MixedInputKernelScheduleType(enum.Enum):
    TmaWarpSpecialized = enum_auto()
    TmaWarpSpecializedPingpong = enum_auto()
    TmaWarpSpecializedCooperative = enum_auto()


MACHETEDataTypeNames: dict[Union[MACHETEDataType, DataType], str] = {
    **DataTypeNames,  # type: ignore
    **{
        MACHETEDataType.u4b8: "u4b8",
        MACHETEDataType.u8b128: "u8b128",
    },
}

MACHETEDataTypeTag: dict[Union[MACHETEDataType, DataType], str] = {
    **DataTypeTag,  # type: ignore
    **{
        MACHETEDataType.u4b8: "cutlass::machete_uint4b8_t",
        MACHETEDataType.u8b128: "cutlass::machete_uint8b128_t",
    },
}

MACHETEDataTypeSize: dict[Union[MACHETEDataType, DataType], int] = {
    **DataTypeSize,  # type: ignore
    **{
        MACHETEDataType.u4b8: 4,
        MACHETEDataType.u8b128: 8,
    },
}

MACHETEDataTypeMACHETEScalarTypeTag: dict[Union[MACHETEDataType, DataType], str] = {
    MACHETEDataType.u4b8: "machete::kU4B8",
    MACHETEDataType.u8b128: "machete::kU8B128",
    DataType.u4: "machete::kU4",
    DataType.u8: "machete::kU8",
    DataType.s4: "machete::kS4",
    DataType.s8: "machete::kS8",
    DataType.f16: "machete::kFloat16",
    DataType.bf16: "machete::kBfloat16",
}

MACHETEDataTypePaddleDataTypeTag: dict[Union[MACHETEDataType, DataType], str] = {
    DataType.u8: "paddle::DataType::UINT8",
    DataType.s8: "paddle::DataType::INT8",
    DataType.e4m3: "paddle::DataType::FLOAT8_E4M3FN",
    DataType.s32: "paddle::DataType::INT32",
    DataType.f16: "paddle::DataType::FLOAT16",
    DataType.bf16: "paddle::DataType::BFLOAT16",
    DataType.f32: "paddle::DataType::FLOAT32",
}

MACHETEKernelScheduleTag: dict[Union[MixedInputKernelScheduleType, KernelScheduleType], str] = {
    **KernelScheduleTag,  # type: ignore
    **{
        MixedInputKernelScheduleType.TmaWarpSpecialized: "cutlass::gemm::KernelTmaWarpSpecialized",
        MixedInputKernelScheduleType.TmaWarpSpecializedPingpong: "cutlass::gemm::KernelTmaWarpSpecializedPingpong",
        MixedInputKernelScheduleType.TmaWarpSpecializedCooperative: "cutlass::gemm::KernelTmaWarpSpecializedCooperative",
    },
}
