#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <cstring>  // for memset
#include "rknn_api.h"  // NOLINT
#include "fastdeploy/backends/backend.h"

typedef enum _rknpu2_cpu_name {
    RK356X = 0,  /* run on RK356X. */
    RK3588 = 1,  /* default,run on RK3588. */
    UNDEFINED,
} rknpu2_cpu_name;

typedef rknn_core_mask rknpu2_core_mask;

namespace fastdeploy {
struct RKNPU2BackendOption {
    rknpu2_cpu_name cpu_name = rknpu2_cpu_name::RK3588;
    // rknn_context context:设置运行核的 rknn_context 对象。
    // RKNN_NPU_CORE_AUTO:表示自动调度模型，自动运行在当前空闲的 NPU 核 上;
    // RKNN_NPU_CORE_0:表示运行在 NPU0 核上
    // RKNN_NPU_CORE_1:表示运行在 NPU1 核上
    // RKNN_NPU_CORE_2:表示运行在 NPU2 核上
    // RKNN_NPU_CORE_0_1:表示同时工作在 NPU0、NPU1 核上
    // RKNN_NPU_CORE_0_1_2:表示同时工作在 NPU0、NPU1、NPU2 核上
    rknpu2_core_mask core_mask = rknpu2_core_mask::RKNN_NPU_CORE_AUTO;
};

class RKNPU2Backend : public BaseBackend  {
 public:
    RKNPU2Backend() = default;

    virtual ~RKNPU2Backend() = default;

    // RKNN API
    bool LoadModel(void *model);

    bool GetSDKAndDeviceVersion();

    bool SetCoreMask(rknpu2_core_mask &core_mask);

    bool GetModelInputOutputInfos();

    // BaseBackend API
    void BuildOption(const RKNPU2BackendOption &option);

    bool InitFromRKNN(const std::string &model_file,
    const std::string &params_file,
    const RKNPU2BackendOption &option = RKNPU2BackendOption());


    int NumInputs() const override { return inputs_desc_.size(); }

    int NumOutputs() const override { return outputs_desc_.size(); }

    TensorInfo GetInputInfo(int index) override;
    TensorInfo GetOutputInfo(int index) override;
    std::vector<TensorInfo> GetInputInfos() override;
    std::vector<TensorInfo> GetOutputInfos() override;
    bool Infer(std::vector<FDTensor>& inputs,
               std::vector<FDTensor>* outputs) override;

    // change dype


 private:
    // ctx句柄，每读取一个新的模型，需要重新创建句柄
    rknn_context ctx{};
    // sdk_ver用于保存sdk和version版本号
    rknn_sdk_version sdk_ver{};
    // io_num用于保存输入输出数目
    rknn_input_output_num io_num{};
    std::vector<TensorInfo> inputs_desc_;
    std::vector<TensorInfo> outputs_desc_;

    rknn_tensor_attr *input_attrs;
    rknn_tensor_attr *output_attrs;

    RKNPU2BackendOption option_;

    void DumpTensorAttr(rknn_tensor_attr &attr);
};
}  // namespace fastdeploy
