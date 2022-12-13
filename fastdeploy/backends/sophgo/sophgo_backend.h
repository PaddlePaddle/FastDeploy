#include "fastdeploy/backends/backend.h"
#include "fastdeploy/core/fd_tensor.h"
//#include "rknn_api.h" // NOLINT
#include "bmruntime_interface.h"
#include <bmlib_runtime.h>
#include "fastdeploy/backends/sophgo/sophgo_config.h"
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>


namespace fastdeploy {
    struct SophgoBackendOption{
        // rknpu2::CpuName cpu_name = rknpu2::CpuName::RK3588;
        // rknpu2::CoreMask core_mask = rknpu2::CoreMask::RKNN_NPU_CORE_AUTO;
    };

    class SophgoBackend : public BaseBackend {
        public:
        SophgoBackend() = default;
        virtual ~SophgoBackend();
        bool LoadModel(void* model);
        bool GetSDKAndDeviceVersion();
        //bool SetCoreMask(rknpu2::CoreMask& core_mask) const;
        bool GetModelInputOutputInfos();
        void BuildOption(const SophgoBackendOption& option);
        bool InitFromSophgo(const std::string& model_file,
                    const SophgoBackendOption& option = SophgoBackendOption());
                    
        int NumInputs() const override {
            return static_cast<int>(inputs_desc_.size());
        }
        
        int NumOutputs() const override {
            return static_cast<int>(outputs_desc_.size());
        }

        TensorInfo GetInputInfo(int index) override;
        TensorInfo GetOutputInfo(int index) override;
        std::vector<TensorInfo> GetInputInfos() override;
        std::vector<TensorInfo> GetOutputInfos() override;
        bool Infer(std::vector<FDTensor>& inputs,
                    std::vector<FDTensor>* outputs,
                    bool copy_to_fd = true) override;

        private:

        // The object of rknn context.
        // rknn_context ctx{};
        // // The structure rknn_sdk_version is used to indicate the version
        // // information of the RKNN SDK.
        // rknn_sdk_version sdk_ver{};
        // // The structure rknn_input_output_num represents the number of
        // // input and output Tensor
        // rknn_input_output_num io_num{};

        std::vector<TensorInfo> inputs_desc_;
        std::vector<TensorInfo> outputs_desc_;

        // rknn_tensor_attr* input_attrs_ = nullptr;
        // rknn_tensor_attr* output_attrs_ = nullptr;

        // rknn_tensor_mem** input_mems_;
        // rknn_tensor_mem** output_mems_;

        bm_handle_t handle_;
        void * p_bmrt_ = nullptr;

        bool infer_init = false;

        const bm_net_info_t* net_info_ = nullptr;

        // RKNPU2BackendOption option_;

        // static void DumpTensorAttr(rknn_tensor_attr& attr);
        // static FDDataType RknnTensorTypeToFDDataType(rknn_tensor_type type);
        // static rknn_tensor_type FDDataTypeToRknnTensorType(FDDataType type);
    };
}