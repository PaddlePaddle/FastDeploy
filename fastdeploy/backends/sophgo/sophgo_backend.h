#include "fastdeploy/backends/backend.h"
#include "fastdeploy/core/fd_tensor.h"
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
        
    };

    class SophgoBackend : public BaseBackend {
        public:
        SophgoBackend() = default;
        virtual ~SophgoBackend();
        bool LoadModel(void* model);
        bool GetSDKAndDeviceVersion();
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

        std::vector<TensorInfo> inputs_desc_;
        std::vector<TensorInfo> outputs_desc_;
        std::string net_name_;

        bm_handle_t handle_;
        void * p_bmrt_ = nullptr;

        bool infer_init = false;

        const bm_net_info_t* net_info_ = nullptr;

        // SophgoTPU2BackendOption option_;

        static FDDataType SophgoTensorTypeToFDDataType(bm_data_type_t type);
        static bm_data_type_t FDDataTypeToSophgoTensorType(FDDataType type);
    };
}