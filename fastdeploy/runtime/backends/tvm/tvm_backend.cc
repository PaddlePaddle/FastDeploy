#include "fastdeploy/runtime/backends/tvm/tvm_backend.h"
#include "yaml-cpp/yaml.h"

namespace fastdeploy {
bool TVMBackend::Init(const fastdeploy::RuntimeOption& runtime_option) {
  if (!(Supported(runtime_option.model_format, Backend::TVM) &&
        Supported(runtime_option.device, Backend::TVM))) {
    FDERROR << "TVMBackend doesn't support this model format or device."
            << std::endl;
    return false;
  }

  if (runtime_option.model_from_memory_) {
    FDERROR << "TVMBackend doesn't support load model from memory, please "
               "load model from disk."
            << std::endl;
    return false;
  }

  if (!BuildDLDevice(runtime_option.device)) {
    FDERROR << "TVMBackend only don't support run in this device." << std::endl;
    return false;
  }

  if (!BuildModel(runtime_option)) {
    FDERROR << "TVMBackend only don't support run with this model path."
            << std::endl;
    return false;
  }
  return true;
}

bool TVMBackend::BuildModel(const RuntimeOption& runtime_option) {
  std::cout << "BuildModel Start" << std::endl;
  // load in the library
  DLDevice dev{kDLCPU, 0};
  tvm::runtime::Module mod_factory = tvm::runtime::Module::LoadFromFile(runtime_option.model_file.data());
  // create the graph executor module
  tvm::runtime::Module gmod = mod_factory.GetFunction("default")(dev);
  tvm::runtime::PackedFunc set_input = gmod.GetFunction("set_input");
  tvm::runtime::PackedFunc get_output = gmod.GetFunction("get_output");
  tvm::runtime::PackedFunc run = gmod.GetFunction("run");
//  bundle_ = dlopen(runtime_option.model_file.data(), RTLD_LAZY | RTLD_LOCAL);
//  assert(bundle_);
//  char* json_data = nullptr;
//  int error =
//      ReadAll("mod.json", runtime_option.tvm_option.model_json_file.data(),
//              &json_data, nullptr);
//  if (error != 0) {
//    FDERROR << "Read json failed." << std::endl;
//    return false;
//  }
//
//  char* params_data = nullptr;
//  size_t params_size;
//  error =
//      ReadAll("mod.params", runtime_option.tvm_option.model_params_file.data(),
//              &params_data, &params_size);
//  if (error != 0) {
//    FDERROR << "Read params failed." << std::endl;
//    return false;
//  }

  std::cout << "BuildModel End" << std::endl;
  return true;
}

bool TVMBackend::Infer(std::vector<FDTensor>& inputs,
                       std::vector<FDTensor>* outputs, bool copy_to_fd) {
  return false;
}

bool TVMBackend::BuildDLDevice(fastdeploy::Device device) {
  if (device == Device::CPU) {
    dev_ = DLDevice{kDLCPU, 0};
  } else {
    FDERROR << "TVMBackend only support run in CPU." << std::endl;
    return false;
  }
  return true;
}
}  // namespace fastdeploy