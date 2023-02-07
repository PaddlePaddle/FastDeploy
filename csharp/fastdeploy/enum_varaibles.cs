namespace fastdeploy{



public enum ModelFormat{
  AUTOREC,      ///< Auto recognize the model format by model file name
  PADDLE,       ///< Model with paddlepaddle format
  ONNX,         ///< Model with ONNX format
  RKNN,         ///< Model with RKNN format
  TORCHSCRIPT,  ///< Model with TorchScript format
  SOPHGO,       ///< Model with SOPHGO format
  }

public enum rknpu2_CpuName{
  RK356X = 0, /* run on RK356X. */
  RK3588 = 1, /* default,run on RK3588. */
  UNDEFINED,
  }

public enum rknpu2_CoreMask{
  RKNN_NPU_CORE_AUTO = 0,  //< default, run on NPU core randomly.
  RKNN_NPU_CORE_0 = 1,    //< run on NPU core 0.
  RKNN_NPU_CORE_1 = 2,    //< run on NPU core 1.
  RKNN_NPU_CORE_2 = 4,    //< run on NPU core 2.
  RKNN_NPU_CORE_0_1 =
      RKNN_NPU_CORE_0 | RKNN_NPU_CORE_1,  //< run on NPU core 1 and core 2.
  RKNN_NPU_CORE_0_1_2 =
      RKNN_NPU_CORE_0_1 | RKNN_NPU_CORE_2,  //< run on NPU core 1 and core 2.
  RKNN_NPU_CORE_UNDEFINED,
}

public enum LitePowerMode{
  LITE_POWER_HIGH = 0,       ///< Use Lite Backend with high power mode
  LITE_POWER_LOW = 1,        ///< Use Lite Backend with low power mode
  LITE_POWER_FULL = 2,       ///< Use Lite Backend with full power mode
  LITE_POWER_NO_BIND = 3,    ///< Use Lite Backend with no bind power mode
  LITE_POWER_RAND_HIGH = 4,  ///< Use Lite Backend with rand high mode
  LITE_POWER_RAND_LOW = 5    ///< Use Lite Backend with rand low power mode
}  

}