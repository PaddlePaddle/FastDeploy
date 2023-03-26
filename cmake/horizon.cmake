set(DNN_PATH "~/.horizon/ddk/xj3_aarch64/dnn/")
set(APPSDK_PATH "~/.horizon/ddk/xj3_aarch64/appsdk/appuser/")

set(DNN_LIB_PATH ${DNN_PATH}/lib)
set(APPSDK_LIB_PATH ${APPSDK_PATH}/lib/hbbpu)
set(BPU_libs dnn cnn_intf hbrt_bernoulli_aarch64)

include_directories(${DNN_PATH}/include
                    ${APPSDK_PATH}/include)
