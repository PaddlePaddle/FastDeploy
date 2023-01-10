if(NOT ENABLE_LITE_BACKEND)
  message("Will force to set ENABLE_LITE_BACKEND when build with KunlunXin.")
  set(ENABLE_LITE_BACKEND ON)
endif()

if(NOT CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64")
  message(FATAL_ERROR "KunlunXin XPU is only supported on Linux x64 platform")
endif()

if(NOT PADDLELITE_URL)
  set(PADDLELITE_URL "https://bj.bcebos.com/fastdeploy/third_libs/lite-linux-x64-xpu-20221215.tgz")
endif()