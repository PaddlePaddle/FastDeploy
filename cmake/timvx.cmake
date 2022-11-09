if (NOT DEFINED TARGET_ABI)
    set(CMAKE_SYSTEM_NAME Linux)
    set(CMAKE_SYSTEM_PROCESSOR arm)
    set(CMAKE_C_COMPILER "arm-linux-gnueabihf-gcc")
    set(CMAKE_CXX_COMPILER "arm-linux-gnueabihf-g++")
    set(CMAKE_CXX_FLAGS "-march=armv7-a -mfloat-abi=hard -mfpu=neon-vfpv4 ${CMAKE_CXX_FLAGS}")
    set(CMAKE_C_FLAGS "-march=armv7-a -mfloat-abi=hard -mfpu=neon-vfpv4 ${CMAKE_C_FLAGS}" )
    set(TARGET_ABI armhf)
    set(CMAKE_BUILD_TYPE MinSizeRel)
else()
    if(NOT ${ENABLE_LITE_BACKEND})
        message(WARNING "Only support Lite backend for TIMVX now. Please set ENABLE_LITE_BACKEND=ON.")
        set(ENABLE_LITE_BACKEND ON)
    endif()
    if(${ENABLE_PADDLE_FRONTEND})
        message(WARNING "Not support Paddle front for TIMVX now. Please set ENABLE_PADDLE_FRONTEND=OFF.")
        set(ENABLE_PADDLE_FRONTEND OFF)
    endif()
    if(${ENABLE_ORT_BACKEND})
        message(WARNING "Not support ONNXRuntime backend for TIMVX now. Please set ENABLE_ORT_BACKEND=OFF.")
        set(ENABLE_ORT_BACKEND OFF)
    endif()
    if(${ENABLE_PADDLE_BACKEND})
        message(WARNING "Not support Paddle backend for TIMVX now. Please set ENABLE_PADDLE_BACKEND=OFF.")
        set(ENABLE_PADDLE_BACKEND OFF)
    endif()
    if(${ENABLE_OPENVINO_BACKEND})
        message(WARNING "Not support OpenVINO backend for TIMVX now. Please set ENABLE_OPENVINO_BACKEND=OFF.")
        set(ENABLE_OPENVINO_BACKEND OFF)
    endif()
    if(${ENABLE_TRT_BACKEND})
        message(WARNING "Not support TensorRT backend for TIMVX now. Please set ENABLE_TRT_BACKEND=OFF.")
        set(ENABLE_TRT_BACKEND OFF)
    endif()

    if(${WITH_GPU})
        message(WARNING "Cannot enable GPU while compling for TIMVX.")
        set(WITH_GPU OFF)
    endif()

    if(${ENABLE_OPENCV_CUDA})
        message(WARNING "Cannot enable opencv with cuda for TIMVX, please set -DENABLE_OPENCV_CUDA=OFF.") 
        set(ENABLE_OPENCV_CUDA OFF) 
    endif()

    if(${ENABLE_TEXT})
        set(ENABLE_TEXT OFF CACHE BOOL "Force ENABLE_TEXT OFF" FORCE)
        message(STATUS "Force ENABLE_TEXT OFF, We do not support faster_tokenizer for TIMVX  now.")
    endif()
    if (DEFINED CMAKE_INSTALL_PREFIX)
        install(FILES ${PROJECT_SOURCE_DIR}/cmake/timvx.cmake DESTINATION ${CMAKE_INSTALL_PREFIX})
    endif()
endif()


