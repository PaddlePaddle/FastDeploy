if (NOT DEFINED TARGET_ABI)
    message(WARNING "TARGET_ABI is DEFINED, will set to TARGET_ABI=amd64")
    set(TARGET_ABI amd64)
endif()

if(NOT ${ENABLE_LITE_BACKEND})
    message(WARNING "While compiling with -DWITH_XPU=ON, will force to set -DENABLE_LITE_BACKEND=ON")
    set(ENABLE_LITE_BACKEND ON)
endif()