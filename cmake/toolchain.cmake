if (DEFINED TARGET_ABI)
    set(CMAKE_SYSTEM_NAME Linux)
    set(CMAKE_BUILD_TYPE MinSizeRel)
    if(${TARGET_ABI} MATCHES "armhf")
        set(CMAKE_SYSTEM_PROCESSOR arm)
        set(CMAKE_C_COMPILER "arm-linux-gnueabihf-gcc")
        set(CMAKE_CXX_COMPILER "arm-linux-gnueabihf-g++")
        set(CMAKE_CXX_FLAGS "-march=armv7-a -mfloat-abi=hard -mfpu=neon-vfpv4 ${CMAKE_CXX_FLAGS}")
        set(CMAKE_C_FLAGS "-march=armv7-a -mfloat-abi=hard -mfpu=neon-vfpv4 ${CMAKE_C_FLAGS}" )
    elseif(${TARGET_ABI} MATCHES "arm64")
        set(CMAKE_SYSTEM_PROCESSOR aarch64)
        set(CMAKE_C_COMPILER "aarch64-linux-gnu-gcc")
        set(CMAKE_CXX_COMPILER "aarch64-linux-gnu-g++")
        set(CMAKE_CXX_FLAGS "-march=armv8-a ${CMAKE_CXX_FLAGS}")
        set(CMAKE_C_FLAGS "-march=armv8-a ${CMAKE_C_FLAGS}")
    else()
        message(FATAL_ERROR "When cross-compiling, please set the -DTARGET_ABI to arm64 or armhf.")
    endif()
endif()

