option(RK "Whether to enable RK deploy." ON)
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)
set(CMAKE_C_COMPILER "arm-linux-gnueabihf-gcc")
set(CMAKE_CXX_COMPILER "arm-linux-gnueabihf-g++")
set(CMAKE_CXX_FLAGS "-march=armv7-a -mfloat-abi=hard -mfpu=neon-vfpv4 ${CMAKE_CXX_FLAGS}")
set(CMAKE_C_FLAGS "-march=armv7-a -mfloat-abi=hard -mfpu=neon-vfpv4 ${CMAKE_C_FLAGS}" )

set(CMAKE_FIND_ROOT_PATH /usr/lib/gcc-cross/arm-linux-gnueabi )

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)