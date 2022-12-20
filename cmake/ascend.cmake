if(NOT CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
  message(FATAL_ERROR "Huawei Ascend NPU is supported on Linux aarch64 platform for now.")
endif()

if(NOT ${ENABLE_LITE_BACKEND})
	set(ENABLE_LITE_BACKEND ON)
endif()

if (NOT BUILD_FASTDEPLOY_PYTHON)
	message(STATUS "Build FastDeploy Ascend C++ library.")
	if(NOT PADDLELITE_URL)
		set(PADDLELITE_URL "https://bj.bcebos.com/fastdeploy/test/lite-linux_arm64_huawei_ascend_npu_1121.tgz")
	endif()
else ()
	message(STATUS "Build FastDeploy Ascend Python library.")
	if(NOT PADDLELITE_URL)
    set(PADDLELITE_URL "https://bj.bcebos.com/fastdeploy/test/lite-linux_arm64_huawei_ascend_npu_python_1207.tgz")
  endif()
  execute_process(COMMAND sh -c "ls *.so*" WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/third_libs/install/paddlelite/lib
  COMMAND sh -c "xargs ${PATCHELF_EXE} --set-rpath '$ORIGIN'" WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/third_libs/install/paddlelite/lib
          RESULT_VARIABLE result
                OUTPUT_VARIABLE curr_out
                ERROR_VARIABLE  curr_out)
  if(ret EQUAL "1")
    message(FATAL_ERROR "Failed to patchelf Paddle Lite libraries when using Ascend.")
  endif()
  message(STATUS "result:${result} out:${curr_out}")
endif()	
