function(add_fastdeploy_executable FIELD CC_FILE)
  # temp target name/file var in function scope
  set(TEMP_TARGET_FILE ${CC_FILE})
  string(REGEX MATCHALL "[0-9A-Za-z_]*.cc" FILE_NAME ${CC_FILE})
  string(REGEX REPLACE ".cc" "" FILE_PREFIX ${FILE_NAME})
  set(TEMP_TARGET_NAME ${FIELD}_${FILE_PREFIX})
  if (EXISTS ${TEMP_TARGET_FILE} AND TARGET fastdeploy)
    add_executable(${TEMP_TARGET_NAME} ${TEMP_TARGET_FILE})
    target_link_libraries(${TEMP_TARGET_NAME} PUBLIC fastdeploy)
    message(STATUS "  Added FastDeploy Executable       : ${TEMP_TARGET_NAME}")
  endif()
  unset(TEMP_TARGET_FILE)
  unset(TEMP_TARGET_NAME)
endfunction()

# vision examples
if(WITH_VISION_EXAMPLES AND EXISTS ${PROJECT_SOURCE_DIR}/examples/vision)
  message(STATUS "")
  message(STATUS "*************FastDeploy Examples Summary**********")
  file(GLOB ALL_VISION_EXAMPLE_SRCS ${PROJECT_SOURCE_DIR}/examples/vision/*.cc)
  foreach(_CC_FILE ${ALL_VISION_EXAMPLE_SRCS})
    add_fastdeploy_executable(vision ${_CC_FILE})
  endforeach()
endif()

# other examples ...
