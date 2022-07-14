function(add_fastdeploy_executable field url model)   
  # temp target name/file var in function scope
  set(TEMP_TARGET_FILE ${PROJECT_SOURCE_DIR}/examples/${field}/${url}_${model}.cc)
  set(TEMP_TARGET_NAME ${field}_${url}_${model})
  if (EXISTS ${TEMP_TARGET_FILE} AND TARGET fastdeploy)
    add_executable(${TEMP_TARGET_NAME} ${TEMP_TARGET_FILE})
    target_link_libraries(${TEMP_TARGET_NAME} PUBLIC fastdeploy)
    message(STATUS "Found source file: [${field}/${url}_${model}.cc], ADD!!! fastdeploy executable: [${TEMP_TARGET_NAME}] !")
  else ()
    message(WARNING "Can not found source file: [${field}/${url}_${model}.cc], SKIP!!! fastdeploy executable: [${TEMP_TARGET_NAME}] !")          
  endif()
  unset(TEMP_TARGET_FILE)
  unset(TEMP_TARGET_NAME)
endfunction()

# vision examples
if (WTIH_VISION_EXAMPLES)
  add_fastdeploy_executable(vision ultralytics yolov5)
  add_fastdeploy_executable(vision meituan yolov6)
endif()

# other examples ...