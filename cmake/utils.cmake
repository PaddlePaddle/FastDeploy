# This function comes from https://blog.csdn.net/yindongjie1221/article/details/90614261
function(redefine_file_macro targetname)
    get_target_property(source_files "${targetname}" SOURCES)
    foreach(sourcefile ${source_files})
        get_property(defs SOURCE "${sourcefile}"
            PROPERTY COMPILE_DEFINITIONS)
        get_filename_component(filepath "${sourcefile}" ABSOLUTE)
        string(REPLACE ${PROJECT_SOURCE_DIR}/ "" relpath ${filepath})
        list(APPEND defs "__REL_FILE__=\"${relpath}\"")
        set_property(
            SOURCE "${sourcefile}"
            PROPERTY COMPILE_DEFINITIONS ${defs}
            )
    endforeach()
endfunction()

function(download_and_decompress url filename decompress_dir)
  if(NOT EXISTS ${filename})
    message("Downloading file from ${url} to ${filename} ...")
    file(DOWNLOAD ${url} "${filename}.tmp" SHOW_PROGRESS)
    file(RENAME "${filename}.tmp" ${filename})
  endif()
  if(NOT EXISTS ${decompress_dir})
    file(MAKE_DIRECTORY ${decompress_dir})
  endif()
  message("Decompress file ${filename} ...")
  execute_process(COMMAND ${CMAKE_COMMAND} -E tar -xf ${filename} WORKING_DIRECTORY ${decompress_dir})
endfunction()

function(get_openvino_libs OPENVINO_RUNTIME_DIR)
  set(LIB_LIST "")
  find_library(OPENVINO_LIB openvino PATHS ${OPENVINO_RUNTIME_DIR}/lib/ ${OPENVINO_RUNTIME_DIR}/lib/intel64 NO_DEFAULT_PATH)
  list(APPEND LIB_LIST ${OPENVINO_LIB})

  set(TBB_DIR ${OPENVINO_RUNTIME_DIR}/3rdparty/tbb/cmake)
  find_package(TBB PATHS ${TBB_DIR})
  if (TBB_FOUND)
    list(APPEND LIB_LIST ${TBB_IMPORTED_TARGETS})
  else()
    # TODO(zhoushunjie): Use openvino with tbb on linux in future.
    set(OMP_LIB "${OPENVINO_RUNTIME_DIR}/3rdparty/omp/lib/libiomp5.so")
    list(APPEND LIB_LIST ${OMP_LIB})
  endif()
  set(OPENVINO_LIBS ${LIB_LIST} PARENT_SCOPE)
endfunction()

function(remove_duplicate_libraries libraries)
  list(LENGTH ${libraries} lib_length)
  set(libraries_temp "")
  set(full_libraries "")
  foreach(lib_path ${${libraries}})
    get_filename_component(lib_name ${lib_path} NAME)
    list(FIND libraries_temp ${lib_name} lib_idx)
    if (${lib_idx} EQUAL -1)
      list(APPEND libraries_temp ${lib_name})
      list(APPEND full_libraries ${lib_path})
    endif()
  endforeach()
  set(${libraries} ${full_libraries} PARENT_SCOPE)
endfunction()

function(get_windows_path win_path origin_path)
  STRING(REGEX REPLACE "/" "\\\\" _win_path ${origin_path})
  set(${win_path} ${_win_path} PARENT_SCOPE)
endfunction()

function(get_osx_architecture)
  if (CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
    set(CURRENT_OSX_ARCH "arm64" PARENT_SCOPE)
  elseif(CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64")
    set(CURRENT_OSX_ARCH "x86_64" PARENT_SCOPE)
  else()
    set(CURRENT_OSX_ARCH ${CMAKE_HOST_SYSTEM_PROCESSOR} PARENT_SCOPE)
  endif()
endfunction()

#only for windows
function(create_static_lib TARGET_NAME)
  set(libs ${ARGN})
  list(REMOVE_DUPLICATES libs)
    set(dummy_index 1)
    set(dummy_offset 1)
    # the dummy target would be consisted of limit size libraries
    set(dummy_limit 60)
    list(LENGTH libs libs_len)

    foreach(lib ${libs})
      list(APPEND dummy_list ${lib})
      list(LENGTH dummy_list listlen)
      if ((${listlen} GREATER ${dummy_limit}) OR (${dummy_offset} EQUAL ${libs_len}))
        merge_static_libs(${TARGET_NAME}_dummy_${dummy_index} ${dummy_list})
        set(dummy_list)
        list(APPEND ${TARGET_NAME}_dummy_list ${TARGET_NAME}_dummy_${dummy_index})
        MATH(EXPR dummy_index "${dummy_index}+1")
      endif()
      MATH(EXPR dummy_offset "${dummy_offset}+1")
    endforeach()
    merge_static_libs(${TARGET_NAME} ${${TARGET_NAME}_dummy_list})
endfunction()

# Bundle several static libraries into one.
# reference: https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/cmake/lite.cmake#L252

# A fake target to include all the libraries and tests the fastdeploy module depends.
add_custom_target(fd_compile_deps COMMAND echo 1)

function(bundle_static_library tgt_name bundled_tgt_name fake_target)
  list(APPEND static_libs fastdelpoy_dummy)
  add_dependencies(fd_compile_deps ${fake_target})

  function(_recursively_collect_dependencies input_target)
    set(_input_link_libraries LINK_LIBRARIES)
    get_target_property(_input_type ${input_target} TYPE)
    if (${_input_type} STREQUAL "INTERFACE_LIBRARY")
      set(_input_link_libraries INTERFACE_LINK_LIBRARIES)
    endif()
    get_target_property(public_dependencies ${input_target} ${_input_link_libraries})
    foreach(dependency IN LISTS public_dependencies)
      if(TARGET ${dependency})
        get_target_property(alias ${dependency} ALIASED_TARGET)
        if (TARGET ${alias})
          set(dependency ${alias})
        endif()
        get_target_property(_type ${dependency} TYPE)
        if (${_type} STREQUAL "STATIC_LIBRARY")
          list(APPEND static_libs ${dependency})
        endif()

        get_property(library_already_added
          GLOBAL PROPERTY _${tgt_name}_static_bundle_${dependency})
        if (NOT library_already_added)
          set_property(GLOBAL PROPERTY _${tgt_name}_static_bundle_${dependency} ON)
          _recursively_collect_dependencies(${dependency})
        endif()
      endif()
    endforeach()
    set(static_libs ${static_libs} PARENT_SCOPE)
  endfunction()

  _recursively_collect_dependencies(${tgt_name})

  list(REMOVE_DUPLICATES static_libs)

  set(bundled_tgt_full_name
    ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${bundled_tgt_name}${CMAKE_STATIC_LIBRARY_SUFFIX})

  message(STATUS "bundled_tgt_full_name:  ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${bundled_tgt_name}${CMAKE_STATIC_LIBRARY_SUFFIX}")

  if(WIN32)
    set(dummy_tgt_name dummy_${bundled_tgt_name})
    create_static_lib(${bundled_tgt_name} ${static_libs})
    add_custom_target(${fake_target} ALL DEPENDS ${bundled_tgt_name})
    add_dependencies(${fake_target} ${tgt_name})

    add_library(${dummy_tgt_name} STATIC IMPORTED)
    set_target_properties(${dummy_tgt_name}
      PROPERTIES
        IMPORTED_LOCATION ${bundled_tgt_full_name}
        INTERFACE_INCLUDE_DIRECTORIES $<TARGET_PROPERTY:${tgt_name},INTERFACE_INCLUDE_DIRECTORIES>)
    add_dependencies(${dummy_tgt_name} ${fake_target})
    return()
  endif()

  add_custom_target(${fake_target} ALL COMMAND ${CMAKE_COMMAND} -E echo "Building fake_target ${fake_target}")
  add_dependencies(${fake_target} ${tgt_name})

  if(NOT IOS AND NOT APPLE)
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/${bundled_tgt_name}.ar.in
      "CREATE ${bundled_tgt_full_name}\n" )

    foreach(tgt IN LISTS static_libs)
      file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/${bundled_tgt_name}.ar.in
        "ADDLIB $<TARGET_FILE:${tgt}>\n")
    endforeach()

    file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/${bundled_tgt_name}.ar.in "SAVE\n")
    file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/${bundled_tgt_name}.ar.in "END\n")

    file(GENERATE
      OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${bundled_tgt_name}.ar
      INPUT ${CMAKE_CURRENT_BINARY_DIR}/${bundled_tgt_name}.ar.in)

    set(ar_tool ${CMAKE_AR})
    if (CMAKE_INTERPROCEDURAL_OPTIMIZATION)
      set(ar_tool ${CMAKE_CXX_COMPILER_AR})
    endif()
    message(STATUS "ar_tool: ${ar_tool}")

    add_custom_command(
      TARGET ${fake_target} PRE_BUILD
      COMMAND rm -f ${bundled_tgt_full_name}
      COMMAND ${ar_tool} -M < ${CMAKE_CURRENT_BINARY_DIR}/${bundled_tgt_name}.ar
      COMMENT "Bundling ${bundled_tgt_name}"
      DEPENDS ${tgt_name}
      VERBATIM)
  else()
    foreach(lib ${static_libs})
      set(libfiles ${libfiles} $<TARGET_FILE:${lib}>)
    endforeach()
    add_custom_command(
      TARGET ${fake_target} PRE_BUILD
      COMMAND rm -f ${bundled_tgt_full_name}
      COMMAND /usr/bin/libtool -static -o ${bundled_tgt_full_name} ${libfiles}
      DEPENDS ${tgt_name}
    )
  endif()

  add_library(${bundled_tgt_name} STATIC IMPORTED GLOBAL)
  set_property(TARGET ${bundled_tgt_name} PROPERTY IMPORTED_LOCATION
                                         ${bundled_tgt_full_name})
  if(TARGET ${bundled_tgt_name})                                       
    message(STATUS "bundled_tgt_name: ${bundled_tgt_name}")     
  endif()                                  
  add_dependencies(${bundled_tgt_name} ${fake_target})
  add_dependencies(${bundled_tgt_name} fastdelpoy_dummy)

endfunction()

