# A fake target to include all the libraries and tests the paddle2onnx module depends.
add_custom_target(p2o_compile_deps COMMAND echo 1)

# A function to grep LINK_ONLY dependencies from INTERFACE_LINK_LIBRARIES
function(regrex_link_only_libraries OUTPUT_DEPS PUBLIC_DEPS)
  string(JOIN "#" _public_deps ${PUBLIC_DEPS})
  string(REPLACE "$<LINK_ONLY:" "" _public_deps ${_public_deps})
  string(REPLACE ">" "" _public_deps ${_public_deps})
  string(REPLACE "#" ";" _public_deps ${_public_deps})
  set(${OUTPUT_DEPS} ${_public_deps} PARENT_SCOPE)
endfunction()

# Bundle several static libraries into one. This function is modified from Paddle Lite. 
# reference: https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/cmake/lite.cmake#L252
function(bundle_static_library tgt_name bundled_tgt_name fake_target)
  list(APPEND static_libs ${tgt_name})
  add_dependencies(p2o_compile_deps ${tgt_name})
  set(REDUNDANT_STATIC_LIBS __fake_no_lib) # may add some redundant libs later

  function(_recursively_collect_dependencies input_target)
    list(FIND REDUNDANT_STATIC_LIBS ${input_target} _input_redunant_id)
    if(${_input_redunant_id} GREATER 0)
      return()
    endif()
    set(_input_link_libraries LINK_LIBRARIES)
    # https://cmake.org/cmake/help/latest/prop_tgt/TYPE.html
    get_target_property(_input_type ${input_target} TYPE)
    if ((${_input_type} STREQUAL "INTERFACE_LIBRARY")
         OR (${_input_type} STREQUAL "STATIC_LIBRARY"))
      set(_input_link_libraries INTERFACE_LINK_LIBRARIES)
    endif()
    get_target_property(_public_dependencies ${input_target} ${_input_link_libraries})
    regrex_link_only_libraries(public_dependencies "${_public_dependencies}")
    
    foreach(dependency IN LISTS public_dependencies)
      if(TARGET ${dependency})
        get_target_property(alias ${dependency} ALIASED_TARGET)
        if (TARGET ${alias})
          set(dependency ${alias})
        endif()
        get_target_property(_type ${dependency} TYPE)
        list(FIND REDUNDANT_STATIC_LIBS ${dependency} _deps_redunant_id)
        if (${_type} STREQUAL "STATIC_LIBRARY" AND 
            (NOT (${_deps_redunant_id} GREATER 0)))
          list(APPEND static_libs ${dependency})
        endif()

        get_property(library_already_added
          GLOBAL PROPERTY _${tgt_name}_static_bundle_${dependency})
        if (NOT library_already_added)
          set_property(GLOBAL PROPERTY _${tgt_name}_static_bundle_${dependency} ON)
          if(NOT (${_deps_redunant_id} GREATER 0))
            _recursively_collect_dependencies(${dependency})
          endif()
        endif()
      endif()
    endforeach()
    set(static_libs ${static_libs} PARENT_SCOPE)
  endfunction()

  _recursively_collect_dependencies(${tgt_name})

  list(REMOVE_DUPLICATES static_libs)
  list(REMOVE_ITEM static_libs ${REDUNDANT_STATIC_LIBS})
  message(STATUS "Found all needed static libs from dependecy tree: ${static_libs}")

  set(bundled_tgt_full_name
    ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${bundled_tgt_name}${CMAKE_STATIC_LIBRARY_SUFFIX})

  message(STATUS "Use bundled_tgt_full_name:  ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${bundled_tgt_name}${CMAKE_STATIC_LIBRARY_SUFFIX}")

  if(WIN32)
    message(FATAL_ERROR "Not support paddle2onnx static lib for windows now.")
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
    # add protobuf
    file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/${bundled_tgt_name}.ar.in
      "ADDLIB ${PROTOBUF_LIBRARIES}\n")
    
    file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/${bundled_tgt_name}.ar.in "SAVE\n")
    file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/${bundled_tgt_name}.ar.in "END\n")
    
    file(GENERATE
      OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${bundled_tgt_name}.ar
      INPUT ${CMAKE_CURRENT_BINARY_DIR}/${bundled_tgt_name}.ar.in)

    set(ar_tool ${CMAKE_AR})
    if (CMAKE_INTERPROCEDURAL_OPTIMIZATION)
      set(ar_tool ${CMAKE_CXX_COMPILER_AR})
    endif()
    message(STATUS "Found ar_tool: ${ar_tool}")

    add_custom_command(
      TARGET ${fake_target} PRE_BUILD
      COMMAND rm -f ${bundled_tgt_full_name}
      COMMAND ${ar_tool} -M < ${CMAKE_CURRENT_BINARY_DIR}/${bundled_tgt_name}.ar
      COMMENT "Bundling ${bundled_tgt_name}"
      COMMAND ${CMAKE_STRIP} --strip-unneeded ${CMAKE_CURRENT_BINARY_DIR}/lib${bundled_tgt_name}.a
      COMMENT "Stripped unneeded symbols in ${bundled_tgt_name}"
      DEPENDS ${tgt_name}
      VERBATIM)
  else()
    foreach(lib ${static_libs})
      set(libfiles ${libfiles} $<TARGET_FILE:${lib}>)
    endforeach()
    # add protobuf
    set(libfiles ${libfiles} ${PROTOBUF_LIBRARIES})
    message(STATUS "Static libfiles for osx: ${libfiles}")
    add_custom_command(
      TARGET ${fake_target} PRE_BUILD
      COMMAND rm -f ${bundled_tgt_full_name}
      COMMAND /usr/bin/libtool -static -o ${bundled_tgt_full_name} ${libfiles}
      COMMENT "Bundling ${bundled_tgt_name}"
      COMMAND ${CMAKE_STRIP} -S ${CMAKE_CURRENT_BINARY_DIR}/lib${bundled_tgt_name}.a
      COMMENT "Stripped unneeded symbols in ${bundled_tgt_name}"
      DEPENDS ${tgt_name}
    )
  endif()

  add_library(${bundled_tgt_name} STATIC IMPORTED GLOBAL)
  set_property(TARGET ${bundled_tgt_name} PROPERTY IMPORTED_LOCATION
                                         ${bundled_tgt_full_name})          
  add_dependencies(${bundled_tgt_name} ${fake_target})
  add_dependencies(${bundled_tgt_name} ${tgt_name})

endfunction()
