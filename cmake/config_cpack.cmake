if(NOT UNIX)
  return()
endif()

set(PACKAGE_SYS_VERSION "linux")
if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
  set(PACKAGE_SYS_VERSION "${PACKAGE_SYS_VERSION}-aarch64")
else()
  set(PACKAGE_SYS_VERSION "${PACKAGE_SYS_VERSION}-x64")
endif()
if(WITH_GPU)
  set(PACKAGE_SYS_VERSION "${PACKAGE_SYS_VERSION}-gpu")
endif()

# set(CPACK_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION ON)
set(CPACK_VERBATIM_VARIABLES TRUE)
set(CPACK_GENERATOR DEB)
set(CPACK_THREADS 0)
set(CPACK_PACKAGE_CONTACT "fastdeploy@baidu.com")
set(CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
set(CPACK_PACKAGE_VERSION "${FASTDEPLOY_VERSION}")
set(CPACK_PACKAGE_FILE_NAME "${PROJECT_NAME}-${PACKAGE_SYS_VERSION}-${FASTDEPLOY_VERSION}")
set(CPACK_PACKAGE_NAME "${PROJECT_NAME}")
set(CPACK_DEBIAN_PACKAGE_CONTROL_STRICT_PERMISSION TRUE)

configure_file(cpack/debian_postinst.in cpack/postinst @ONLY)
configure_file(cpack/debian_prerm.in cpack/prerm @ONLY)

set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA
    "${CMAKE_CURRENT_BINARY_DIR}/cpack/postinst"
    "${CMAKE_CURRENT_BINARY_DIR}/cpack/prerm")

include(CPack)
