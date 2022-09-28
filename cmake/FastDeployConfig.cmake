#    This file will define the following variables for find_package method:
#      - FastDeploy_LIBS                 : The list of libraries to link against.
#      - FastDeploy_INCLUDE_DIRS         : The FastDeploy include directories.
#      - FastDeploy_Found                : The status of FastDeploy

include(${CMAKE_CURRENT_LIST_DIR}/FastDeploy.cmake)
# setup FastDeploy cmake variables
set(FastDeploy_LIBS ${FASTDEPLOY_LIBS})
set(FastDeploy_INCLUDE_DIRS ${FASTDEPLOY_INCS})
set(FastDeploy_FOUND TRUE)    