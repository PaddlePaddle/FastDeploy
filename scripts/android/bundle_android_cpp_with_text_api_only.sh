#!/bin/bash
set -e
set +x

FASTDEPLOY_DIR=$(pwd)
BUILT_PACKAGE_DIR=build/Android
CXX_PACKAGE_PREFIX=fastdeploy-android-latest-shared-dev
CXX_PACKAGE_NAME=${BUILT_PACKAGE_DIR}/${CXX_PACKAGE_PREFIX}
ARMV8_CXX_PACKAGE_NAME=${BUILT_PACKAGE_DIR}/arm64-v8a-api-21/install
ARMV7_CXX_PACKAGE_NAME=${BUILT_PACKAGE_DIR}/armeabi-v7a-api-21/install

# check package name
echo "[INFO] --- FASTDEPLOY_DIR: ${FASTDEPLOY_DIR}"

# check arm v7 & v8 c++ sdk
if [ ! -d "${BUILT_PACKAGE_DIR}" ]; then
    echo "[ERROR] --- ${BUILT_PACKAGE_DIR} not exist, please build c++ sdk first!"
    exit 0
fi
if [ ! -d "${ARMV8_CXX_PACKAGE_NAME}" ]; then
    echo "[ERROR] --- ${ARMV8_CXX_PACKAGE_NAME} not exist, please build c++ sdk first!"
    exit 0
fi
if [ ! -d "${ARMV7_CXX_PACKAGE_NAME}" ]; then
    echo "[ERROR] --- ${ARMV7_CXX_PACKAGE_NAME} not exist, please build c++ sdk first!"
    exit 0
fi

# remove old package
echo "[INFO] --- Packing ${CXX_PACKAGE_NAME} package ..."
if [ -d "${CXX_PACKAGE_NAME}" ]; then
	rm -rf ${CXX_PACKAGE_NAME}
    echo "[INFO] --- Removed old ${CXX_PACKAGE_NAME} done !"
    if [ -f "${CXX_PACKAGE_NAME}.tgz" ]; then
        rm ${CXX_PACKAGE_NAME}.tgz
        echo "[INFO] --- Removed old ${CXX_PACKAGE_NAME} done !"
    fi
fi

# package latest c++ sdk
mkdir ${CXX_PACKAGE_NAME}
echo "[INFO] --- Collecting package contents ..."
cp -r ${ARMV7_CXX_PACKAGE_NAME}/* ${CXX_PACKAGE_NAME}/
cp -r ${ARMV8_CXX_PACKAGE_NAME}/* ${CXX_PACKAGE_NAME}/
if [ -d "${CXX_PACKAGE_NAME}/examples" ]; then
    rm -rf ${CXX_PACKAGE_NAME}/examples
fi
echo "[INFO] --- Removed examples files ..."
echo "[INFO] --- Removing static .a files: "
static_files=$(find ${CXX_PACKAGE_NAME}/third_libs/install/ -name "*.a")
if [ ${#static_files[@]} -gt 10 ]; then
    echo "${#static_files[@]}: ${static_files}"
    rm $(find ${CXX_PACKAGE_NAME}/third_libs/install/ -name "*.a")
fi
echo "[INFO] --- Taring ${CXX_PACKAGE_NAME}.tgz package ..."
tar -zcvf ${CXX_PACKAGE_NAME}.tgz ${CXX_PACKAGE_NAME}/* >> ${BUILT_PACKAGE_DIR}/pkg.log 2>&1
echo "[INFO] --- Package ${CXX_PACKAGE_NAME}.tgz done ! Package size info: "
du -sh ${BUILT_PACKAGE_DIR}/* | grep ${CXX_PACKAGE_PREFIX}

# Usage:
# ./scripts/android/bundle_android_cpp_with_text_api_only.sh
