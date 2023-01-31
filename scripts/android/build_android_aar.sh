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
# Fix Paddle Lite headers to support FastDeploy static lib
SDK_LITE_INCLUDE_DIR=${CXX_PACKAGE_NAME}/third_libs/install/paddlelite/include
ARMV7_LITE_INCLUDE_DIR=${ARMV7_CXX_PACKAGE_NAME}/third_libs/install/paddlelite/include
ARMV8_LITE_INCLUDE_DIR=${ARMV8_CXX_PACKAGE_NAME}/third_libs/install/paddlelite/include
rm ${SDK_LITE_INCLUDE_DIR}/paddle_use_kernels.h
cat ${ARMV8_LITE_INCLUDE_DIR}/paddle_use_kernels.h | head -n 3 >> ${SDK_LITE_INCLUDE_DIR}/paddle_use_kernels.h
echo "#if (defined(__aarch64__) || defined(_M_ARM64))" >> ${SDK_LITE_INCLUDE_DIR}/paddle_use_kernels.h
cat ${ARMV8_LITE_INCLUDE_DIR}/paddle_use_kernels.h | grep -v "#" | grep "USE" >> ${SDK_LITE_INCLUDE_DIR}/paddle_use_kernels.h
echo "#else" >> ${SDK_LITE_INCLUDE_DIR}/paddle_use_kernels.h
cat ${ARMV7_LITE_INCLUDE_DIR}/paddle_use_kernels.h | grep -v "#" | grep "USE" >> ${SDK_LITE_INCLUDE_DIR}/paddle_use_kernels.h
echo "#endif" >> ${SDK_LITE_INCLUDE_DIR}/paddle_use_kernels.h
echo "[INFO] --- Fixed Paddle Lite paddle_use_kernels.h to support FastDeploy static lib."
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

# update c++ sdk to jni lib
echo "[INFO] --- Update c++ sdk for jni lib ..."
JAVA_ANDROID_DIR=${FASTDEPLOY_DIR}/java/android
JNI_LIB_DIR=${JAVA_ANDROID_DIR}/fastdeploy
CXX_LIB_FOR_JNI_DIR=${JNI_LIB_DIR}/libs/${CXX_PACKAGE_PREFIX}
if [ -d "${CXX_LIB_FOR_JNI_DIR}" ]; then
	rm -rf ${CXX_LIB_FOR_JNI_DIR}
	echo "[INFO] --- Remove old ${CXX_LIB_FOR_JNI_DIR} done!"
fi
cp -r ${CXX_PACKAGE_NAME} ${JNI_LIB_DIR}/libs
echo "[INFO] --- Update ${CXX_LIB_FOR_JNI_DIR} done!"

# build java aar package
cd ${JAVA_ANDROID_DIR}
echo "[INFO] --- JAVA_ANDROID_DIR: ${JAVA_ANDROID_DIR}"
echo "[INFO] --- Building java aar package ... "
chmod +x gradlew
./gradlew fastdeploy:assembleDebug
echo "[INFO] --- Built java aar package!"
ls -lh ${JNI_LIB_DIR}/build/outputs/aar/

# Usage:
# ./scripts/android/build_android_aar.sh
