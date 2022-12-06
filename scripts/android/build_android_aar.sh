#!/bin/bash
set -e
set +x

PACKAGE_VERSION=$1
FASTDEPLOY_DIR=$(pwd)
BUILT_PACKAGE_DIR=build/Android
CXX_PACKAGE_PREFIX=fastdeploy-android-${PACKAGE_VERSION}-shared
CXX_PACKAGE_NAME=${BUILT_PACKAGE_DIR}/${CXX_PACKAGE_PREFIX}
ARMV8_CXX_PACKAGE_NAME=${BUILT_PACKAGE_DIR}/arm64-v8a-api-21/install
ARMV7_CXX_PACKAGE_NAME=${BUILT_PACKAGE_DIR}/armeabi-v7a-api-21/install

# check package name
echo "[INFO] --- FASTDEPLOY_DIR: ${FASTDEPLOY_DIR}"
if [ "$PACKAGE_VERSION" = "dev" ]; then
    CXX_PACKAGE_PREFIX=fastdeploy-android-latest-shared-dev
    CXX_PACKAGE_NAME=${BUILT_PACKAGE_DIR}/fastdeploy-android-latest-shared-dev
fi

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
	echo "[INFO] --- Removed old package done !"
	rm ${CXX_PACKAGE_NAME}.tgz
	rm -rf ${CXX_PACKAGE_NAME}
fi

# package latest c++ sdk
mkdir ${CXX_PACKAGE_NAME}
echo "[INFO] --- Collecting package contents ..."
cp -r ${ARMV7_CXX_PACKAGE_NAME}/* ${CXX_PACKAGE_NAME}/
cp -r ${ARMV8_CXX_PACKAGE_NAME}/* ${CXX_PACKAGE_NAME}/
rm -rf ${CXX_PACKAGE_NAME}/examples
echo "[INFO] --- Removed examples files ..."
echo "[INFO] --- Removing static .a files: "
find ${CXX_PACKAGE_NAME}/third_libs/install/ -name "*.a"
rm $(find ${CXX_PACKAGE_NAME}/third_libs/install/ -name "*.a")
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
# ./scripts/android/build_android_aar.sh dev
