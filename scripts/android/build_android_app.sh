# build FastDeploy app
FASTDEPLOY_DIR=$(pwd)
JAVA_ANDROID_DIR=${FASTDEPLOY_DIR}/java/android
JNI_LIB_DIR=${JAVA_ANDROID_DIR}/fastdeploy
AAR_DEBUG_PACKAGE=${JNI_LIB_DIR}/build/outputs/aar/fastdeploy-debug.aar
APP_DIR=${JAVA_ANDROID_DIR}/app
APP_LIBS_DIR=${APP_DIR}/libs

cd ${JAVA_ANDROID_DIR}
# check aar package first!
echo "[INFO] --- JAVA_ANDROID_DIR: ${JAVA_ANDROID_DIR}"
if [ ! -d "${JNI_LIB_DIR}/build/outputs/aar" ]; then
   echo "-- [ERROR] ${JNI_LIB_DIR} not exists, please build aar package first!"
   exit 0
fi
ls -lh ${JNI_LIB_DIR}/build/outputs/aar/
if [ ! -d "${APP_LIBS_DIR}" ]; then
    mkdir -p "${APP_LIBS_DIR}" && echo "-- [INFO] Created ${APP_LIBS_DIR} !"
fi
# update aar package
echo "[INFO] --- Update aar package ..."
if [ -f "${APP_LIBS_DIR}/fastdeploy-android-sdk-latest-dev.aar" ]; then
    rm -f "${APP_LIBS_DIR}/fastdeploy-android-sdk-latest-dev.aar"
    echo "[INFO] --- Removed old aar package: ${APP_LIBS_DIR}/fastdeploy-android-sdk-latest-dev.aar"
fi
cp ${AAR_DEBUG_PACKAGE} ${APP_LIBS_DIR}/fastdeploy-android-sdk-latest-dev.aar
if [ -f "${APP_LIBS_DIR}/fastdeploy-android-sdk-latest-dev.aar" ]; then
    echo "[INFO] --- Update aar package done!"
fi
# build android app
echo "[INFO] --- Building FastDeploy Android App ..."
chmod +x gradlew
./gradlew app:assembleDebug
echo "[INFO] --- Built FastDeploy Android app."

# Usage:
# ./scripts/android/build_android_app.sh
