rm -rf build
mkdir build
cd build

cmake -DCMAKE_TOOLCHAIN_FILE=../fastdeploy-cann/cann.cmake -DFASTDEPLOY_INSTALL_DIR=fastdeploy-cann ..
make -j8
make install