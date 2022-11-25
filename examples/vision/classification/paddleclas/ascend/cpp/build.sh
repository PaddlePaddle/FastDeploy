rm -rf build
mkdir build
cd build

cmake .. -DFASTDEPLOY_INSTALL_DIR=../../../../../../build/fastdeploy-cann/
make -j8
