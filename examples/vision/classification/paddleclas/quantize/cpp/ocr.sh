rm -rf build
mkdir build

cd build

#/xieyunyao/project/FastDeploy

cmake .. -DFASTDEPLOY_INSTALL_DIR=/xieyunyao/project/FastDeploy

make -j
