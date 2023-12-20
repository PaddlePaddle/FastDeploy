# This script offer a demo to build triton fastdeploy backend only.

cd serving
rm -rf build && mkdir build

cd build
cmake .. -DFASTDEPLOY_DIR=${FD_GPU_SDK} -DTRITON_COMMON_REPO_TAG=r21.10 -DTRITON_CORE_REPO_TAG=r21.10 -DTRITON_BACKEND_REPO_TAG=r21.10;
make -j`nproc`