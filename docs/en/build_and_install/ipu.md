
# How to Build IPU Deployment Environment

FastDeploy only supports Paddle Inference in the IPU environment.

## How to Build and Install C++ SDK

Prerequisite for Compiling on Linux:

- gcc/g++ >= 5.4 (8.2 is recommended)
- cmake >= 3.16.0, < 3.23.0
- popart >= 3.0.0

```bash
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy
mkdir build && cd build
cmake .. -DENABLE_PADDLE_BACKEND=ON \
         -DWITH_IPU=ON \
         -DCMAKE_INSTALL_PREFIX=${PWD}/compiled_fastdeploy_sdk \
         -DENABLE_VISION=ON
make -j8
make install
```

Once compiled, the C++ inference library is generated in the directory specified by `CMAKE_INSTALL_PREFIX`

## How to Build and Install Python SDK

Prerequisite for Compiling on Linux:

- gcc/g++ >= 5.4 (8.2 is recommended)
- cmake >= 3.16.0, < 3.23.0
- popart >= 3.0.0
- python >= 3.6

All compilation options are imported via environment variables

```bash
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/python
export ENABLE_VISION=ON
export ENABLE_PADDLE_BACKEND=ON
export WITH_IPU=ON

python setup.py build
python setup.py bdist_wheel
```

The compiled `wheel` package will be generated in the `FastDeploy/python/dist` directory once finished. Users can pip-install it directly.

During the compilation, if developers want to change the compilation parameters, it is advisable to delete the `build` and `.setuptools-cmake-build` subdirectories in the `FastDeploy/python` to avoid the possible impact from cache, and then recompile.
