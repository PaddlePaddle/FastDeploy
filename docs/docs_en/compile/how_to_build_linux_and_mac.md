# Compile on Linux & Mac

## Dependencies

- cmake >= 3.12
- g++ >= 8.2
- cuda >= 11.2 (WITH_GPU=ON)
- cudnn >= 8.0 (WITH_GPU=ON)
- TensorRT >= 8.4 (ENABLE_TRT_BACKEND=ON)

## Compile C++

```bash
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy
git checkout develop
mkdir build & cd build
cmake .. -DENABLE_ORT_BACKEND=ON \
         -DENABLE_VISION=ON \
         -DCMAKE_INSTALL_PREFIX=${PWD}/fastdeploy-0.0.3
make -j8
make install
```

The compiled prediction library is in the `fastdeploy-0.0.3`of current directory 

## Compile Python

```bash
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy
git checkout develop
# set compile options via export environment variable on Python
export ENABLE_ORT_BACKEND=ON
export ENABLE_VISION=ON
python setup.py build
python setup.py bdist_wheel
```

The compiled wheel package is in the `dist` directory of current directory

For more details, please refer to [Compile Readme](./README.md)
