# Linux & Mac编译

## 环境依赖

- cmake >= 3.12
- g++ >= 8.2
- cuda >= 11.2 (当WITH_GPU=ON)
- cudnn >= 8.0 (当WITH_GPU=ON)
- TensorRT >= 8.4 (当ENABLE_TRT_BACKEND=ON)

## 编译C++
```bash
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy
git checkout develop
mkdir build && cd build
cmake .. -DENABLE_ORT_BACKEND=ON \
         -DENABLE_VISION=ON \
         -DCMAKE_INSTALL_PREFIX=${PWD}/fastdeploy-0.0.3
make -j8
make install
```
编译后的预测库即在当前目录下的`fastdeploy-0.0.3`

## 编译Python安装包
```bash
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/python
git checkout develop
# Python通过export环境变量设置编译选项
export ENABLE_ORT_BACKEND=ON
export ENABLE_VISION=ON
python setup.py build
python setup.py bdist_wheel
```
编译后的wheel包即在当前目录下的`dist`目录中

更多编译选项说明参考[编译指南](./README.md)
