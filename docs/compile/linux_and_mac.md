# Linux & Mac编译

## 编译C++
```
git clone https://gitee.com/jiangjiajun/FastDeploy.git
cd FastDeploy
git submodule init
git submodule update
mkdir build & cd build
cmake .. -DENABLE_ORT_BACKEND=ON \
         -DENABLE_VISION=ON \
         -DCMAKE_INSTALL_PREFIX=${PWD}/fastdeploy-0.0.3
make -j8
make install
```
编译后的预测库即在当前目录下的`fastdeploy-0.0.3`

## 编译Python安装包
```
git clone https://gitee.com/jiangjiajun/FastDeploy.git
cd FastDeploy
git submodule init
git submodule update
# Python通过export环境变量设置编译选项
export ENABLE_ORT_BACKEND=ON
export ENABLE_VISION=ON
python setup.py build
python setup.py bdist_wheel
```
编译后的wheel包即在当前目录下的`dist`目录中

编译选项说明参考[编译指南](./README.md)
