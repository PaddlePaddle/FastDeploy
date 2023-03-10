# Build and Install

Make sure the environment satisfy
- cmake >= 3.18.0
- protobuf >= 3.16.0

## Install on Linux/Mac

### Install Protobuf
```
git clone https://github.com/protocolbuffers/protobuf.git
cd protobuf
git checkout v3.16.0
mkdir build_source && cd build_source
cmake ../cmake -DCMAKE_INSTALL_PREFIX=${PWD}/installed_protobuf_lib -Dprotobuf_BUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON -Dprotobuf_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release
make -j8
make install

# Add the library to environment
export PATH=${PWD}/installed_protobuf_lib:${PATH}
```
### Install Paddle2ONNX
```
git clone https://github.com/PaddlePaddle/Paddle2ONNX.git
cd Paddle2ONNX
git submodule init
git submodule update

python setup.py install
```

## Install on Windows

Make sure you have installed Visual Studio 2019 first

### Open VS command tool
Find **x64 Native Tools Command Prompt for VS 2019** in system menu, and open it

### Install Protobuf

Notice: Please change the `-DCMAKE_INSTALL_PREFIX` to your custom path

```
git clone https://github.com/protocolbuffers/protobuf.git
cd protobuf
git checkout v3.16.0
cd cmake
cmake -G "Visual Studio 16 2019" -A x64 -DCMAKE_INSTALL_PREFIX=D:\Paddle\installed_protobuf_lib -Dprotobuf_MSVC_STATIC_RUNTIME=OFF -Dprotobuf_BUILD_SHARED_LIBS=OFF -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_BUILD_EXAMPLES=OFF .
msbuild protobuf.sln /m /p:Configuration=Release /p:Platform=x64
msbuild INSTALL.vcxproj /p:Configuration=Release /p:Platform=x64

# Add the library to environment
set PATH=D:\Paddle\installed_protobuf_lib\bin;%PATH%
```

### Install Paddle2ONNX
```
git clone https://github.com/PaddlePaddle/Paddle2ONNX.git
cd Paddle2ONNX
git submodule init
git submodule update

python setup.py install

```
