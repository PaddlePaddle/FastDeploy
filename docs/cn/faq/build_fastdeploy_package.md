中文 | [English](../../en/faq/build_fastdeploy_package.md)

# 编译FastDeploy C++安装包

FastDeploy提供了Debian安装包和RPM安装包的打包工具，用于生成FastDeploy C++ SDK的安装包。相比于Tar压缩包，安装包具有以下优势：
- 安装时，自动运行脚本来配置lib路径，不需要用户手动设置LD_LIBRARY_PATH等环境变量
- 自动管理依赖库关系和版本，自动安装依赖项

## Debian安装包

Debian安装包适用于Debian系列的Linux发行版，例如Ubuntu

```
# 设置编译选项，运行cmake和make
cmake .. -DENABLE_PADDLE_BACKEND=ON  -DENABLE_VISION=ON -DCMAKE_INSTALL_PREFIX=/opt/paddlepaddle/fastdeploy
make -j

# 运行cpack，生成.deb安装包
cpack -G DEB

# 安装.deb
dpkg -i xxx.deb
```

## RPM安装包

RPM安装包适用于RedHat系列的Linux发行版，例如CentOS

```
# 设置编译选项，运行cmake和make
cmake .. -DENABLE_PADDLE_BACKEND=ON  -DENABLE_VISION=ON -DCMAKE_INSTALL_PREFIX=/opt/paddlepaddle/fastdeploy
make -j

# 运行cpack，生成.rpm安装包
cpack -G RPM

# 安装.rpm
rpm -i xxx.rpm
```
