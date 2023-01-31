English | [中文](../../cn/faq/build_fastdeploy_package.md)

# Build FastDeploy C++ SDK Installation Package

FastDeploy provides packaging tools for Debian installation packages and RPM installation packages, which are used to generate installation packages for FastDeploy C++ SDK. Compared with the Tar compression package, the installation package has the following advantages:
- During installation, the script is automatically run to configure the lib path, so that users don't need to manually set environment variables such as LD_LIBRARY_PATH
- Automatically manage dependencies and versions, and automatically install dependencies

## Debian Package

Debian Package is for Linux distributions of the Debian family, such as Ubuntu

```
# Setup build options, run cmake and make
cmake .. -DENABLE_PADDLE_BACKEND=ON  -DENABLE_VISION=ON -DCMAKE_INSTALL_PREFIX=/opt/paddlepaddle/fastdeploy
make -j

# Run cpack to generate a .deb package
cpack -G DEB

# Install .deb package
dpkg -i xxx.deb
```

## RPM Package

RPM Package is for Linux distributions of the RedHat family, such as CentOS

```
# Setup build options, run cmake and make
cmake .. -DENABLE_PADDLE_BACKEND=ON  -DENABLE_VISION=ON -DCMAKE_INSTALL_PREFIX=/opt/paddlepaddle/fastdeploy
make -j

# Run cpack to generate a .rpm package
cpack -G RPM

# Install .rpm package
rpm -i xxx.rpm
```
