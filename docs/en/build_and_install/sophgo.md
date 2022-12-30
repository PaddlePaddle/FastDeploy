
# How to Build SOPHGO Deployment Environment

## How to Build and Install C++ SDK
SOPHGO supports compilation on linux, using Debian/Ubuntu as an example  
The installation package consists of three files
- [sophon-driver\_0.4.2\_$arch.deb](http://219.142.246.77:65000/sharing/KWqbmEcKp)
- [sophon-libsophon\_0.4.2\_$arch.deb](http://219.142.246.77:65000/sharing/PlvlBXhWY)
- [sophon-libsophon-dev\_0.4.2\_$arch.deb](http://219.142.246.77:65000/sharing/zTErLlpS7)

$arch indicates the hardware architecture of the current machine. Run the following command to obtain the current server arch:
```shell
uname -m
```
Generally, the hardware architecture of x86_64 machines is amd64, so the hardware architecture is arm64:  
```text
- sophon-driver_0.4.2_$arch.deb
- sophon-libsophon_0.4.2_$arch.deb
- sophon-libsophon-dev_0.4.2_$arch.deb  
```

sophon-driver contains PCIe acceleration card drivers; sophon-libsophon contains the runtime environment (librarys, tools, etc); sophon-libsophon-dev contains the development environment (header files, etc.). If you install packages only on a deployment environment, you do not need to install sophon-libsophon-dev.
You can perform the following steps to install:
```shell
#To install a dependency library, you only need to do this once:
sudo apt install dkms libncurses5
#install libsophon:
sudo dpkg -i sophon-*.deb
#Run the following command on the terminal, log out and then log in the current user to use commands such as bm-smi:
source /etc/profile
```
The position of installation:：
```text
/opt/sophon/
├── driver-0.4.2
├── libsophon-0.4.2
|    ├──bin
|    ├──data
|    ├──include
|    └──lib
└── libsophon-current->/opt/sophon/libsophon-0.4.2
```
