# fastdeploy_init.bat工具使用方式

<div id="CommandLineDeps"></div>  

## 1 方式一：使用 fastdeploy_init.bat 进行配置（推荐）  
<div id="CommandLineDeps1"></div>  

对于版本高于0.2.1的SDK，我们提供了 **fastdeploy_init.bat** 工具来管理FastDeploy中所有的依赖库。可以通过该脚本工具查看(show)、拷贝(install) 和 设置(init and setup) SDK中所有的dll，方便用户快速完成运行时环境配置。

### 1.1 fastdeploy_init.bat 使用说明  
<div id="CommandLineDeps11"></div>  

首先进入SDK的根目录，运行以下命令，可以查看 fastdeploy_init.bat 的用法说明
```bat
D:\path-to-your-fastdeploy-sdk-dir>fastdeploy_init.bat help
------------------------------------------------------------------------------------------------------------------------------------------------------------
[1] [help]    print help information:                      fastdeploy_init.bat help
[2] [show]    show all dlls/libs/include paths:            fastdeploy_init.bat show fastdeploy-sdk-dir
[3] [init]    init all dlls paths for current terminal:    fastdeploy_init.bat init fastdeploy-sdk-dir  [WARNING: need copy onnxruntime.dll manually]
[4] [setup]   setup path env for current terminal:         fastdeploy_init.bat setup fastdeploy-sdk-dir [WARNING: need copy onnxruntime.dll manually]
[5] [install] install all dlls to a specific dir:          fastdeploy_init.bat install fastdeploy-sdk-dir another-dir-to-install-dlls **[RECOMMEND]**
[6] [install] install all dlls with logging infos:         fastdeploy_init.bat install fastdeploy-sdk-dir another-dir-to-install-dlls info
------------------------------------------------------------------------------------------------------------------------------------------------------------
```  
用法简要说明如下：  
- help:     打印所有的用法说明  
- show:     查看SDK中所有的 dll、lib 和 include 路径
- init:     初始化所有dll路径信息，后续用于设置terminal环境变量（不推荐，请参考4.3中关于onnxruntime的说明）
- setup:    在init之后运行，设置terminal环境便令（不推荐，请参考4.3中关于onnxruntime的说明）  
- install:  将SDK中所有的dll安装到某个指定的目录（推荐）

### 1.2  fastdeploy_init.bat 查看 SDK 中所有的 dll、lib 和 include 路径  
<div id="CommandLineDeps12"></div>  

进入SDK的根目录，运行show命令，可以查看SDK中所有的 dll、lib 和 include 路径。以下命令中 %cd% 表示当前目录（SDK的根目录）。  
```bat
D:\path-to-fastdeploy-sdk-dir>fastdeploy_init.bat show %cd%
------------------------------------------------------------------------------------------------------------------------------------------------------------
[SDK] D:\path-to-fastdeploy-sdk-dir
------------------------------------------------------------------------------------------------------------------------------------------------------------
[DLL] D:\path-to-fastdeploy-sdk-dir\lib\fastdeploy.dll **[NEEDED]**
[DLL] D:\path-to-fastdeploy-sdk-dir\third_libs\install\faster_tokenizer\lib\core_tokenizers.dll  **[NEEDED]**
[DLL] D:\path-to-fastdeploy-sdk-dir\third_libs\install\opencv\build\x64\vc15\bin\opencv_ffmpeg3416_64.dll  **[NEEDED]**
......
------------------------------------------------------------------------------------------------------------------------------------------------------------
[Lib] D:\path-to-fastdeploy-sdk-dir\lib\fastdeploy.lib **[NEEDED][fastdeploy]**
[Lib] D:\path-to-fastdeploy-sdk-dir\third_libs\install\faster_tokenizer\lib\core_tokenizers.lib  **[NEEDED][fastdeploy::text]**
[Lib] D:\path-to-fastdeploy-sdk-dir\third_libs\install\opencv\build\x64\vc15\lib\opencv_world3416.lib  **[NEEDED][fastdeploy::vision]**
......
------------------------------------------------------------------------------------------------------------------------------------------------------------
[Include] D:\path-to-fastdeploy-sdk-dir\include **[NEEDED][fastdeploy]**
[Include] D:\path-to-fastdeploy-sdk-dir\third_libs\install\faster_tokenizer\include  **[NEEDED][fastdeploy::text]**
[Include] D:\path-to-fastdeploy-sdk-dir\third_libs\install\opencv\build\include  **[NEEDED][fastdeploy::vision]**
......
------------------------------------------------------------------------------------------------------------------------------------------------------------
[XML] D:\path-to-fastdeploy-sdk-dir\third_libs\install\openvino\runtime\bin\plugins.xml **[NEEDED]**
------------------------------------------------------------------------------------------------------------------------------------------------------------
```  
可以看到该命令会根据您当前的SDK，输出对应的信息，包含 dll、lib 和 include 的路径信息。对于 dll，被标记为 `[NEEDED]`的，是运行时所需要的，如果包含OpenVINO后端，还需要将他的plugins.xml拷贝到exe所在的目录；对于 lib 和 include，被标记为`[NEEDED]`的，是开发时所需要配置的最小依赖。并且，我们还增加了对应的API Tag标记，如果您只使用vision API，则只需要配置标记为 `[NEEDED][fastdeploy::vision]` 的 lib 和 include 路径.  

### 1.3 fastdeploy_init.bat 安装 SDK 中所有的 dll 到指定的目录 （推荐）
<div id="CommandLineDeps13"></div>  

进入SDK的根目录，运行install命令，可以将SDK 中所有的 dll 安装到指定的目录（如exe所在的目录）。我们推荐这种方式来配置exe运行所需要的依赖库。比如，可以在SDK根目录下创建一个临时的bin目录备份所有的dll文件。以下命令中 %cd% 表示当前目录（SDK的根目录）。
```bat  
% info参数为可选参数，添加info参数后会打印详细的安装信息 %
D:\path-to-fastdeploy-sdk-dir>fastdeploy_init.bat install %cd% bin
D:\path-to-fastdeploy-sdk-dir>fastdeploy_init.bat install %cd% bin info
```
```bat
D:\path-to-fastdeploy-sdk-dir>fastdeploy_init.bat install %cd% bin
[INFO] Do you want to install all FastDeploy dlls ?
[INFO] From: D:\path-to-fastdeploy-sdk-dir
[INFO]   To: bin
Choose y means YES, n means NO: [y/n]y
YES.
请按任意键继续. . .
[INFO] Created bin done!
已复制         1 个文件。
已复制         1 个文件。
已复制         1 个文件。
已复制         1 个文件。
.....
已复制         1 个文件。
已复制         1 个文件。
已复制         1 个文件。
已复制         1 个文件。
.....
```  
### 1.4 fastdeploy_init.bat 配置 SDK 环境变量  
<div id="CommandLineDeps14"></div>  

您也可以选择通过配置环境变量的方式来设置运行时的依赖库环境，这种方式只在当前的terminal有效。如果您使用的SDK中包含了onnxruntime推理后端，我们不推荐这种方式，详细原因请参考（4.3）中关于onnxruntime配置的说明（需要手动拷贝onnxruntime所有的dll到exe所在的目录）。配置 SDK 环境变量的方式如下。以下命令中 %cd% 表示当前目录（SDK的根目录）。
```bat
% 先运行 init 初始化当前SDK所有的dll文件路径 %
D:\path-to-fastdeploy-sdk-dir>fastdeploy_init.bat init %cd%
% 再运行 setup 完成 SDK 环境变量配置  %
D:\path-to-fastdeploy-sdk-dir>fastdeploy_init.bat setup %cd%
```

## 2 方式二：修改CMakeLists.txt，一行命令配置（推荐）
<div id="CommandLineDeps2"></div>  

考虑到Windows下C++开发的特殊性，如经常需要拷贝所有的lib或dll文件到某个指定的目录，FastDeploy提供了`install_fastdeploy_libraries`的cmake函数，方便用户快速配置所有的dll。修改ppyoloe的CMakeLists.txt，添加：  
```cmake
install_fastdeploy_libraries(${CMAKE_CURRENT_BINARY_DIR}/Release)
```
