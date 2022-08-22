# FastDeploy FAQ 文档

## 1. 在Windows 10 配置 CUDA v11.2 环境变量  
FastDeploy Windows 10 x64 的 GPU 版本需要依赖 CUDA 11.2，在安装完 CUDA v11.2 之后，需要设置`CUDA_DIRECTORY`、`CUDA_HOME`、`CUDA_PATH`和`CUDA_ROOT`中**任意一个**环境变量，这样FastDeploy才能链接到相关的库。有两种方式设置环境变量，通过终端命令行设置以及在系统环境变量中设置。  
- 方式一: 终端命令行设置。该方式只在当前终端有效。Windows菜单打开`x64 Native Tools Command Prompt for VS 2019`命令工具，假设你需要在该终端运行类似`python infer_ppyoloe.py`的命令。
  ```bat
  % 选择以下任意一个环境变量设置即可 %
  set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2
  set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2
  set CUDA_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2
  set CUDA_DIRECTORY=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2
  ```
- 方式二: 系统环境变量设置。该方式会修改系统环境变量。设置步骤为：  
  - (1) 打开 "设置->系统->关于"
  - (2) 找到 "高级系统设置"，点击打开
  - (3) 点击右下角的 "环境变量设置"  
  - (4) 注意，在 "系统变量" 一栏右下角点击 "新建"，如果已有相关的环境变量，则只需确认路径是否正确
  - (5) 设置`CUDA_DIRECTORY`、`CUDA_HOME`、`CUDA_PATH`和`CUDA_ROOT`中**任意一个**环境变量  
  - (6) 根据以下提示来设置环境变量，并点击确认
  ```text
  变量名(N): CUDA_DIRECTORY、CUDA_HOME、CUDA_PATH和CUDA_ROOT中任意一个
  变量值(V): 类似 C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2
  ```
