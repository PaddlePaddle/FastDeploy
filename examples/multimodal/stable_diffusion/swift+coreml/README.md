# swift+coreml stable-diffusion pipeline
这里演示如何在苹果芯片上部署从paddle转换的stable-diffusion demo。

包括swift stable-diffusion pipeline，在ml-stable-diffusion repo中。另包含两个示例。一个是最简单的示例，只支持Macbook，另一个是开源的示例，支持Macbook和iPhone。

## 1. 准备模型
参考mll-stable-diffusion的README.md，将模型转换为coreml格式。也可以直接下载我们已经转换好的[模型](https://ecloud.baidu.com?t=e2808cf7d527f679334538439571f969)


## 2. 运行示例

### 2.1 最简示例（仅Macbook）

**Step 1:**
打开swift_simplest_demo中的xcodeproj工程。

**Step 2:**
将ml-stable-diffusion拖入。

**Step 3:**
在代码中指定模型路径即可运行。

### 2.2 开源示例（Macbook + iPhone）

**Step 1:**
打开swift_coreml_diffusers中的xcodeproj工程。

**Step 2:**
将ml-stable-diffusion拖入, 替换掉原有的stable-diffusion package。

**Step 3:**
启动一个简单的服务，便于iPhone用于下载我们之前准备的模型。

**Step 4:**
修改ModelInfo.swift中的URL为我们开启的服务。这样iPhone就可以下载模型了，并且加载运行。
