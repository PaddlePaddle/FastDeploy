# Use CMakeGUI + VS 2019 IDE to Compile FastDeploy

Note: This method only supports FastDeploy C++ SDK

## Contents

- [How to Use CMake GUI for Basic Compliation](#CMakeGuiAndVS2019Basic)
- [How to Set for CPU version C++ SDK Compilation](#CMakeGuiAndVS2019CPU)
- [How to Set for GPU version C++ SDK Compilation](#CMakeGuiAndVS2019GPU)
- [How to Use Visual Studio 2019 IDE for Compliation](#CMakeGuiAndVS2019Build)

### How to Use CMake GUI for Basic Compilation
<div id="CMakeGuiAndVS2019Basic"></div>

Step 1: First, open the CMake GUI and initialize the FastDeploy project.：

![image](https://user-images.githubusercontent.com/31974251/192094881-c5beb0e5-82ae-4a62-a88c-73f3d80f7936.png)  

Step 2: After clicking Configure, set the compile "x64" architecture in the pop-up window.

![image](https://user-images.githubusercontent.com/31974251/192094951-958a0a22-2090-4ab6-84f5-3573164d0835.png)

Once initialization is completed, it is shown as follows.  ：  

![image](https://user-images.githubusercontent.com/31974251/192095053-874b9c73-fc0d-4325-b555-ac94ab9a9f38.png)

Step 3: As FastDeploy currently only supports Release version, please change "CMAKE_CONFIGURATION_TYPES" to "Release" first.

![image](https://user-images.githubusercontent.com/31974251/192095175-3aeede95-a633-4b3c-81f8-067f0a0a44a3.png)

Developers can customize compilation options and generate sln solutions according to their needs. We offer two examples for compiling the CPU and GPU versions of the SDK.

### How to Set for CPU version C++ SDK Compilation

<div id="CMakeGuiAndVS2019CPU"></div>  

Step 1: Select the compilation option according to the CPU version. Please `do not` select WITH_GPU and ENABLE_TRT_BACKEND

![image](https://user-images.githubusercontent.com/31974251/192095848-b3cfdf19-e378-41e0-b44e-5edb49461eeb.png)

In this example, we enable ORT, Paddle, OpenVINO and other inference backends, and select the APIs that need to compile TEXT and VISION.


Step 2: Customize the SDK installation path and modify CMAKE_INSTALL_PREFIX

![image](https://user-images.githubusercontent.com/31974251/192095961-5f6e348a-c30b-4473-8331-8beefb7cd2e6.png)

As the default installation path is C drive, we can modify CMAKE_INSTALL_PREFIX to specify our own installation path. Here we modify the installation path to the `build\fastdeploy-win-x64-0.2.1` directory. 

![image](https://user-images.githubusercontent.com/31974251/192096055-8a276a9e-6017-4447-9ded-b95c5579d663.png)



### How to Set for GPU version C++ SDK Compilation
<div id="CMakeGuiAndVS2019GPU"></div>  

Step 1: Select the compilation option according to the CPU version. Please `do` select WITH_GPU

![image](https://user-images.githubusercontent.com/31974251/192099254-9f82abb0-8a29-41ce-a0ce-da6aacf23582.png)

In this example, we enable ORT, Paddle, OpenVINO and TRT inference backends, and select the APIs that need to compile TEXT and VISION. As we enabled GPU and TensorRT, we need to specify CUDA_DIRECTORY and TRT_DIRECTOR. Find these two variables in the GUI interface, select the options box on the right, and select the path where you installed CUDA and TensorRT respectively.


![image](https://user-images.githubusercontent.com/31974251/192098907-9dd9a49c-4a3e-4641-8e68-f25da1cafbba.png)


![image](https://user-images.githubusercontent.com/31974251/192098984-7fefd824-7e3b-4185-abba-bae5d8765e2a.png)


Step 2: Customize the SDK installation path and modify CMAKE_INSTALL_PREFIX

![image](https://user-images.githubusercontent.com/31974251/192099125-81fc8217-e51f-4039-9421-ba7a09c0027c.png)


As the default installation path is C drive, we can modify CMAKE_INSTALL_PREFIX to specify our own installation path. Here we modify the installation path to `build\fastdeploy-win-x64-gpu-0.2.1` directory. 


### How to Use Visual Studio 2019 IDE for Compliation

<div id="CMakeGuiAndVS2019Build"></div>  

Step 1: Click "Generate" to generate the sln solution and open it with Visual Studio 2019

![image](https://user-images.githubusercontent.com/31974251/192096162-c05cbb11-f96e-4c82-afde-c7fc02cddf68.png)

During this process, the model will download some resources needed for compilation by default. Developers can ignore the dev warning of cmake. After the generation is completed, the following interface will be on display.

CPU version SDK: 

![image](https://user-images.githubusercontent.com/31974251/192096478-faa570bd-7569-43c3-ad79-cc6be5b605e3.png)

GPU Version SDK: 

![image](https://user-images.githubusercontent.com/31974251/192099583-300e4680-1089-45cf-afaa-d2afda8fd436.png)

On the left, developers can see all the include paths and lib paths needed for compilation have been set up. Developers can record these paths for later development. On the right, Developers can see that the generated fastdeploy.sln solution file. Please open this solution file with Visual Studio 2019 (VS2022 can also be compiled, but VS2019 is recommended for now).

![image](https://user-images.githubusercontent.com/31974251/192096765-2aeadd68-47fb-4cd6-b083-4a478cf5e584.jpg)


Step 2: Click "ALL BUILD" in Visual Studio 2019 -> right click "Generate" to start compiling

![image](https://user-images.githubusercontent.com/31974251/192096893-5d6bc428-b824-4ffe-8930-0ec6d4dcfd02.png)  

CPU version SDK compiled successfully!

![image](https://user-images.githubusercontent.com/31974251/192097020-979bd7a3-1cdd-4fb5-a931-864c5372933d.png)

GPU version SDK compiled successfully! 

![image](https://user-images.githubusercontent.com/31974251/192099902-4b661f9a-7691-4f7f-b573-92ca9397a890.png)


Step 3: After compiling, click "INSTALL" -> right click "Generate" in Visual Studio 2019 to install the compiled SDK to the previously specified directory


![image](https://user-images.githubusercontent.com/31974251/192097073-ce5236eb-1ed7-439f-8098-fef7a2d02779.png)

![image](https://user-images.githubusercontent.com/31974251/192097122-d675ae39-35fb-4dbb-9c75-eefb0597ec2e.png)  

SDK successfully installed to the specified directory!  

### Compile all examples（Optional）

Developers can select the BUILD_EXAMPLES option in the CMake GUI to compile all the examples together. All the executable files of the examples will be saved in the build/bin/Release directory after the compilation is finished.

![image](https://user-images.githubusercontent.com/31974251/192110769-a4f0940d-dea3-4524-831b-1c2a6ab8e871.png)

![image](https://user-images.githubusercontent.com/31974251/192110930-e7e49bc6-c271-4076-be74-3d103f27bc78.png)


## Note

For self compilation of the SDK, we support Windows 10/11, VS 2019/2022, CUDA 11.x and TensorRT 8.x configurations. And we recommend the default configuration of Windows 10, VS 2019, CUDA 11.2, and TensorRT 8.4.x versions.

Moreover, if there is problem with encoding Chinese characters during compilation (e.g. UIE example must input Chinese characters for inference), please refer to the official Visual Studio tutorials and set the source character set to `/utf-8` to solve this problem..

- [/utf-8（Set the source character set and execution character set to UTF-8）](https://learn.microsoft.com/zh-cn/cpp/build/reference/utf-8-set-source-and-executable-character-sets-to-utf-8?view=msvc-170)
