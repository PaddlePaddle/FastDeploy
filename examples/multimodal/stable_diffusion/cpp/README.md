English | [简体中文](README_CN.md)
# StableDiffusion C++ Deployment

Before deployment, the following two steps need to be confirmed:

- 1. Hardware and software environment meets the requirements. Please refer to [Environment requirements for FastDeploy](../../../../docs/en/build_and_install/download_prebuilt_libraries.md)
- 2. Download pre-compiled libraries and samples according to the development environment. Please refer to [FastDeploy pre-compiled libraries](../../../../docs/en/build_and_install/download_prebuilt_libraries.md)

This directory provides `*_infer.cc` to quickly complete C++ deployment examples for each task of StableDiffusion.

## Inpaint Task

The StableDiffusion Inpaint task is a task that completes the image based on the prompt text. User provides the prompt text, the original image and the mask image of the original image, and the task outputs the completed image.
