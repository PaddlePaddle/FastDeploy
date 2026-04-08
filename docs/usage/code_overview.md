[简体中文](../zh/usage/code_overview.md)

# Code Overview

Below is an overview of the FastDeploy code structure and functionality organized by directory.

- ```custom_ops```: Contains C++ operators used by FastDeploy for large model inference. Operators for different hardware are placed in corresponding subdirectories (e.g., `cpu_ops`, `gpu_ops`). The root-level `setup_*.py` files are used to compile these C++ operators.
- ```dockerfiles```: Stores Dockerfiles for building FastDeploy runtime environment images.
- ```docs```: Documentation related to the FastDeploy codebase.
- ```fastdeploy```
  - ```agent```: Scripts for launching large model services.
  - ```cache_manager```: Cache management module for large models.
  - ```engine```: Core engine classes for managing large model execution.
  - ```entrypoints```: User-facing APIs for interaction.
  - ```input```: Input processing module, including preprocessing, multimodal input handling, tokenization, etc.
  - ```model_executor```
    - ```layers```: Layer modules required for large model architecture.
    - ```model_runner```: Model inference execution module.
    - ```models```: Built-in large model classes in FastDeploy.
    - ```ops```: Python-callable operator modules compiled from `custom_ops`, organized by hardware platform.
  - ```output```: Post-processing for large model outputs.
  - ```platforms```: Platform-specific modules for underlying hardware support.
  - ```scheduler```: Request scheduling module for large models.
  - ```metrics```: Core component for collecting, managing, and exporting Prometheus metrics, tracking key runtime performance data (e.g., request latency, resource utilization, successful request counts).
  - ```splitwise```: Modules related to PD disaggregation deployment.
- ```scripts```/```tools```: Utility scripts for FastDeploy operations (e.g., compilation, unit testing, code style fixes).
- ```test```: Code for unit testing and validation.
