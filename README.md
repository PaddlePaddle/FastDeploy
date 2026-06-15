# FastDeploy with SQAttn Integration

This repository is a fork of [PaddlePaddle/FastDeploy](https://github.com/PaddlePaddle/FastDeploy), extended with SQAttn support for efficient long-context Transformer inference.

FastDeploy is a high-performance deployment framework for large language models and vision-language models based on PaddlePaddle. This fork integrates SQAttn into the FastDeploy inference stack, enabling sparse-quantized attention computation for ultra-long-context scenarios.

## Main Changes in This Fork

Compared with the upstream FastDeploy repository, this fork mainly includes the following modifications:

- Added SQAttn support for long-context Transformer inference.
- Integrated SQAttn-related attention components into the FastDeploy execution workflow.
- Adapted the inference path to support sparse-quantized attention computation.
- Cleaned unused or intermediate files related to the integration.
- Applied minor fixes and adjustments for compatibility and usability.
