# Supported Models

FastDeploy currently supports the following models, which can be downloaded via three methods:

- 1. During FastDeploy deployment, specify the ```model``` parameter as the model name in the table below to automatically download model weights from AIStudio (supports resumable downloads)
- 2. Download Paddle-version ERNIE models from [HuggingFace/baidu/models](https://huggingface.co/baidu/models), e.g., `baidu/ERNIE-4.5-0.3B-Paddle`
- 3. Search for corresponding Paddle-version ERNIE models on [ModelScope/PaddlePaddle](https://www.modelscope.cn/models?name=PaddlePaddle&page=1&tabKey=task), e.g., `ERNIE-4.5-0.3B-Paddle`

For the first method (auto-download), the default download path is ```~/``` (user home directory). Users can modify this path by setting the ```FD_MODEL_CACHE``` environment variable, e.g.:
```bash
export FD_MODEL_CACHE=/ssd1/download_models
```

| Model Name | Context Length | Quantization | Minimum Deployment Resources | Notes |
| :--------- | :------------- | :----------- | :-------------------------- | :---- |
| baidu/ERNIE-4.5-VL-424B-A47B-Paddle | 32K/128K | WINT2 | 1*96G GPU VRAM/1T RAM | Chunked Prefill required for 128K |
| baidu/ERNIE-4.5-VL-424B-A47B-Paddle | 32K/128K | WINT4 | 4*80G GPU VRAM/1T RAM | Chunked Prefill required for 128K |
| baidu/ERNIE-4.5-VL-424B-A47B-Paddle | 32K/128K | WINT8 | 8*80G GPU VRAM/1T RAM | Chunked Prefill required for 128K |
| baidu/ERNIE-4.5-300B-A47B-Paddle | 32K/128K | WINT4 | 4*64G GPU VRAM/600G RAM | Chunked Prefill required for 128K |
| baidu/ERNIE-4.5-300B-A47B-Paddle | 32K/128K | WINT8 | 8*64G GPU VRAM/600G RAM | Chunked Prefill required for 128K |
| baidu/ERNIE-4.5-300B-A47B-W4A8C8-TP4-Paddle | 32K/128K | W4A8C8 | 4*64G GPU VRAM/160G RAM | Fixed 4-GPU setup, Chunked Prefill recommended |
| baidu/ERNIE-4.5-300B-A47B-FP8-Paddle | 32K/128K | FP8 | 8*64G GPU VRAM/600G RAM | Chunked Prefill recommended, only supports PD Disaggragated Deployment with EP parallelism |
| baidu/ERNIE-4.5-300B-A47B-Base-Paddle | 32K/128K | WINT4 | 4*64G GPU VRAM/600G RAM | Chunked Prefill recommended |
| baidu/ERNIE-4.5-300B-A47B-Base-Paddle | 32K/128K | WINT8 | 8*64G GPU VRAM/600G RAM | Chunked Prefill recommended |
| baidu/ERNIE-4.5-VL-28B-A3B-Paddle | 32K | WINT4 | 1*24G GPU VRAM/128G RAM | Chunked Prefill required |
| baidu/ERNIE-4.5-VL-28B-A3B-Paddle | 128K | WINT4 | 1*48G GPU VRAM/128G RAM | Chunked Prefill required |
| baidu/ERNIE-4.5-VL-28B-A3B-Paddle | 32K/128K | WINT8 | 1*48G GPU VRAM/128G RAM | Chunked Prefill required |
| baidu/ERNIE-4.5-21B-A3B-Paddle | 32K/128K | WINT4 | 1*24G GPU VRAM/128G RAM | Chunked Prefill required for 128K |
| baidu/ERNIE-4.5-21B-A3B-Paddle | 32K/128K | WINT8 | 1*48G GPU VRAM/128G RAM | Chunked Prefill required for 128K |
| baidu/ERNIE-4.5-21B-A3B-Base-Paddle | 32K/128K | WINT4 | 1*24G GPU VRAM/128G RAM | Chunked Prefill required for 128K |
| baidu/ERNIE-4.5-21B-A3B-Base-Paddle | 32K/128K | WINT8 | 1*48G GPU VRAM/128G RAM | Chunked Prefill required for 128K |
| baidu/ERNIE-4.5-0.3B-Paddle | 32K/128K | BF16 | 1*16G GPU VRAM/2G RAM | |
| baidu/ERNIE-4.5-0.3B-Base-Paddle | 32K/128K | BF16 | 1*16G GPU VRAM/2G RAM | |

More models are being supported. You can submit requests for new model support via [Github Issues](https://github.com/PaddlePaddle/FastDeploy/issues).
