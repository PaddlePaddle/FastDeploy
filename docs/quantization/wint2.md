# WINT2 Quantization

Weights are compressed offline using the CCQ (Convolutional Coding Quantization) method. The actual stored numerical type of weights is INT8, with 4 weights packed into each INT8 value, equivalent to 2 bits per weight. Activations are not quantized. During inference, weights are dequantized and decoded in real-time to BF16 numerical type, and calculations are performed using BF16 numerical type.
- **Supported Hardware**: GPU
- **Supported Architecture**: MoE architecture

CCQ WINT2 is generally used in resource-constrained and low-threshold scenarios. Taking ERNIE-4.5-300B-A47B as an example, weights are compressed to 89GB, supporting single-card deployment on 141GB H20.

## Run WINT2 Inference Service

```
python -m fastdeploy.entrypoints.openai.api_server \
       --model baidu/ERNIE-4.5-300B-A47B-2Bits-Paddle \
       --port 8180 --engine-worker-queue-port 8181 \
       --cache-queue-port 8182 --metrics-port 8182 \
       --tensor-parallel-size 1 \
       --max-model-len 32768 \
       --max-num-seqs 32
```

By specifying `--model baidu/ERNIE-4.5-300B-A47B-2Bits-Paddle`, the offline quantized WINT2 model can be automatically downloaded from AIStudio. In the config.json file of this model, there will be WINT2 quantization-related configuration information, so there's no need to set `--quantization` when starting the inference service.

Example of quantization configuration in the model's config.json file:

```
"quantization_config": {
    "dense_quant_type": "wint8",
    "moe_quant_type": "w4w2",
    "quantization": "wint2",
    "moe_quant_config": {
    "moe_w4_quant_config": {
        "quant_type": "wint4",
        "quant_granularity": "per_channel",
        "quant_start_layer": 0,
        "quant_end_layer": 6
    },
    "moe_w2_quant_config": {
        "quant_type": "wint2",
        "quant_granularity": "pp_acc",
        "quant_group_size": 64,
        "quant_start_layer": 7,
        "quant_end_layer": 53
    }
  }
}
```

- For more deployment tutorials, please refer to [get_started](../get_started/ernie-4.5.md);
- For more model descriptions, please refer to [Supported Model List](https://console.cloud.baidu-int.com/devops/icode/repos/baidu/paddle_internal/FastDeploy/blob/feature%2Finference-refactor-20250528/docs/supported_models.md).

## WINT2 Performance

On the ERNIE-4.5-300B-A47B model, comparison of WINT2 vs WINT4 performance:

| Test Set | Dataset Size | WINT4 | WINT2 |
|---------|---------|---------|---------|
| IFEval |500|88.17 | 85.40 |
|BBH|6511|94.43|92.02|
|DROP|9536|91.17|89.97|
