# WINT2量化

权重经过CCQ（Convolutional Coding Quantization）方法离线压缩。权重实际存储的数值类型是INT8，每个INT8数值中打包了4个权重，等价于每个权重2bits. 激活不做量化，计算时将权重实时地反量化、解码为BF16数值类型，并用BF16数值类型计算。
- **支持硬件**：GPU
- **支持结构**：MoE结构

CCQ WINT2一般用于资源受限的低门槛场景，以ERNIE-4.5-300B-A47B为例，将权重压缩到89GB，可支持141GB H20单卡部署。

## 启动WINT2推理服务

```
python -m fastdeploy.entrypoints.openai.api_server \
       --model baidu/ERNIE-4.5-300B-A47B-2Bits-Paddle \
       --port 8180 --engine-worker-queue-port 8181 \
       --cache-queue-port 8182 --metrics-port 8182 \
       --tensor-parallel-size 1 \
       --max-model-len 32768 \
       --max-num-seqs 32
```

通过指定 `--model baidu/ERNIE-4.5-300B-A47B-2Bits-Paddle` 可自动从AIStudio下载已离线量化好的WINT2模型，在该模型的config.json文件中，会有WINT2量化相关的配置信息，不用再在启动推理服务时设置 `--quantization`.

模型的config.json文件中的量化配置示例如下：

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

- 更多部署教程请参考[get_started](../get_started/ernie-4.5.md)；
- 更多模型说明请参考[支持模型列表](../supported_models.md)。

## WINT2效果

在ERNIE-4.5-300B-A47B模型上，WINT2与WINT4效果对比：

| 测试集 |数据集大小| WINT4 | WINT2 |
|---------|---------|---------|---------|
| IFEval |500|88.17 | 85.40 |
|BBH|6511|94.43|92.02|
|DROP|9536|91.17|89.97|

## WINT2推理性能
