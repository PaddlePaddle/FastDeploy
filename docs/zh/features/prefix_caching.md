[English](../../features/prefix_caching.md)

# Prefix Caching

Prefix Caching（前缀缓存）是一种优化生成式模型推理效率的技术，核心思想是通过缓存输入序列的中间计算结果（KV Cache），避免重复计算，从而加速具有相同前缀的多个请求的响应速度。

**工作原理**

前缀识别：当多个请求共享相同的输入前缀（如提示词或上下文开头部分），系统会缓存该前缀对应的中间状态（KV Cache）。

增量计算：对于后续请求，只需计算新增部分（如用户追加的输入）并复用缓存的中间结果，显著减少计算量。

## 服务化部署开启 Prefix Caching

启动服务增加以下参数 `enable-prefix-caching`，默认只开启一级缓存（GPU 缓存）。

若需要开启 CPU 缓存，需要指定参数 `swap-space`， 指定开启的 CPU Cache 的空间大小，单位为 GB。具体大小根据加载完模型机器可用内存大小设置。

> 注：ERNIE-4.5-VL 多模态模型暂不支持开启 prefix caching。

具体参数说明可参考文档[参数说明](../parameters.md)。

启动示例：

```shell
python -m fastdeploy.entrypoints.openai.api_server \
       --model "baidu/ERNIE-4.5-21B-A3B-Paddle" \
       --port 8180 --engine-worker-queue-port 8181 \
       --metrics-port 8182 \
       --cache-queue-port 8183 \
       --enable-prefix-caching \
       --swap-space 50 \
       --max-model-len 8192 \
       --max-num-seqs 32
```

## 离线推理开启Prefix Caching

FastDeploy 启动时设置 `enable_prefix_caching=True`，CPU Cache 根据机器内存选择开启 `swap_space`。

提供了测试示例 `demo/offline_prefix_caching_demo.py`。
