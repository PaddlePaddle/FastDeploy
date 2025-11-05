## FastDeploy 服务化性能压测工具（PaddleOCR-VL）

本文档主要介绍如何对 [PaddleOCR-VL](https://www.paddleocr.ai/latest/version3.x/pipeline_usage/PaddleOCR-VL.html) 进行性能测试。

### 数据集：

下载数据集到本地用于性能测试：

<table>
  <thead>
    <tr>
      <th>数据集</th>
      <th>获取地址</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>OmniDocBench v1 数据集，共 981 个 pdf 文件</td>
      <td><code>https://github.com/opendatalab/OmniDocBench</code></td>
    </tr>
  </tbody>
</table>

### 使用方式

1. 启动 FastDeploy 服务，下面为 A100-80G 测试时使用的参数，可以根据实际情况进行调整：

    ```shell
    python -m fastdeploy.entrypoints.openai.api_server \
            --model PaddlePaddle/PaddleOCR-VL \
            --port 8118 \
            --metrics-port 8471 \
            --engine-worker-queue-port 8472 \
            --cache-queue-port 55660 \
            --max-model-len 16384 \
            --max-num-batched-tokens 16384 \
            --gpu-memory-utilization 0.7 \
            --max-num-seqs 256 \
            --workers 2 \
            --graph-optimization-config '{"graph_opt_level":0, "use_cudagraph":true}'
    ```

2. 在同一环境安装依赖后启动测试脚本：

    ```shell
    # 安装依赖
    pip install -U paddlex
    # 启动测试脚本
    python benchmark.py ./test_data -b 512 -o ./benchmark.json --paddlex_config_path ./PaddleOCR-VL.yaml --gpu_ids 0
    ```

    测试脚本参数说明：

    <table>
        <thead>
            <tr>
                <th>参数</th>
                <th>说明</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><code>input_dirs</code></td>
                <td>输入的目录路径，会自动识别到目录下的 pdf 或图片。可以提供一个或多个。</td>
            </tr>
            <tr>
                <td><code>-b, --batch_size</code></td>
                <td>推理时使用的批处理大小。</td>
            </tr>
            <tr>
                <td><code>-o, --output_path</code></td>
                <td>输出结果文件的路径。</td>
            </tr>
            <tr>
                <td><code>--paddlex_config_path</code></td>
                <td>PaddleX 的 YAML 配置文件路径。</td>
            </tr>
            <tr>
                <td><code>--gpu_ids</code></td>
                <td>指定要使用的 GPU 设备 ID，可提供一个或多个。</td>
            </tr>
        </tbody>
    </table>

3. 测试结束后，会输出类似于下面的结果：

    ```text
    Throughput (file): 1.3961 files per second
    Average latency (batch): 351.0812 seconds
    Processed pages: 981
    Throughput (page): 1.3961 pages per second
    Generated tokens: 1510337
    Throughput (token): 2149.5 tokens per second
    GPU utilization (%): 100.0, 0.0, 68.1
    GPU memory usage (MB): 77664.8, 58802.8, 74402.7
    ```

    输出结果说明：

    <table>
        <thead>
            <tr>
                <th>参数</th>
                <th>说明</th>
            </tr>
        </thead>
        <tr>
            <td>Throughput (file)</td>
            <td>每秒处理的文件数量</td>
        </tr>
        <tr>
            <td>Average latency (batch)</td>
            <td>每批次处理的平均延迟时间，单位为秒</td>
        </tr>
        <tr>
            <td>Processed pages</td>
            <td>已处理的页面总数</td>
        </tr>
        <tr>
            <td>Throughput (page)</td>
            <td>每秒处理的页面数量</td>
        </tr>
        <tr>
            <td>Generated tokens</td>
            <td>生成的token总数</td>
        </tr>
        <tr>
            <td>Throughput (token)</td>
            <td>每秒生成的token数量</td>
        </tr>
        <tr>
            <td>GPU utilization (%)</td>
            <td>GPU 的最大、最小、平均利用率</td>
        </tr>
        <tr>
            <td>GPU memory usage (MB)</td>
            <td>GPU 的最大、最小、平均显存占用，单位为 MB</td>
        </tr>
    </table>
