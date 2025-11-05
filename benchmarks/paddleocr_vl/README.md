## FastDeploy 服务化性能压测工具（PaddleOCR-VL）

### 数据集：

下载到本地用于性能测试：

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

1. 安装依赖：

    ```shell
    python -m pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
    pip install -U paddlex
    ```

2. 启动测试脚本：

    ```shell
    python benchmark.py ./OmniDocBenchv1 -b 512 --paddlex_config_path ./PaddleOCR-VL.yaml --gpu_ids 0
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
                <td>输入的目录路径。可以提供一个或多个。</td>
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
    Throughput (file): 1.3477 files per second
    Average latency (batch): 363.7301 seconds
    Processed pages: 981
    Throughput (page): 1.3477 pages per second
    Generated tokens: 1507157
    Throughput (token): 2070.5 tokens per second
    GPU utilization (%): 100.0, 0.0, 69.2
    GPU memory usage (MB): 81500.8, 58808.8, 77409.0
    Config and results saved to benchmark.json
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
            <td>每秒处理的文件数量/td>
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
