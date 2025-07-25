### FastDeploy服务化性能压测工具

#### 数据集：

wget下载到本地用于性能测试

<table style="width:100%; border-collapse: collapse;">
  <thead>
    <tr>
      <th style="width:15%; text-align: left;">Dataset</th>
      <th style="width:65%; text-align: left;">Data Path</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>开源数据集 2k条</strong></td>
      <td><code>https://fastdeploy.bj.bcebos.com/eb_query/filtered_sharedgpt_2000_input_1136_output_200_fd.json</code></td>
    </tr>
  </tbody>
</table>

#### 使用方式：

```
# 安装依赖
python -m pip install -r requirements.txt
```

##### 参数说明

```bash
--backend openai-chat：压测使用的后端接口，指定为"openai-chat"使用chat/completion接口
--model EB45T：模型名，任意取名，影响最后保存的结果文件名 EB45T \
--endpoint /v1/chat/completions：endpoint，用于组url
--host 0.0.0.0：服务ip地址，用于组url
--port 9812：服务HTTP端口，用于组url
--dataset-name EBChat：指定数据集类，指定为"EBChat"可读取转存的FD格式数据集
--dataset-path ./eb45t_spv4_dataserver_1w_waigua_fd：压测数据集路径
--hyperparameter-path EB45T.yaml：(可选)超参文件，请求时会更新进payload中，默认不带任何超参
--percentile-metrics ttft,tpot,itl,e2el,s_ttft,s_itl,s_e2el,s_decode,input_len,s_input_len,output_len：性能结果中展示的指标集合
--metric-percentiles 80,95,99,99.9,99.95,99.99：性能结果中展示的性能指标分位值
--num-prompts 1：总计发送多少条请求
--max-concurrency 1：压测并发数
--save-result：开启结果保存，结果文件会存入json，默认False不保存
--debug：开启debug模式，逐条打印payload和output内容，默认False
--shuffle：是否打乱数据集，默认False不打乱
--seed：打乱数据集时的随机种子，默认0
```

##### /v1/chat/completions接口压测单条数据调试

```
python benchmark_serving.py \
  --backend openai-chat \
  --model EB45T \
  --endpoint /v1/chat/completions \
  --host 0.0.0.0 \
  --port 9812 \
  --dataset-name EBChat \
  --dataset-path ./filtered_sharedgpt_2000_input_1136_output_200_fd.json \
  --hyperparameter-path yaml/request_yaml/eb45t-32k.yaml \
  --percentile-metrics ttft,tpot,itl,e2el,s_ttft,s_itl,s_e2el,s_decode,input_len,s_input_len,output_len \
  --metric-percentiles 80,95,99,99.9,99.95,99.99 \
  --num-prompts 1 \
  --max-concurrency 1 \
  --save-result
```

##### /v1/chat/completions接口完整100并发 2000条压测

```
# 保存infer_log.txt
python benchmark_serving.py \
  --backend openai-chat \
  --model EB45T \
  --endpoint /v1/chat/completions \
  --host 0.0.0.0 \
  --port 9812 \
  --dataset-name EBChat \
  --dataset-path ./filtered_sharedgpt_2000_input_1136_output_200_fd.json \
  --hyperparameter-path yaml/request_yaml/eb45t-32k.yaml \
  --percentile-metrics ttft,tpot,itl,e2el,s_ttft,s_itl,s_e2el,s_decode,input_len,s_input_len,output_len \
  --metric-percentiles 80,95,99,99.9,99.95,99.99 \
  --num-prompts 2000 \
  --max-concurrency 100 \
  --save-result > infer_log.txt 2>&1 &
```

##### /v1/completions接口压测

修改endpoint为/v1/completions，backend为openai，会对/v1/completions接口进行压测

```
# 保存infer_log.txt
python benchmark_serving.py \
  --backend openai \
  --model EB45T \
  --endpoint /v1/completions \
  --host 0.0.0.0 \
  --port 9812 \
  --dataset-name EBChat \
  --dataset-path ./filtered_sharedgpt_2000_input_1136_output_200_fd.json \
  --hyperparameter-path yaml/request_yaml/eb45t-32k.yaml \
  --percentile-metrics ttft,tpot,itl,e2el,s_ttft,s_itl,s_e2el,s_decode,input_len,s_input_len,output_len \
  --metric-percentiles 80,95,99,99.9,99.95,99.99 \
  --num-prompts 2000 \
  --max-concurrency 100 \
  --save-result > infer_log.txt 2>&1 &
```

### 投机解码性能测试工具

#### 使用方式：

```bash
python benchmarks/benchmark_mtp.py \
  --host 127.0.0.1 --port 8000 \
  --max-concurrency 16 32 64 96 --num-prompts 256 \
  --acceptance-rate 0.8 --draft-token-steps 1 2 3 \
  --s_itl-base-model 15.88 22.84 16.47 16.93 \
  --dataset-name EBChat \
  --dataset-path ./filtered_sharedgpt_2000_input_1136_output_200_fd.json
```

#### 参数说明

```bash
--host：服务ip地址，用于组url
--port：服务HTTP端口，用于组url
--max-concurrency：测试并发数
--num-prompts：总计发送多少条请求
--acceptance-rate：投机解码的模拟接受率
--draft-token-steps：投机解码的步数
--s_itl-base-model：主模型的解码延迟，可由上述的性能压测工具获得，与batch-size一一对应
--dataset-name：指定数据集类，指定为"EBChat"可读取转存的FD格式数据集
--dataset-path：测试数据集路径
```
