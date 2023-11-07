# 环境安装

- Step 1. 安装develop版本PaddlePaddle
- Step 2. 从源码安装PaddleNLP
- Step 3. 进入源码PaddleNLP/csrc，执行`python3 setup_cuda.py install --user`安装自定义OP


## 导出模型
```
cd PaddleNLP/llm
python export_model.py \
    --model_name_or_path meta-llama/Llama-2-7b-chat \
    --output_path ./inference \
    --dtype float16
```

## 本地测试模型

```
wget https://bj.bcebos.com/paddle2onnx/third_libs/inputs_63.jsonl
mkdir res
```
测试脚本如下，预测结果将会保存在res目录下
```
import fastdeploy_llm as fdlm
import copy
config = fdlm.Config("chatglm-6b")
config.max_batch_size = 1
config.mp_num = 1
config.max_dec_len = 1024
config.max_seq_len = 1024
config.decode_strategy = "sampling"
config.stop_threshold = 2
config.disable_dynamic_batching = 1
config.max_queue_num = 512
config.is_ptuning = 0

inputs = list()
with open("inputs_63.jsonl", "r") as f:
    for line in f:
        data = eval(line.strip())
        prompt = data["src"]
        inputs.append((prompt, data))

model = fdlm.ServingModel(config)

def call_back(call_back_task, token_tuple, index, is_last_token, sender=None):
    with open("res/{}".format(call_back_task.task_id), "a+") as f:
        f.write("{}\n".format(token_tuple))

for i, ipt in enumerate(inputs):
    task = fdlm.Task()
    task.text = ipt[0]
    task.max_dec_len = 1024
    task.min_dec_len = 1
    task.penalty_score = 1.0
    task.temperature = 1.0
    task.topp = 0.0
    task.frequency_score = 0.0
    task.eos_token_id = 2
    task.presence_score = 0.0
    task.task_id = i
    task.call_back_func = call_back
    model.add_request(task)

model.start()
# 停止接收新的请求，处理完请求后，全部自行退出
model.stop()
```
