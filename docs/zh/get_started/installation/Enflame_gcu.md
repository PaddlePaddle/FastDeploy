# 使用 FastDeploy 在燧原 S60 上运行 ERNIE-4.5-21B-A3B模型

燧原 S60（[了解燧原](https://www.enflame-tech.com/)）是面向数据中心大规模部署的新一代人工智能推理加速卡，满足大语言模型、搜广推及传统模型的需求，具有模型覆盖面广、易用性强、易迁移易部署等特点，可广泛应用于图像及文本生成等应用、搜索与推荐、文本、图像及语音识别等主流推理场景。

FastDeploy 在燧原 S60 上对 ernie-4_5-21b-a3b-bf16-paddle 模型进行了深度适配和优化，实现了 GCU 推理入口和 GPU 的统一，无需修改即可完成推理任务的迁移。

## 🚀 快速开始 🚀

### 0. 机器准备。快速开始之前，您需要准备一台插有燧原 S60 加速卡的机器，要求如下：

| 芯片类型 | 驱动版本 | TopsRider 版本 |
| :---: | :---: | :---: |
| 燧原 S60 | 1.5.0.5 | 3.4.623 |

**注：如果需要验证您的机器是否插有燧原 S60 加速卡，只需主机环境下输入以下命令，看是否有输出：**
```bash
lspci | grep S60

# 例如：lspci | grep S60 , 输出如下
08:00.0 Processing accelerators: Shanghai Enflame Technology Co. Ltd S60 [Enflame] (rev 01)
09:00.0 Processing accelerators: Shanghai Enflame Technology Co. Ltd S60 [Enflame] (rev 01)
0e:00.0 Processing accelerators: Shanghai Enflame Technology Co. Ltd S60 [Enflame] (rev 01)
11:00.0 Processing accelerators: Shanghai Enflame Technology Co. Ltd S60 [Enflame] (rev 01)
32:00.0 Processing accelerators: Shanghai Enflame Technology Co. Ltd S60 [Enflame] (rev 01)
38:00.0 Processing accelerators: Shanghai Enflame Technology Co. Ltd S60 [Enflame] (rev 01)
3b:00.0 Processing accelerators: Shanghai Enflame Technology Co. Ltd S60 [Enflame] (rev 01)
3c:00.0 Processing accelerators: Shanghai Enflame Technology Co. Ltd S60 [Enflame] (rev 01)
```
### 1. 环境准备：(这将花费您 5～10min 时间)
1. 拉取镜像
```bash
# 注意此镜像仅为paddle开发环境，镜像中不包含预编译的飞桨安装包
docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-gcu:topsrider3.4.623-ubuntu20-x86_64-gcc84
```
2. 参考如下命令启动容器
```bash
docker run --name paddle-gcu-llm -v /home:/home -v /work:/work --network=host --ipc=host -it --privileged ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-gcu:topsrider3.4.623-ubuntu20-x86_64-gcc84 /bin/bash
```
3. 获取并安装驱动<br/>
**docker 内提前放置了全量软件包，需拷贝至 docker 外目录，如：```/home/workspace/deps/```**
```bash
mkdir -p /home/workspace/deps/ && cp /root/TopsRider_i3x_*/TopsRider_i3x_*_deb_amd64.run /home/workspace/deps/
```
4. 安装驱动<br/>
**此操作需要在主机环境下执行**
```bash
cd /home/workspace/deps/
bash TopsRider_i3x_*_deb_amd64.run --driver --no-auto-load -y
```
驱动安装完成后**重新进入 docker**，参考如下命令
```bash
docker start paddle-gcu-llm
docker exec -it paddle-gcu-llm bash
```
5. 安装 PaddlePaddle<br/>
```bash
# PaddlePaddle『飞桨』深度学习框架，提供运算基础能力
python -m pip install paddlepaddle==3.1.0a0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
```
6. 安装 PaddleCustomDevice<br/>
```bash
# PaddleCustomDevice是PaddlePaddle『飞桨』深度学习框架的自定义硬件接入实现，提供GCU的算子实现
python -m pip install paddle-custom-gcu==3.1.0 -i https://www.paddlepaddle.org.cn/packages/stable/gcu/
# 如想源码编译安装，请参考https://github.com/PaddlePaddle/PaddleCustomDevice/blob/develop/backends/gcu/README_cn.md
```
7. 安装 FastDeploy 和 依赖<br/>
```bash
python -m pip install fastdeploy -i https://www.paddlepaddle.org.cn/packages/stable/gcu/ --extra-index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simplels
apt install python3.10-distutils
```
### 2. 数据准备：(这将花费您 2～5min 时间)
使用训练好的模型，在 GSM8K 上推理
```bash
mkdir -p /home/workspace/benchmark/ && cd /home/workspace/benchmark/
wget https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl
```
准备模型和权重，置于环境目录，如：```/work/models/ernie-4_5-21b-a3b-bf16-paddle/```
### 3. 推理：(这将花费您 2~5min 时间)
执行如下命令启动推理服务
```bash
python -m fastdeploy.entrypoints.openai.api_server \
    --model "/work/models/ernie-4_5-21b-a3b-bf16-paddle/" \
    --port 8188 \
    --metrics-port 8200 \
    --tensor-parallel-size 4 \
    --max-model-len 8192 \
    --num-gpu-blocks-override 1024
```
使用如下命令请求模型服务
```bash
curl -X POST "http://0.0.0.0:8188/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": "The largest ocean is"}
  ]
}'
```
成功运行后，可以查看到推理结果的生成，样例如下
```json
{"id":"chatcmpl-5cd96f3b-eff3-4dc0-8aa2-8b5d7b7b86f2","object":"chat.completion","created":1751167862,"model":"default","choices":[{"index":0,"message":{"role":"assistant","content":"3. **Pacific Ocean**: The Pacific Ocean is the largest and deepest of the world's oceans. It covers an area of approximately 181,344,000 square kilometers, which is more than 30% of the Earth's surface. It is located between the Americas to the west and east, and Asia and Australia to the north and south. The Pacific Ocean is known for its vastness, diverse marine life, and numerous islands.\n\nIn summary, the largest ocean in the world is the Pacific Ocean.","reasoning_content":null,"tool_calls":null},"finish_reason":"stop"}],"usage":{"prompt_tokens":11,"total_tokens":127,"completion_tokens":116,"prompt_tokens_details":{"cached_tokens":0}}}
```
### 4. 精度测试：(这将花费您 60~180min 时间)
准备精度脚本 ```bench_gsm8k.py``` 置于 ```/home/workspace/benchmark/``` ，并修改采样参数，如：
```bash
data = {
    "messages": [
        {
            "role": "user",
            "content": prompt,
        }
    ],
    "temperature": 0.6,
    "max_tokens": 2047,
    "top_p": 0.95,
    "do_sample": True,
}
```
执行以下命令启动精度测试
```bash
cd /home/workspace/benchmark/
python -u bench_gsm8k.py --port 8188 --num-questions 1319 --num-shots 5 --parallel 2
```
执行成功运行后，当前目录可以查看到精度结果的生成，文件为 ```result.jsonl```，样例如下（部分数据集，仅示例）
```json
{"task": "gsm8k", "backend": "paddlepaddle", "num_gpus": 1, "latency": 365.548, "accuracy": 0.967, "num_requests": 30, "other": {"num_questions": 30, "parallel": 2}}
```

