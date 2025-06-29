# Running ERNIE-4.5-21B-A3B with FastDeploy

The Enflame S60 ([Learn about Enflame](https://www.enflame-tech.com/)) is a next-generation AI inference accelerator card designed for large-scale deployment in data centers. It meets the demands of large language models (LLMs), search/advertising/recommendation systems, and traditional models. Characterized by broad model coverage, user-friendliness, and high portability, it is widely applicable to mainstream inference scenarios such as image and text generation applications, search and recommendation systems, and text/image/speech recognition.

FastDeploy has deeply adapted and optimized the ernie-4_5-21b-a3b-bf16-paddle model for the Enflame S60, achieving a unified inference interface between GCU and GPU. This allows seamless migration of inference tasks without code modifications.

## 🚀 Quick Start 🚀

### 0. Machine Preparation
Before starting, prepare a machine equipped with Enflame S60 accelerator cards. Requirements:

| Chip Type | Driver Version | TopsRider Version |
| :---: | :---: | :---: |
| Enflame S60 | 1.5.0.5 | 3.4.623 |

**Note: To verify if your machine has Enflame S60 accelerator cards installed, run the following command in the host environment and check for output:**
```bash
lspci | grep S60

# Example: lspci | grep S60, Example Output:
08:00.0 Processing accelerators: Shanghai Enflame Technology Co. Ltd S60 [Enflame] (rev 01)
09:00.0 Processing accelerators: Shanghai Enflame Technology Co. Ltd S60 [Enflame] (rev 01)
0e:00.0 Processing accelerators: Shanghai Enflame Technology Co. Ltd S60 [Enflame] (rev 01)
11:00.0 Processing accelerators: Shanghai Enflame Technology Co. Ltd S60 [Enflame] (rev 01)
32:00.0 Processing accelerators: Shanghai Enflame Technology Co. Ltd S60 [Enflame] (rev 01)
38:00.0 Processing accelerators: Shanghai Enflame Technology Co. Ltd S60 [Enflame] (rev 01)
3b:00.0 Processing accelerators: Shanghai Enflame Technology Co. Ltd S60 [Enflame] (rev 01)
3c:00.0 Processing accelerators: Shanghai Enflame Technology Co. Ltd S60 [Enflame] (rev 01)
```
### 1. Environment Setup (Estimated time: 5–10 minutes)
1. Pull the Docker image
```bash
# Note: This image only contains the Paddle development environment, not precompiled PaddlePaddle packages
docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-gcu:topsrider3.4.623-ubuntu20-x86_64-gcc84
```
2. Start the container
```bash
docker run --name paddle-gcu-llm -v /home:/home -v /work:/work --network=host --ipc=host -it --privileged ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-gcu:topsrider3.4.623-ubuntu20-x86_64-gcc84 /bin/bash
```
3. Obtain and install drivers<br/>
**Full software packages are preloaded in the Docker container. Copy them to an external directory, e.g., ```/home/workspace/deps/```**
```bash
mkdir -p /home/workspace/deps/ && cp /root/TopsRider_i3x_*/TopsRider_i3x_*_deb_amd64.run /home/workspace/deps/
```
4. Install drivers<br/>
**Execute this operation in the host environment**
```bash
cd /home/workspace/deps/
bash TopsRider_i3x_*_deb_amd64.run --driver --no-auto-load -y
```
After driver installation, **re-enter the Docker container**:
```bash
docker start paddle-gcu-llm
docker exec -it paddle-gcu-llm bash
```
5. Install PaddlePaddle
```bash
# PaddlePaddle Deep Learning Framework provides fundamental computing capabilities
python -m pip install paddlepaddle==3.1.0a0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
```
6. Install PaddleCustomDevice<br/>
```bash
# PaddleCustomDevice implements custom hardware backend for PaddlePaddle, providing GCU operator implementations
python -m pip install paddle-custom-gcu==3.1.0 -i https://www.paddlepaddle.org.cn/packages/stable/gcu/
# For source compilation, refer to: https://github.com/PaddlePaddle/PaddleCustomDevice/blob/develop/backends/gcu/README_cn.md
```
7. Install FastDeploy and dependencies
```bash
python -m pip install fastdeploy -i https://www.paddlepaddle.org.cn/packages/stable/gcu/ --extra-index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simplels
apt install python3.10-distutils
```
### 2. Data Preparation (Estimated time: 2–5 minutes)
Use a trained model for inference on GSM8K dataset:
```bash
mkdir -p /home/workspace/benchmark/ && cd /home/workspace/benchmark/
wget https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl
```
Place model weights in a directory, e.g., ```/work/models/ernie-4_5-21b-a3b-bf16-paddle/```
### 3. Inference (Estimated time: 2–5 minutes)
Start the inference service:
```bash
python -m fastdeploy.entrypoints.openai.api_server \
    --model "/work/models/ernie-4_5-21b-a3b-bf16-paddle/" \
    --port 8188 \
    --metrics-port 8200 \
    --tensor-parallel-size 4 \
    --max-model-len 8192 \
    --num-gpu-blocks-override 1024
```
Query the model service:
```bash
curl -X POST "http://0.0.0.0:8188/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": "The largest ocean is"}
  ]
}'
```
Successful execution returns inference results, e.g.:
```json
{"id":"chatcmpl-5cd96f3b-eff3-4dc0-8aa2-8b5d7b7b86f2","object":"chat.completion","created":1751167862,"model":"default","choices":[{"index":0,"message":{"role":"assistant","content":"3. **Pacific Ocean**: The Pacific Ocean is the largest and deepest of the world's oceans. It covers an area of approximately 181,344,000 square kilometers, which is more than 30% of the Earth's surface. It is located between the Americas to the west and east, and Asia and Australia to the north and south. The Pacific Ocean is known for its vastness, diverse marine life, and numerous islands.\n\nIn summary, the largest ocean in the world is the Pacific Ocean.","reasoning_content":null,"tool_calls":null},"finish_reason":"stop"}],"usage":{"prompt_tokens":11,"total_tokens":127,"completion_tokens":116,"prompt_tokens_details":{"cached_tokens":0}}}
```
### 4. Accuracy Testing (Estimated time: 60–180 minutes)
Place the accuracy script ```bench_gsm8k.py``` in ```/home/workspace/benchmark/``` and modify sampling parameters, e.g.:
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
Run accuracy tests:
```bash
cd /home/workspace/benchmark/
python -u bench_gsm8k.py --port 8188 --num-questions 1319 --num-shots 5 --parallel 2
```
Upon completion, accuracy results are saved in ```result.jsonl```, e.g.:
```json
{"task": "gsm8k", "backend": "paddlepaddle", "num_gpus": 1, "latency": 365.548, "accuracy": 0.967, "num_requests": 30, "other": {"num_questions": 30, "parallel": 2}}
```

