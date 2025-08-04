# ERNIE-4.5-VL多模态模型

本文档讲解如何部署ERNIE-4.5-VL多模态模型，支持用户使用多模态数据与模型进行对话交互(包含思考Reasoning)，在开始部署前，请确保你的硬件环境满足如下条件：

- GPU驱动 >= 535
- CUDA >= 12.3
- CUDNN >= 9.5
- Linux X86_64
- Python >= 3.10
- 80G A/H 8卡

安装FastDeploy方式参考[安装文档](./installation/README.md)。

>💡 **提示**：  ERNIE多模态系列模型均支持思考模式，可以通过在发起服务请求时设置 ```enable_thinking``` 开启（参考如下示例）。

## 准备模型
部署时指定```--model baidu/ERNIE-4.5-VL-424B-A47B-Paddle```即可自动从AIStudio下载模型，并支持断点续传。你也可以自行从不同渠道下载模型，需要注意的是FastDeploy依赖Paddle格式的模型，更多说明参考[支持模型列表](../supported_models.md)。

## 启动服务

执行如下命令，启动服务,其中启动命令配置方式参考[参数说明](../parameters.md)

**注意**： 由于模型参数量为424B-A47B，在80G * 8卡的机器上，需指定```--quantization wint4```(wint8也可部署)。

```shell
python -m fastdeploy.entrypoints.openai.api_server \
       --model baidu/ERNIE-4.5-VL-424B-A47B-Paddle \
       --port 8180 --engine-worker-queue-port 8181 \
       --cache-queue-port 8182 --metrics-port 8182 \
       --tensor-parallel-size 8 \
       --quantization wint4 \
       --max-model-len 32768 \
       --max-num-seqs 32 \
       --mm-processor-kwargs '{"video_max_frames": 30}' \
       --limit-mm-per-prompt '{"image": 10, "video": 3}' \
       --reasoning-parser ernie-45-vl
```

## 用户发起服务请求
执行启动服务指令后，当终端打印如下信息，说明服务已经启动成功。

```shell
api_server.py[line:91] Launching metrics service at http://0.0.0.0:8181/metrics
api_server.py[line:94] Launching chat completion service at http://0.0.0.0:8180/v1/chat/completions
api_server.py[line:97] Launching completion service at http://0.0.0.0:8180/v1/completions
INFO:     Started server process [13909]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8180 (Press CTRL+C to quit)
```

FastDeploy提供服务探活接口，用以判断服务的启动状态，执行如下命令返回 ```HTTP/1.1 200 OK``` 即表示服务启动成功。

```shell
curl -i http://0.0.0.0:8180/health
```

通过如下命令发起服务请求

```shell
curl -X POST "http://0.0.0.0:8180/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": "把李白的静夜思改写为现代诗"}
  ]
}'
```

输入包含图片时，按如下命令发起请求

```shell
curl -X POST "http://0.0.0.0:8180/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": [
      {"type":"image_url", "image_url": {"url":"https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example2.jpg"}},
      {"type":"text", "text":"图中的文物属于哪个年代?"}
    ]}
  ]
}'
```

输入包含视频时，按如下命令发起请求

```shell
curl -X POST "http://0.0.0.0:8180/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": [
      {"type":"video_url", "video_url": {"url":"https://bj.bcebos.com/v1/paddlenlp/datasets/paddlemix/demo_video/example_video.mp4"}},
      {"type":"text", "text":"画面中有几个苹果?"}
    ]}
  ]
}'
```

当前ERNIE-4.5-VL模型支持思考模式且默认开启，按如下命令可关闭思考模式

```shell
curl -X POST "http://0.0.0.0:8180/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": [
      {"type": "image_url", "image_url": {"url": "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example2.jpg"}},
      {"type": "text", "text": "图中的文物属于哪个年代"}
    ]}
  ],
  "chat_template_kwargs":{"enable_thinking": false}
}'
```

FastDeploy服务接口兼容OpenAI协议，可以通过如下Python代码发起服务请求, 以下示例开启流式用法。

```python
import openai
host = "0.0.0.0"
port = "8180"
client = openai.Client(base_url=f"http://{host}:{port}/v1", api_key="null")

response = client.chat.completions.create(
    model="null",
    messages=[
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example2.jpg"}},
            {"type": "text", "text": "图中的文物属于哪个年代?"},
        ]},
    ],
    stream=True,
)
for chunk in response:
    if chunk.choices[0].delta:
        print(chunk.choices[0].delta.content, end='')
print('\n')
```

## 模型输出
包含思考的输出示例如下, 思考内容在 `reasoning_content` 字段中, 模型回复内容在 `content` 字段中。

```json
{
    "id": "chatcmpl-c4772bea-1950-4bf4-b5f8-3d3c044aab06",
    "object": "chat.completion",
    "created": 1750236617,
    "model": "default",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "图中的文物是**唐代（7-8世纪）的佛陀坐像**，现藏于东京国立博物馆。其年代判断依据如下：\n\n1. **造型特征**：\n   - 佛陀结跏趺坐，双手结禅定印，身披通肩袈裟，衣纹呈阶梯状排列，线条厚重且富有层次感，体现了唐代佛像的典型衣饰风格。\n   - 面部圆润丰腴，双目微闭，嘴角含笑，展现了唐代佛像的慈悲祥和之态，与北魏时期的清瘦造型形成鲜明对比。\n\n2. **背光设计**：\n   - 背光呈舟形，内层雕刻密集的千佛（小佛像），外层装饰火焰纹，这种繁复的背光设计在唐代尤为盛行，象征佛法无边。\n\n3. **工艺与材质**：\n   - 石像表面有风化痕迹，符合唐代石雕历经千年的自然侵蚀特征。唐代多采用汉白玉、砂岩等材质雕刻佛像，注重细节刻画与整体气势。\n\n4. **历史背景**：\n   - 唐代是中国佛教发展的鼎盛时期，统治者推崇佛教，各地开窟造像之风盛行。此像的庄严法相与盛唐时期“丰腴为美”的审美取向高度契合。\n\n综上，此像从艺术风格到工艺特征均符合唐代佛教造像的典型特点，是研究唐代佛教艺术的重要实物资料。",
                "reasoning_content": "用户问的是图中的文物属于哪个年代。首先，我需要确定这张图片中的文物是什么。看起来像是一尊佛像，可能是中国的佛教造像。佛像的造型和装饰风格可能能帮助判断年代。\n\n首先，观察佛像的衣纹和姿势。这尊佛像结跏趺坐，双手放在腿上，可能是在禅定印，这是比较常见的姿势。佛像的衣纹比较厚重，有层次感，可能是北魏或者隋唐时期的风格。北魏时期的佛像通常比较清瘦，衣纹线条硬朗，而隋唐时期的佛像则更丰腴，衣纹流畅。\n\n接下来看背光部分。背光上有许多小佛像，排列成同心圆，这种设计在隋唐时期比较常见，尤其是唐代。北魏时期的背光可能更简单，或者有飞天等装饰，但这种密集的小佛像排列可能更晚一些。\n\n另外，佛像的头部有螺发，肉髻较高，面部圆润，这些都是唐代佛像的特点。北魏的佛像面部通常较为清瘦，鼻梁高挺，而唐代的佛像面部更丰满，表情慈祥。\n\n综合这些特征，这尊佛像可能属于唐代，大约7到8世纪。不过，也有可能属于北魏晚期到隋代之间的过渡时期，但结合衣纹和背光的设计，唐代的可能性更大。需要进一步确认是否有其他特征，比如底座的样式、铭文等，但图片中没有显示这些细节。\n\n可能还需要考虑材质，如果是石雕，唐代常用汉白玉或砂岩，而北魏可能更多使用石灰岩。但图片中的材质看起来像是青铜或铁质，不过也有可能是石雕经过风化后的颜色。不过，佛像的金属质感可能更接近唐代，尤其是如果表面有鎏金的话，但这里看起来有些氧化，可能为铜质。\n\n总之，结合造型、衣纹、背光和面部特征，这尊佛像很可能属于中国唐代的佛教造像，大约7至8世纪。"
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 1260,
        "total_tokens": 2042,
        "completion_tokens": 782,
        "prompt_tokens_details": {
            "cached_tokens": 0
        }
    }
}
```

不包含思考的输出示例如下, 模型回复内容在 `content` 字段中。

```python

{
    "id": "chatcmpl-4d508b96-0ea1-4430-98a6-ae569f74f25b",
    "object": "chat.completion",
    "created": 1750236495,
    "model": "default",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "图中的文物是**北魏太和元年（477年）释迦牟尼佛像**，现收藏于故宫博物院。这尊佛像具有显著的北魏佛像艺术特征，其年代明确，题记中记载了“太和元年”的纪年，即北魏孝文帝元宏的年号。北魏时期（386-534年）是佛教艺术在中国发展的重要阶段，佛像造型逐渐从外来风格转向本土化，此像正是这一转变的典型代表。其衣纹流畅、面相慈祥，背光雕刻精美，展现了北魏中晚期佛像艺术的成熟与独特力。",
                "reasoning_content": null
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 1265,
        "total_tokens": 1407,
        "completion_tokens": 142,
        "prompt_tokens_details": {
            "cached_tokens": 0
        }
    }
}
```
