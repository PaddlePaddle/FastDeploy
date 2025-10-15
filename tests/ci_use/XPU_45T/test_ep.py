import os
import copy
from fastdeploy import LLM, SamplingParams

msg1=[
    {"role": "system", "content": ""},
    {"role": "user", "content": "北京天安门广场在哪里?"},
]

messages = [msg1,
           ]

# 采样参数
sampling_params = SamplingParams(top_p=0, max_tokens=500)
model=os.getenv("model_path", "/ssd3/model/ERNIE-4.5-300B-A47B-Paddle")

xpu_visible_devices=os.getenv("XPU_VISIBLE_DEVICES", "0")
xpu_device_num=len(xpu_visible_devices.split(','))
enable_expert_parallel=True
if enable_expert_parallel:
    tensor_parallel_size=1
    data_parallel_size=xpu_device_num
else:
    tensor_parallel_size=xpu_device_num
    data_parallel_size=1
engine_worker_queue_port=[str(8023+i*10) for i in range(data_parallel_size)]
engine_worker_queue_port=",".join(engine_worker_queue_port)

# messages=[copy.deepcopy(msg1) for i in range(data_parallel_size)]
print(f"messages: {messages}")

llm = LLM(model=model,
          enable_expert_parallel=enable_expert_parallel,
          tensor_parallel_size=tensor_parallel_size,
          data_parallel_size=data_parallel_size,
          max_model_len=8192,
          quantization="wint4",
          engine_worker_queue_port=engine_worker_queue_port,
          max_num_seqs=8,
         )

# 批量进行推理（llm内部基于资源情况进行请求排队、动态插入处理）
outputs = llm.chat(messages, sampling_params)

# 输出结果
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs.text
    print(f"-"*100)
    print(f"prompt: {prompt}")
    print(f"-"*100)
    print(f"generated_text: {generated_text}")
    print(f"-"*100)