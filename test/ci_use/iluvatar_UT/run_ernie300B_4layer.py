from fastdeploy import LLM, SamplingParams

prompts = [
    "Hello, my name is",
]

# 采样参数
sampling_params = SamplingParams(temperature=0.8, top_p=0.00001, max_tokens=16)

# 加载模型
llm = LLM(model="/data1/fastdeploy/ERNIE_300B_4L", tensor_parallel_size=16, max_model_len=8192, static_decode_blocks=0, quantization='wint8', block_size=16)

# 批量进行推理（llm内部基于资源情况进行请求排队、动态插入处理）
outputs = llm.generate(prompts, sampling_params)

assert outputs[0].outputs.token_ids==[23768, 97000, 47814, 59335, 68170, 183, 49080, 94717, 82966, 99140, 31615, 51497, 94851, 60764, 10889, 2]
