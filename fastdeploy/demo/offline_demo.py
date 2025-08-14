# """
# # Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License"
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# """
# # export CUTLASS_GEMM_STREAM_K=1
# # model_name_or_path = "/root/paddlejob/workspace/env_run/output/45t04_wint2_perm_2card_all_safetensors"
# # model_name_or_path = "/cwb/models/DeepSeek-V3-0324"
# # model_name_or_path = "/cwb/ernie-4_5-300b-a47b-bf16-paddle"
# model_name_or_path = "/cwb/models/Qwen3-30B-A3B"
# prompts = ["解析三首李白的诗?"]
# from fastdeploy import LLM, SamplingParams
# sampling_params = SamplingParams(temperature=0.7, top_p=0, max_tokens=128)
# llm = LLM(model=model_name_or_path, tensor_parallel_size=1,enable_custom_all_reduce=False,quantization="wint4", use_cudagraph=False,)
# # outputs = llm.chat(messages=[{"role": "user", "content": "给我三首李白的诗"}]*20, sampling_params=sampling_params,)
# outputs = llm.generate(prompts, sampling_params)

# print(outputs)




from fastdeploy import LLM, SamplingParams

prompts = 20 * ["勒布朗詹姆斯是谁?"]


# 采样参数
sampling_params = SamplingParams(temperature=0.7, top_p=0, max_tokens=100)

llm = LLM(model="/cwb/DeepSeekV3-0324-5layers", tensor_parallel_size=8,engine_worker_queue_port=8008, 
quantization="wint4", use_cudagraph=False, enable_custom_all_reduce=True)

# 批量进行推理（llm内部基于资源情况进行请求排队、动态插入处理）
outputs = llm.generate(prompts, sampling_params)
print("==== The outputs are:", outputs)
# 输出结果
for output in outputs:
    prompt = output.prompt
    #print("===The prompt is : ", prompt)
    generated_text = output.outputs.text
    print("===The generated_text is : ", generated_text)