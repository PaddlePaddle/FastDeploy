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

prompts =  20*["勒布朗詹姆斯是谁?",
            "背景信息:\n请根据给出的\"话题名称\"、\"话题释义\"，依据下列审核规则逐级判定是否相关，给出推理过程以及推理结果。\n******\n输出风格:\n1. 输出格式必须以json格式为输出，包含两个字段，分别是\"reason\"和\"result\"，其中“reason”代表推理过程，“result”代表推理结果。\n2. 请注意\"result\"中只能输出一个答案，答案包含2种类型，分别是“是”、“否”，不允许回答其他答案。\n3. 任一一条规则判定为\"是\"，立即输出\"result\"为是，并结束判断。\n******\n审核规则：\n规则1：\n判定\"话题释义\"中是否含有无需审核、不需要审核、不限内容等词汇或近义词：如果含有，直接输出\"result\"为是，停止后续规则的判断。\n规则2：\n判定\"话题名称\"中是否含有创作挑战赛、全勤挑战赛、创作激励、激励计划、招募计划、优质计划、二创计划、乘风计划、萤火计划、轻知计划、达人招募、质感创作人等词汇：如果含有，直接输出\"result\"为是，停止后续规则的判断。\n规则3：\n判定\"标题\"和\"话题名称\"是否存在一定的关联性、泛相关、潜在关联：如果存在，注意不需要考虑是否为直接相关、严格相关、完全相关及关联性强弱，直接输出\"result\"为是，停止后续规则的判断。\n规则4：\n判定\"作者名称\"和\"话题名称\"是否相关：如果存在，直接输出\"result\"为是，停止后续规则的判断。\n规则5：\n判定\"话题名称\"为影视、电影、电视剧、综艺、动漫、短剧及其解说、推荐、片段、欣赏、剪辑、分销等，但不涉及具体的剧名、节目名称，如果\"兴趣点\"包含影视、综艺、动漫、情景剧，直接输出\"result\"为是，停止后续规则的判断。\n规则6：\n判定\"话题名称\"为美女、美女评选、性感美女、模特、性感模特及类似内容，如果\"兴趣点\"包含拍人、秀身材、秀颜值、热舞、美女、性感、写真、模特、名模、车模、秀场、比基尼，直接输出\"result\"为是，停止后续规则的判断。\n规则7：\n判定\"兴趣点\"和\"话题名称\"是否存在一定的关联性、泛相关、潜在关联：如果存在，注意不需要考虑是否为直接相关、严格相关、完全相关及关联性强弱，直接输出\"result\"为是，停止后续规则的判断。\n规则8：\n判定\"兴趣点\"和\"话题释义\"是否存在一定的关联性、泛相关、潜在关联：如果存在，注意不需要考虑是否为直接相关、严格相关、完全相关及关联性强弱，注意这里不需要考虑与话题名称是否相关，直接输出\"result\"为是，停止后续规则的判断。\n规则9：\n判定\"ASR\"和\"话题名称\"是否存在一定的关联性、泛相关、潜在关联：如果存在，注意不需要考虑是否为直接相关、严格相关、完全相关及关联性强弱，直接输出\"result\"为是，停止后续规则的判断。\n规则10：\n判定\"OCR\"和\"话题名称\"是否存在一定的关联性、泛相关、潜在关联：如果存在，不需要考虑是否为直接相关、严格相关、完全相关及关联性强弱，直接输出\"result\"为是，停止后续规则的判断。\n******\n话题名称：大S\n话题释义：\n标题：妈把孩子给小汪\n作者名称：二橙运乐子1q\n兴趣点：明星周边、明星八卦、[\"小汪\",\"徐熙媛\"]\nASR：s家操作看不懂s妈把孩子给小汪不是好心是为了遗产户口不在s家继承的少大虚线也一样，小汪大家都说好其实都是计策现在人走了孩子没人管遗产分三分之一，卤蛋安排的明明白白一个老人要那么多钱干嘛卤蛋要人陪吃饭小s还得去真是离谱s这操作，绝了无语，\nOCR：s妈把孩子给小汪、s妈把孩子给小汪S家操作看不懂s妈把孩子给小汪S家操作看不懂s妈把孩子给小汪s妈把孩子给小汪S妈把孩子给小汪s妈把孩子给小汪不是好心s妈把孩子给小汪不是好心s妈把孩子给小汪不是好心s妈把孩子给小汪是为了遗产s妈把孩子给小汪户口不在S家s妈把孩子给小汪户口不在S家s妈把孩子给小汪继承的少s妈把孩子给小汪大曲线也一样s妈把孩子给小汪大曲线也一样s妈把孩子给小汪大曲线也一样s妈把孩子给小汪孩子给小汪s妈把孩子给小汪孩子给小汪s妈把孩子给小汪大家都说好s妈把孩子给小汪大家都说好s妈把孩子给小汪大家都说好s妈把孩子给小汪其实都是计策s妈把孩子给小汪其实都是计策s妈把孩子给小汪TTISSOT现在人走了s妈把孩子给小汪TTISSOT孩子没人管s妈把孩子给小汪TISSOT孩子没人管s妈把孩子给小汪孩子没人管s妈把孩子给小汪孩子没人管s妈把孩子给小汪遗产分三分之一s妈把孩子给小汪遗产分三分之一s妈把孩子给小汪卤蛋安排得明明白白s妈把孩子给小汪卤蛋安排得明明白白s妈把孩子给小汪一个老人要那么多钱干嘛s妈把孩子给小汪一个老人要那么多钱干嘛s妈把孩子给小汪卤蛋要人陪吃饭s妈把孩子给小汪卤蛋要人陪吃饭s妈把孩子给小汪小S还得去s妈把孩子给小汪真是离谱s妈把孩子给小汪S家这操作s妈把孩子给小汪绝了s妈把孩子给小汪无语\n",
            "日用药金额，我们叫它DDC。这个数就是药物每天平均花了多少钱算出来的, 用总购入金额除以药物的每日用量强度DDDs就得到了。\n\n  请对以上内容进行格式校对，要求：如果缺少对应的标点符号请补上，如果有多余的空格，请将其删除，文字原本的内容不需要有任何变动。你最终只需要把校对后的结果以json格式输出，json格式：```json{\"result\":校对后结果}``` ",
            "给我三首李白的诗",
            "给我解释什么是花儿",
            "介绍你自己"]


# 采样参数
sampling_params = SamplingParams(temperature=0.7, top_p=0, max_tokens=256)

llm = LLM(model="/cwb/DeepSeekV3-0324-5layers", tensor_parallel_size=8,engine_worker_queue_port=8008, max_model_len=32768,max_num_seqs=256,use_cudagraph=True)

# 批量进行推理（llm内部基于资源情况进行请求排队、动态插入处理）
outputs = llm.generate(prompts, sampling_params)
print("==== The outputs are:", outputs)
# 输出结果
for output in outputs:
    prompt = output.prompt
    #print("===The prompt is : ", prompt)
    generated_text = output.outputs.text
    print("===The generated_text is : ", generated_text)