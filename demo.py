from fastdeploy import LLM, SamplingParams
import os

# os.environ["FD_ATTENTION_BACKEND"] = "MOBA_ATTN"
# os.environ["FD_ATTENTION_BACKEND"] = "FLASH_ATTN"
os.environ["FD_ATTENTION_BACKEND"] = "DYNAMIC_QUANT_CACHE_ATTN"

os.environ["FLAGS_flash_attn_version"] = "3"

prompts = [
    "User: 小县城加盟蜜雪冰城如何赚钱。\nAssistant: 对于小县城来说，加盟蜜雪冰城是一个不错的赚钱方式。以下是一些可能的赚钱方式：\n\n1. 提供优质的服务和产品：蜜雪冰城在市场上拥有良好的声誉，因此加盟商可以借助它的品牌和知名度来吸引更多的消费者。加盟商可以通过提供高品质的食品和服务来提高客户满意度，同时也可以通过不断创新和改进来保持竞争力。\n\n2. 开展促销活动：加盟商可以在节假日、店庆等时间点开展促销活动，以提高销售额。促销活动可以包括打折、满减、买一送一等。\n\n3. 拓展新市场：加盟商可以通过开发新市场来扩大业务范围。例如，可以在不同的地区开设新的店铺，或者在不同的行业、品牌之间开展合作。\n\n4. 利用社交媒体：社交媒体可以帮助加盟商扩大品牌知名度和影响力。加盟商可以在社交媒体上发布照片、视频、文章等来宣传品牌和产品，吸引更多的消费者。"
]


sampling_params = SamplingParams(top_p=0.0, max_tokens=1024)
model_dir = "/root/paddlejob/workspace/output/yangjianfeng/base_c2_attn_ckpt"

if not os.environ["FD_ATTENTION_BACKEND"] == "DYNAMIC_QUANT_CACHE_ATTN":
    model_dir = "/root/paddlejob/workspace/output/yangjianfeng/base_23k_ckpt"

llm = LLM(model=model_dir, tensor_parallel_size=1, max_model_len=8192, engine_worker_queue_port=8521)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs.text
    print("Generated text:", generated_text)


# fa3 = [23, 23, 8, 93937]
# c2 =  [23, 23, 23, 94171]