from fastdeploy.engine.sampling_params import SamplingParams
from fastdeploy.entrypoints.llm import LLM

model_name_or_path = "./models/Qwen2-7B-Instruct"

IMAGE_PLACEHOLDER = "<|image@placeholder|>"

# 超参设置
sampling_params = SamplingParams(temperature=0.1, max_tokens=30)
llm = LLM(model=model_name_or_path, tensor_parallel_size=1)
prompt = {
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": "file://mytest/images/demo.jpeg"
                },
                {
                    "type": "text",
                    "text": "Describe this image."
                },
            ]
        }
    ]
}
output = llm.generate(prompts="who are you？", use_tqdm=True, sampling_params=sampling_params)

print(output)
