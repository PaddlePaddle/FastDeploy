import os

os.environ["MACA_VISIBLE_DEVICES"] = "0,1"
os.environ["FD_MOE_BACKEND"] = "cutlass"
os.environ["PADDLE_XCCL_BACKEND"] = "metax_gpu"
os.environ["FLAGS_weight_only_linear_arch"] = "80"
os.environ["FD_METAX_KVCACHE_MEM"] = "8"
os.environ["ENABLE_V1_KVCACHE_SCHEDULER"] = "1"
os.environ["FD_ENC_DEC_BLOCK_NUM"] = "2"
os.environ["FD_ENABLE_E2W_TENSOR_CONVERT"] = "0"
os.environ["FD_ENGINE_TASK_QUEUE_WITH_SHM"] = "0"


import fastdeploy

sampling_params = fastdeploy.SamplingParams(top_p=0.95, max_tokens=2048, temperature=0.6)

llm = fastdeploy.LLM(
    model="/data/models/PaddlePaddle/ERNIE-4.5-VL-28B-A3B-Thinking",
    tensor_parallel_size=2,
    engine_worker_queue_port=8899,
    max_model_len=2048,
    quantization="wint8",
    load_choices="default_v1",
    disable_custom_all_reduce=True,
    graph_optimization_config={"use_cudagraph": False, "graph_opt_level": 0},
)

prompts = [
    "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
]

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs.text
    print(f"Prompt: {prompt!r}")
    print(f"Generated: {generated_text!r}")
