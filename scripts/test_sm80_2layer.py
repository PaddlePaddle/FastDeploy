"""
FD MiniMax-M2.5 SM80 BF16 e2e test.

Usage:
    CUDA_VISIBLE_DEVICES=4,5 FD_MARLIN_FP8=1 python scripts/test_sm80_2layer.py
"""
import os, sys, json
sys.path.insert(0, '/data/lizhijun/work/fd-vllm/FastDeploy')

MODEL_DIR = '/data-ssd/lizhijun/models/MiniMax/MiniMax-M2.5'
cfg_path = os.path.join(MODEL_DIR, 'config.json')

# Backup & set 2 layers
with open(cfg_path) as f:
    orig_cfg = f.read()
d = json.loads(orig_cfg)
d['num_hidden_layers'] = 2
with open(cfg_path, 'w') as f:
    json.dump(d, f, indent=2)

try:
    from fastdeploy.entrypoints.llm import LLM
    from fastdeploy.engine.sampling_params import SamplingParams

    llm = LLM(
        model=MODEL_DIR,
        tensor_parallel_size=2,
        enable_expert_parallel=True,
        disable_sequence_parallel_moe=True,
        max_model_len=512,
        gpu_memory_utilization=0.85,
        max_num_seqs=4,
        num_gpu_blocks_override=100,
        max_num_batched_tokens=512,
        graph_optimization_config={'use_cudagraph': False},
    )

    outputs = llm.generate(['Hello, how are you?'], SamplingParams(temperature=0, max_tokens=5))
    for out in outputs:
        print('prompt:', repr(out.prompt))
        if out.outputs:
            print('text:', repr(out.outputs[0].text))
            print('tokens:', out.outputs[0].token_ids)
    print('SUCCESS')
finally:
    with open(cfg_path, 'w') as f:
        f.write(orig_cfg)
    print('config.json restored')
