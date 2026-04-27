"""
MiniMax-M2.5 SM80 BF16 End-to-End Test

Usage:
    # Test with different layer counts
    FD_MARLIN_FP8=1 CUDA_VISIBLE_DEVICES=4,5 python tests/e2e/test_minimax_sm80_e2e.py --n-layers 2
    FD_MARLIN_FP8=1 CUDA_VISIBLE_DEVICES=4,5 python tests/e2e/test_minimax_sm80_e2e.py --n-layers 10
    FD_MARLIN_FP8=1 CUDA_VISIBLE_DEVICES=4,5,6,7 python tests/e2e/test_minimax_sm80_e2e.py --n-layers 30 --tp-size 4

Expected: top-1 token should be 367 (\\n\\n) matching vLLM baseline.
"""

import argparse
import json
import os
import sys

parser = argparse.ArgumentParser(description="MiniMax-M2.5 SM80 BF16 E2E Test")
parser.add_argument("--n-layers", type=int, default=2, help="Number of decoder layers to test")
parser.add_argument("--max-tokens", type=int, default=5, help="Max generation tokens")
parser.add_argument("--tp-size", type=int, default=2, help="Tensor parallel size")
parser.add_argument("--prompt", type=str, default="Hello, how are you?", help="Input prompt")
parser.add_argument(
    "--model-dir",
    type=str,
    default="/data-ssd/lizhijun/models/MiniMax/MiniMax-M2.5",
    help="Model directory",
)
args = parser.parse_args()

# Add FastDeploy to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

cfg_path = os.path.join(args.model_dir, "config.json")

# Backup and patch config
with open(cfg_path) as f:
    orig_cfg = f.read()
cfg = json.loads(orig_cfg)
cfg["num_hidden_layers"] = args.n_layers
with open(cfg_path, "w") as f:
    json.dump(cfg, f, indent=2)

try:
    from fastdeploy.engine.sampling_params import SamplingParams
    from fastdeploy.entrypoints.llm import LLM

    tp_size = args.tp_size
    gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if gpus:
        tp_size = len(gpus.split(","))

    llm = LLM(
        model=args.model_dir,
        tensor_parallel_size=tp_size,
        enable_expert_parallel=tp_size > 1,
        disable_sequence_parallel_moe=True,
        max_model_len=256,
        gpu_memory_utilization=0.95,
        max_num_seqs=4,
        num_gpu_blocks_override=500,
        max_num_batched_tokens=256,
        graph_optimization_config={"use_cudagraph": False},
    )

    outputs = llm.generate([args.prompt], SamplingParams(temperature=0, max_tokens=args.max_tokens))

    for out in outputs:
        print(f"Prompt: {out.prompt}")
        if out.outputs:
            o = out.outputs[0]
            print(f"Output text: {repr(o.text)}")
            print(f"Token IDs: {list(o.token_ids)}")
            if o.token_ids and o.token_ids[0] == 367:
                print("PASS: top-1 token is 367 (\\n\\n) - matches vLLM baseline")
            else:
                first_tok = o.token_ids[0] if o.token_ids else None
                print(f"FAIL: top-1 token is {first_tok}, expected 367 (\\n\\n)")

finally:
    # Restore original config
    with open(cfg_path, "w") as f:
        f.write(orig_cfg)
    print(f"\nConfig restored to {json.loads(orig_cfg).get('num_hidden_layers', '?')} layers")
