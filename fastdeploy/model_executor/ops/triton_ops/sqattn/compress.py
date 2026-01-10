import gc
import json
from dataclasses import dataclass
from typing import Dict

import numpy as np
import paddle
from loguru import logger
from .attn_replacer import replace_mp_triton_for_block
from .calibration import get_static_calib_dataset_longbench
from .window_search import calibrate_layer_windows
from .model_utils import (batch_layer_infer, get_blocks, move_embed)
from tqdm import tqdm

paddle.set_device("gpu" if paddle.device.is_compiled_with_cuda() else "cpu")
device = "gpu" if paddle.device.is_compiled_with_cuda() else "cpu"


@dataclass
class CalibrationConfig:
    """Calibration configuration"""

    use_calibration: bool = True
    energy_threshold: float = 0.95
    save_checkpoints: bool = True
    checkpoint_interval: int = 4
    default_bit8_window: int = 128
    default_bit4_window: int = 256
    default_sink_window: int = 16


def get_layer_inputs(model, layers, sample, device):
    """
    Get inputs to first layer using Catcher mechanism
    """
    inps = []
    layer_kwargs = {}

    # class Catcher(paddle.nn.Layer):
    #     def __init__(self, module):
    #         super().__init__()
    #         self.module = module

    #     def forward(self, inp, **kwargs):
    #         inps.append(inp)
    #         layer_kwargs.update(kwargs)
    #         layer_kwargs["use_cache"] = False
    #         raise ValueError
    #     def __getattr__(self, name):
    #         # 先尝试父类属性
    #         try:
    #             return super().__getattribute__(name)
    #         except AttributeError:
    #             # 向下转发到被包装的子模块
    #             return getattr(self.module, name)
    def capture_inputs(layer, inp):
        inps.append(inp)
        return inp
    hook = layers[0].register_forward_pre_hook(capture_inputs)

    layers[0] = layers[0].to(device)
    move_embed(model, device)
    try:
        if model.__class__.__name__ == "LlavaLlamaModel":
            model.llm(sample)
        else:
            model(sample)
    except ValueError:
        pass
    layers[0] = layers[0].to("cpu")
    move_embed(model, "cpu")
    paddle.device.synchronize()
    gc.collect()
    return inps[0], layer_kwargs


def save_checkpoint(layer_idx: int, windows_dict: Dict, bits_alloc: Dict):
    """Save checkpoint for recovery"""
    checkpoint = {
        "layer_idx": layer_idx,
        "windows": {f"{k[0]}_{k[1]}": v for k, v in windows_dict.items()},
        "bits_alloc": bits_alloc,
    }
    path = f"checkpoint_layer_{layer_idx}.json"
    with open(path, "w") as f:
        json.dump(checkpoint, f, indent=2)
    logger.info(f"Checkpoint saved: {path}")


def save_final_results(
    windows_dict: Dict, bits_alloc: Dict, config: CalibrationConfig, output_path: str
):
    """Save final calibration results with statistics"""
    all_bit8 = []
    all_bit4 = []
    for layer_config in bits_alloc.values():
        if isinstance(layer_config, dict):
            all_bit8.append(layer_config.get("bit8", 0))
            all_bit4.append(layer_config.get("bit4", 0))
    results = {
        "configuration": {
            "energy_threshold": config.energy_threshold,
            "use_calibration": config.use_calibration,
        },
        "bits_allocation": bits_alloc,
        "statistics": {
            "bit8": {
                "mean": np.mean(all_bit8) if all_bit8 else 0,
                "std": np.std(all_bit8) if all_bit8 else 0,
                "min": np.min(all_bit8) if all_bit8 else 0,
                "max": np.max(all_bit8) if all_bit8 else 0,
            },
            "bit4": {
                "mean": np.mean(all_bit4) if all_bit4 else 0,
                "std": np.std(all_bit4) if all_bit4 else 0,
                "min": np.min(all_bit4) if all_bit4 else 0,
                "max": np.max(all_bit4) if all_bit4 else 0,
            },
        },
    }
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved: {output_path}")


def print_gpu_mem(tag=""):
    device = paddle.device.get_device()
    if device.startswith("gpu"):
        gpu_id = int(device.split(":")[-1])
        alloc = paddle.device.cuda.memory_allocated(gpu_id) / 1024**3
        resv = paddle.device.cuda.memory_reserved(gpu_id) / 1024**3
        print(f"[{tag}] allocated={alloc:.2f} GB, reserved={resv:.2f} GB on {device}")
    else:
        print(f"[{tag}] running on {device}, no GPU memory info.")
    return alloc, resv

def compress_model(model, tokenizer, device, args):
    """
    Compress model with integrated layer-by-layer calibration
    """
    layers = get_blocks(model)
    logger.info(f"Starting compression with {len(layers)} layers")
    logger.info("Phase 1: Preparing calibration data...")
    raw_samples, _ = get_static_calib_dataset_longbench(
        tokenizer=tokenizer
    )
    if not isinstance(raw_samples, list):
        raw_samples = [raw_samples]
    # max_len = max([sample.size(1) for sample in raw_samples])
    max_len = max([sample.shape[1] for sample in raw_samples])
    samples_inps, samples_layer_kwargs = [], []

    for i, sample in enumerate(raw_samples):
        sample = sample[:, :500]
        inps, layer_kwargs = get_layer_inputs(model, layers, sample, device)
        
        samples_inps.append(inps[0])
        layer_kwargs['position_embeddings'] = inps[4]
        layer_kwargs['attention_mask'] = inps[1]
        layer_kwargs['past_key_value'] = inps[2]
        layer_kwargs['use_cache'] = inps[3]
        
        samples_layer_kwargs.append(layer_kwargs)
        del inps, layer_kwargs
        paddle.device.cuda.empty_cache()
        alloc, resv = print_gpu_mem(f"sample {i}")
        if alloc > 95: import pdb; pdb.set_trace()
        break  #  for debug
    del raw_samples
    paddle.device.cuda.empty_cache()
    gc.collect()

    logger.info("Phase 2: Processing layers...")
    bits_alloc = {}
    
    for layer_idx in tqdm(range(len(layers)), desc="Compressing"):
        layer = layers[layer_idx].to(device)
        if layer_idx not in [0, len(layers) - 1]:
            bit8_window, bit4_window, head_windows = calibrate_layer_windows(
                layer, layer_idx, samples_inps, samples_layer_kwargs, max_len, args
            )
            bit8_windows, bit4_windows = [], []
            for hw in head_windows:
                bit8_windows.append(hw["bit8_relative"])
                bit4_windows.append(hw["bit4_relative"])
            bits_alloc[layer_idx] = {
                "bit8": bit8_windows,
                "bit4": bit4_windows,
                "sink": 256,
            }
            replace_mp_triton_for_block(
                layer,
                layer_idx,
                args,
                bit8_window_sizes=bit8_windows,
                bit4_window_sizes=bit4_windows,
                sink_window_size=256,
            )
            
        samples_inps = batch_layer_infer(
            layer, samples_inps, samples_layer_kwargs, args
        )
        
        layer.to("cpu")
        paddle.device.cuda.empty_cache()

    del args.current_attention
    logger.info("Compression complete!")
    return bits_alloc


def process_model(model, window_sizes=None, args=None, method="sqattn"):
    if method == 'naive':
        from fastdeploy.model_executor.ops.triton_ops.sqattn.attn_replacer import replace_sdpa_for_block as replace_attn_fn
        for i in range(len(model.model.layers)):
            layer = model.model.layers[i]
            replace_attn_fn(
                layer,
                i,
                args,
            )
    elif method == 'sqattn':
        from fastdeploy.model_executor.ops.triton_ops.sqattn.attn_replacer import \
            replace_mp_triton_for_block as replace_attn_fn

        for i in range(len(model.model.layers)):
            if i in window_sizes.keys():
                layer = model.model.layers[i]
                bit8_window_sizes = window_sizes[i]["bit8"]
                bit4_window_sizes = window_sizes[i]["bit4"]
                sink_size = window_sizes[i]["sink"]
                replace_attn_fn(
                    layer,
                    i,
                    args,
                    bit8_window_sizes=bit8_window_sizes,
                    bit4_window_sizes=bit4_window_sizes,
                    sink_window_size=256,
                )
    return model
