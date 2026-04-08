import argparse
import json
import os
import re
import time

import paddle
from paddleformers.trainer import strtobool
from paddleformers.transformers.configuration_utils import PretrainedConfig
from paddleformers.transformers.model_utils import shard_checkpoint
from paddleformers.utils.env import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME
from paddleformers.utils.log import logger
from safetensors.numpy import save_file as safe_save_file

from fastdeploy.input.ernie4_5_tokenizer import Ernie4_5Tokenizer
from fastdeploy.model_executor.layers.utils import get_tensor
from fastdeploy.model_executor.load_weight_utils import (
    get_all_weights_file,
    safetensors_weights_iterator,
)
from fastdeploy.model_executor.ops.gpu import w4afp8_gemm_scale_permute


def parse_arguments():
    """
    parse_arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name_or_path",
        default=None,
        required=True,
        help="The directory of model.",
    )

    parser.add_argument(
        "--output_dir",
        default="merged_output",
        required=True,
        help="The directory of merged model output.",
    )

    parser.add_argument(
        "--safe_serialization",
        type=strtobool,
        default="True",
        help="Whether merge the model into safetensors format.",
    )

    parser.add_argument(
        "--moe_quant_type",
        default="w4a8",
        choices=["w4a8", "w4afp8"],
        help="The moe quant type of the model.",
    )

    return parser.parse_args()


def reorder():
    def fn(weight, moe_quant_type):
        from paddle.nn.quant import weight_quantize

        quant_weight, _ = weight_quantize(weight.cuda(), algo=moe_quant_type, arch=80)
        return quant_weight.cpu()

    return fn


def deal_in_scale():
    def fn(in_scale):
        processed_in_scale = 1 / in_scale
        return processed_in_scale

    return fn


def deal_weight_scale():
    def fn(weight_scale, processed_in_scale, moe_quant_type):
        if moe_quant_type == "w4a8":
            processed_weight_scale = weight_scale / (127 * 112) / processed_in_scale
            return processed_weight_scale
        elif moe_quant_type == "w4afp8":
            processed_weight_scale = weight_scale / (448 * 7 * 2 ** (-9)) / processed_in_scale
            processed_weight_scale = w4afp8_gemm_scale_permute(processed_weight_scale.cuda())
            return processed_weight_scale

    return fn


# tmp support w4a8
def deal_quant(state_dict, save_state_dict, moe_quant_type):
    param_mapping = [
        # pattern,fn
        (r"layers\.(\d+)\.mlp\.experts\.(\d+)\.([^.]+)\.activation_scale", deal_in_scale()),
        (r"layers\.(\d+)\.mlp\.experts\.(\d+)\.([^.]+)\.weight_scale", deal_weight_scale()),
        (r"layers\.(\d+)\.mlp\.experts\.(\d+)\.([^.]+)\.quant_weight", reorder()),
    ]
    for pattern, fn in param_mapping:
        for key in list(state_dict.keys()):
            # print(f"deal {key}")
            match = re.search(pattern, key)
            if match:
                # print(f"{key} is match")
                weight_or_scale = state_dict.pop(key)
                if "weight_scale" in key:
                    in_scale_key = key.replace("weight_scale", "activation_scale")
                    in_scale = save_state_dict[in_scale_key]
                    save_state_dict[key] = fn(weight_or_scale, in_scale, moe_quant_type)
                elif "activation_scale" in key:
                    save_state_dict[key] = fn(weight_or_scale)
                else:
                    save_state_dict[key] = fn(weight_or_scale, moe_quant_type)


def save_safetensors(state_dict, args):
    """
    save_safetensors
    """
    logger.info("Move to numpy.")
    for k in list(state_dict.keys()):
        if isinstance(state_dict[k], paddle.Tensor):
            state_dict[k] = state_dict.pop(k).cpu().numpy()

    logger.info("Save safetensors files.")
    shards, index = shard_checkpoint(
        state_dict,
        max_shard_size="5GB",
        weights_name=SAFE_WEIGHTS_NAME,
        shard_format="naive",
    )
    for shard_file, shard in shards.items():
        save_file = os.path.join(args.output_dir, shard_file)
        logger.info(f"Saving {save_file}")
        safe_save_file(shard, save_file, metadata={"format": "np"})

    save_index_file = os.path.join(args.output_dir, SAFE_WEIGHTS_INDEX_NAME)
    with open(save_index_file, "w", encoding="utf-8") as f:
        content = json.dumps(index, indent=2) + "\n"
        f.write(content)


def main():
    """
    main
    """
    args = parse_arguments()
    pretrained_config, _ = PretrainedConfig.get_config_dict(args.model_name_or_path)
    pretrained_config = PretrainedConfig.from_dict(pretrained_config)
    vocab_file_names = [
        "tokenizer.model",
        "spm.model",
        "ernie_token_100k.model",
    ]
    for i in range(len(vocab_file_names)):
        if os.path.exists(os.path.join(args.model_name_or_path, vocab_file_names[i])):
            Ernie4_5Tokenizer.resource_files_names["vocab_file"] = vocab_file_names[i]
            break
    tokenizer = Ernie4_5Tokenizer.from_pretrained(args.model_name_or_path)
    _, safetensor_files, _ = get_all_weights_file(args.model_name_or_path)
    weights_iterator = safetensors_weights_iterator(safetensor_files)
    state_dict = {}
    save_state_dict = {}
    start = time.perf_counter()
    for k, v in weights_iterator:
        state_dict[k] = get_tensor(v).cpu()
    end = time.perf_counter()
    logger.info("Finish Quantize.")
    logger.info(f"load and quantize took : {end - start:.6f} seconds")
    deal_quant(state_dict, save_state_dict, args.moe_quant_type)
    for key in list(state_dict.keys()):
        save_state_dict[key] = state_dict.pop(key)
    logger.info("Begin to save model")
    os.makedirs(args.output_dir, exist_ok=True)
    start = time.perf_counter()
    if not args.safe_serialization:
        paddle.save(
            save_state_dict,
            os.path.join(args.output_dir, "model_state.pdparams"),
        )
    else:
        save_safetensors(save_state_dict, args)
    pretrained_config.is_permuted = True
    pretrained_config.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    end = time.perf_counter()
    logger.info(f"save model took: {end - start:.6f} seconds")
    logger.info("Finish.")


if __name__ == "__main__":
    main()
