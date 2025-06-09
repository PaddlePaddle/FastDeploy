"""
# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import threading

import paddle
import paddle.distributed as dist
from paddle.common_ops_import import convert_dtype
from fastdeploy.model_executor.models.utils import convert_ndarray_dtype
from paddlenlp.trainer import RuntimeTimer
from fastdeploy.inference_args import GenerationPhase

from .utils import (
    _vocab_size_with_padding,
    generate_rank_mapping,
    get_infer_model_path,
    model_convert_fp8,
)
from paddlenlp.transformers import AutoTokenizer
from paddle.distributed import fleet
from paddlenlp.utils.env import USE_FAST_TOKENIZER
from paddlenlp.utils.log import logger
from fastdeploy.model_executor.models.utils import load_checkpoint

from fastdeploy.config import (AdditionalConfig, DecodingConfig, DeviceConfig,
                               LLMConfig, LoadConfig, ModelConfig, MoEConfig,
                               ParallelConfig, SpeculativeConfig, TmpConfig)
from fastdeploy.inference_args import GenerationPhase

from ..layers.quantization import get_quantization_config
from .model_base import ModelRegistry
from .qwen2 import Qwen2PretrainedModel
from .utils import (_vocab_size_with_padding, convert_ndarray_dtype,
                    load_checkpoint, parser_quant_type)
from paddlenlp.transformers.configuration_utils import PretrainedConfig
from paddlenlp.trl import llm_utils
model_classes_mapping = {
    "Qwen2ForCausalLM": Qwen2PretrainedModel,
}

current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.abspath(
    os.path.join(current_dir, os.pardir, os.pardir))
sys.path.append(grandparent_dir)


def offload_model(model):
    """
    Offload the model to CUDAPinnedPlace.
    """
    device = paddle.CUDAPinnedPlace()
    for name, src in model.named_parameters():
        if src._is_initialized() and not isinstance(src.place,
                                                    paddle.CUDAPinnedPlace):
            dst = src._copy_to(device, True)
            dst_tensor = dst.value().get_tensor()
            src_tensor = src.value().get_tensor()
            src_tensor._clear()
            src_tensor._share_data_with(dst_tensor)


def reload_model(model):
    """
    Reload the model from CUDAPinnedPlace to GPU.
    """
    model.to(paddle.device.get_device())


def reconstruct_memory(model):
    """
    reconstruct_memory to avoid memory chunks
    """
    offload_model(model)
    paddle.device.cuda.empty_cache()
    reload_model(model)


def load_tensor_from_ipc_meta(state_dict):
    """
    convert ipc_meta to tensor, but keep keys unchanged
    { 'key': ipc_meta } --> { 'key': tensor }
    example:
    state_dict = load_tensor_from_ipc_meta(state_dict)
    """
    for k, v in state_dict.items():
        # for pickling, we have to convert bytes object before save
        v[0] = v[0].encode("latin-1")
        state_dict[k] = paddle.to_tensor(
            paddle.base.core.LoDTensor._new_shared_cuda(tuple(v)))
    return state_dict


def build_stream_line_model(
    config_path,
    model_path,
    dtype,
    block_size,
    max_len,
    stage_flag,
    min_dec_len=1,
    max_dec_len=128,
    temperature=1,
    top_k=8,
    top_p=0.8,
    pre_caches_length=0,
    export_model_type="default",
    use_stop_seqs=False,
    use_fake_parameter=False,
    show_topk: int = 0,
    msg_queue_id=None,
    pad_vocab=True,
    tokenizer=None,
    cache_quant_dtype="default",
    use_beam_search: bool = False,
    enf_gen: bool = False,
    speculate_method=None,
    speculate_max_draft_token_num: int = 1,
    speculate_max_candidate_len: int = 5,
    speculate_verify_window: int = 2,
    return_all_hidden_states: bool = False,
    draft_type: str = "None",
    start_layer_index: int = 0,
    moe_quant_type: str = "default",
    use_ep: bool = False,
    ep_just_for_test: bool = False,
    generation_phase: GenerationPhase = GenerationPhase.PREFILL,
    use_micro_batch: bool = False,
    fake_server_p: bool = False,
    scale_dir: str = "None",
    output_via_mq: bool = True,
    use_safetensors: bool = False,
    enable_redundant_experts: bool = False,
    redundant_experts_num: int = 0,
    max_batch_size: int = 128,
    use_offline_quant: bool = False,
    return_state_dicts: bool = False,
    sharing_model=None,
    sharing_state_dicts=None,
):
    """
    Build a fused inference model

    Args:
        config_path (str): Path to the configuration file
        model_path (str): Path to the model file
        dtype (str): Data type of the model
        block_size (int): Block size
        max_len (int): Maximum sequence length
        stage_flag (str): Qianfan requirement, stage flag, used to identify different stages in \
            time-consuming statistics logs, such as prediction ("msgid-1 predict") or export ("convert").
        min_dec_len (int, optional): Minimum decoding length. Default is 1.
        max_dec_len (int, optional): Maximum decoding length. Default is 128.
        temperature (float, optional): Temperature coefficient. Default is 1.
        top_k (int, optional): k value in top-k sampling. Default is 0.
        top_p (float, optional): p value in top-p sampling. Default is 0.8.
        pre_caches_length (int, optional): Pre-cache length. Default is 0.
        export_model_type (str, optional): Type of model to export. Default is "default".
        use_stop_seqs (bool, optional): Whether to use stop sequences. Default is False.
        use_fake_parameter (bool, optional): Whether to use fake parameters. Default is False.
        show_topk (int, optional): Whether to show top-k results. Default is 0.
        msg_queue_id (int, optional): Message queue ID. Default is None.
        pad_vocab (bool, optional): Whether to pad the vocabulary. Default is True.
        cache_quant_dtype (str, optional): Cache quantization data type. Default is "default".
        use_beam_search (bool, optional): Whether to use beam search . Defaults is False.
        enf_gen (bool, optional): Whether to use enforce generation. Defaults is False.
    Returns:
        tuple[dict, Tokenizer, CausalLM]:
        A tuple containing the configuration, tokenizer, and model.
    """
    runtime_timer = RuntimeTimer("build_model")
    runtime_timer.start(f"{stage_flag} stage model loading time")

    # config_path = os.path.join(model_path,"config.json")
    with open(config_path, "r") as fin:
        config = json.load(fin)
    architectures = config.get("architectures")
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side="left",
            use_fast=USE_FAST_TOKENIZER,
        )

    config, _ = PretrainedConfig.get_config_dict(model_path)
    model_config = ModelConfig.from_dict(config)

    parallel_config = ParallelConfig()
    speculative_config = SpeculativeConfig()
    device_config = DeviceConfig()
    additional_config = AdditionalConfig()
    load_config = LoadConfig()
    tmp_config = TmpConfig()
    moe_config = MoEConfig()
    decoding_config = DecodingConfig()

    tensor_parallel_rank, tensor_parallel_degree = llm_utils.init_dist_env()
    parallel_config.tensor_parallel_rank = tensor_parallel_rank
    parallel_config.tensor_parallel_degree = tensor_parallel_degree
    parallel_config.mp_size = tensor_parallel_degree
    parallel_config.ep_size = 1
    parallel_config.column_cut = False

    speculative_config.is_mtp = draft_type in ["eagle", "mtp"]
    speculative_config.draft_type = draft_type

    # Note(tangbinhan): used for load_checkpoint
    model_config.tensor_parallel_rank = parallel_config.tensor_parallel_rank
    model_config.tensor_parallel_degree = parallel_config.tensor_parallel_degree
    model_config.use_ep = use_ep
    model_config.is_mtp = speculative_config.is_mtp

    additional_config.use_fake_parameter = use_fake_parameter
    additional_config.ep_just_for_test = ep_just_for_test

    tmp_config.use_offline_quant = use_offline_quant
    if use_ep:
        if isinstance(model_config.moe_num_experts, list):
            model_config.has_multimodality = True
            moe_config.num_experts = model_config.moe_num_experts[0]
        else:
            moe_config.num_experts = model_config.moe_num_experts
        moe_config.num_experts_per_rank = (
            moe_config.num_experts // parallel_config.tensor_parallel_degree
        )
        moe_config.num_experts_start_offset = (
            moe_config.num_experts_per_rank * parallel_config.tensor_parallel_rank
        )

    # use the length of tokenizer as the origin vocab size
    ori_vocab_size = len(tokenizer)
    moe_intermediate_size = (config.get("moe_intermediate_size", None),)
    if isinstance(moe_intermediate_size, list) or isinstance(
        moe_intermediate_size, tuple
    ):
        moe_intermediate_size = moe_intermediate_size[0]

    if not use_ep and pad_vocab:
        config["vocab_size"] = _vocab_size_with_padding(
            config.get("vocab_size", tokenizer.vocab_size),
            config.pop("vocab_size_divisible_unit", 128),
            paddle.distributed.get_world_size(),
        )

    group_size = config.get("group_size", -1)
    num_key_value_heads = config.get("num_key_value_heads", -1)
    if num_key_value_heads is None:
        num_key_value_heads = -1

    if config.get("ffn_hidden_size", None) is not None:
        ffn_hidden_size = config["ffn_hidden_size"]
    elif config.get("intermediate_size", None) is not None:
        ffn_hidden_size = config["intermediate_size"]
    else:
        ffn_hidden_size = 4 * config["hidden_size"]
        if config["hidden_act"].lower() == "swiglu":
            if paddle.distributed.get_world_size() > 1:
                multiple_of = 8 * config["num_attention_heads"]
            else:
                multiple_of = 4 * config["num_attention_heads"]
            ffn_hidden_size = multiple_of * (
                (int(2 * ffn_hidden_size / 3) + multiple_of - 1) //
                multiple_of)

    if draft_type in ["mtp", "eagle"]:
        num_layers = 1
    else:
        num_layers = config.get("num_layers", None) or config.get(
            "num_hidden_layers", None
        )
    if num_layers is None:
        raise ValueError(f"num_layers<{num_layers}> is invalid")

    use_moe = config.get(
        "moe_layer_start_index", num_layers
    ) < num_layers or draft_type in ["mtp", "eagle"]

    if not sharing_state_dicts:
        if use_fake_parameter:
            context = contextlib.nullcontext()
        elif use_safetensors:
            context = paddle.LazyGuard()
            model_class = model_classes_mapping[architectures[0]]
            state_dict = load_checkpoint(model_path,
                                        model_class,
                                        model_config,
                                        return_numpy=True)
        elif use_moe:
            tensor_parallel_degree = dist.get_world_size()
            if tensor_parallel_degree > 1:
                hcg = fleet.get_hybrid_communicate_group()
                mp_id = hcg.get_model_parallel_rank()
                # 统计文件子目录数量
                subdir_count = 0
                for entry in os.listdir(model_path):
                    if "pp" in entry:
                        full_path = os.path.join(model_path, entry)
                        if os.path.isdir(full_path):
                            subdir_count += 1

                pp_num = subdir_count
                rank_model_paths = [
                    os.path.join(model_path, f"pp{i}/model_state.tp0{mp_id}.pdparams")
                    for i in range(pp_num)
                ]

            context = paddle.LazyGuard()
            if not use_ep:
                logger.info(f"start to loading weight: {rank_model_paths}")
                state_dicts = [None for _ in rank_model_paths]

                def load_ckpt(i):
                    state_dicts[i] = paddle.load(rank_model_paths[i], return_numpy=True)

                threads = []
                for i in range(len(rank_model_paths)):
                    thread = threading.Thread(target=load_ckpt, args=(i,))
                    threads.append(thread)
                    thread.start()

                for t in threads:
                    t.join()

                logger.info("Loading finished")

            else:
                # for EP loading state_dicts
                import glob

                state_dicts = []
                files = glob.glob(model_path + "/merged_tp1_state_split/*")
                for file_name in files:
                    try:
                        state_dicts += [
                            {file_name.split("/")[-1]: file_name}
                        ]  # save {layer_name: weight_file_name}
                    except Exception:
                        pass

            need_reset_moe_intermediate_size = False
            if not use_ep:
                logger.info(f"moe_intermediate_size is: {moe_intermediate_size}")
                need_reset_moe_intermediate_size = (
                    (not use_ep)
                    and (moe_quant_type == "fp8")
                    and (moe_intermediate_size // 8 % 128 != 0)
                )
                ori_up_size = moe_intermediate_size // 8 * 2
                ori_down_size = ori_up_size // 2
                if need_reset_moe_intermediate_size:
                    moe_intermediate_size = (
                        128 - moe_intermediate_size // 8 % 128
                    ) * 8 + moe_intermediate_size
                    logger.info(
                        f"moe_intermediate_size reset to {moe_intermediate_size}!"
                    )
                    up_size = moe_intermediate_size // 8 * 2
                    down_size = up_size // 2
            new_state_dict = {}

            def padding(key, value):
                import numpy as np

                # logger.info(f"deal {key}")
                if ("experts" in key) and ("up_gate_proj" in key):
                    # logger.info("up_gate_proj")
                    v_new = np.zeros(shape=[value.shape[0], up_size], dtype=value.dtype)
                    v_new[:, :ori_down_size] = value[:, :ori_down_size]
                    v_new[:, down_size : (down_size + ori_down_size)] = value[
                        :, ori_down_size:
                    ]
                elif ("experts" in key) and ("down_proj" in key):
                    # logger.info("down_proj")
                    v_new = np.zeros(
                        shape=[down_size, value.shape[1]], dtype=value.dtype
                    )
                    v_new[:ori_down_size, :] = value
                else:
                    v_new = value
                new_state_dict[key] = v_new
                if ("experts" in key) and ("up_gate_proj" in key or "down_proj" in key):
                    pass
                    # logger.info(f"padding {key}: {value.shape}->{v_new.shape}")

            threads = []
            for state_dict in state_dicts:
                for key, value in state_dict.items():
                    if need_reset_moe_intermediate_size:
                        thread = threading.Thread(target=padding, args=(key, value))
                        threads.append(thread)
                        thread.start()
                    else:
                        new_state_dict[key] = value

            for t in threads:
                t.join()
            logger.info("Finish padding")
            state_dict = new_state_dict
        elif config.get("quant_type", None) is not None:
            # TODO(@wangbojun) currently, we use paddle.load for ptq model.
            tensor_parallel_degree = dist.get_world_size()
            if tensor_parallel_degree > 1:
                hcg = fleet.get_hybrid_communicate_group()
                mp_id = hcg.get_model_parallel_rank()
                rank_model_path = os.path.join(
                    model_path, f"model_state.tp0{mp_id}.pdparams"
                )
                if not os.path.exists(rank_model_path):
                    full_model_path = os.path.join(model_path, "model_state.pdparams")
                    if not os.path.exists(full_model_path):
                        raise ValueError(
                            f"can not find <model_state.tp0{mp_id}.pdparams> "
                            + f"and model_state.pdparams under dir<{model_path}>"
                        )
                    raise ValueError(
                        "please run `split_weights.py` to gen weights for multi-gpu inference."
                    )
                if not os.path.exists(rank_model_path):
                    full_model_path = os.path.join(model_path, "model_state.pdparams")
                    if not os.path.exists(full_model_path):
                        raise ValueError(
                            f"can not find <model_state.tp0{mp_id}.pdparams> "
                            + f"and model_state.pdparams under dir<{model_path}>"
                        )
                    raise ValueError(
                        "please run `split_weights.py` to gen weights for multi-gpu inference."
                    )
                model_state_path = rank_model_path
                if num_key_value_heads > 0:
                    assert (
                        num_key_value_heads % tensor_parallel_degree == 0
                    ), "num_key_value_heads must be an integer multiple of tensor_parallel_degree"
            else:
                model_state_path = os.path.join(model_path, "model_state.pdparams")
            context = paddle.LazyGuard()
            logger.info(f"start to loading weight: {model_state_path}")
            if os.path.exists(model_state_path):
                state_dict = paddle.load(model_state_path, return_numpy=True)
    else:
        state_dict = sharing_state_dicts
        context = paddle.LazyGuard()

    use_rmsnorm = config.get("use_rmsnorm", True)

    if use_beam_search:
        decode_strategy = "beam_search"
    elif speculate_method is not None:
        if draft_type in ["draft_model", "eagle", "mtp"]:
            decode_strategy = "draft_model_sampling"
        else:
            decode_strategy = "speculate_decoding"
    else:
        decode_strategy = "sampling"

    logger.info(f"{runtime_timer.log()}")
    runtime_timer.start(f"{stage_flag} stage set parameters time")

    if config["hidden_act"].lower() == "swiglu":
        model_config.hidden_act = "swiglu"
    model_config.ffn_hidden_size = ffn_hidden_size
    model_config.max_seq_len = max_len
    model_config.num_layers = num_layers
    model_config.dtype = dtype
    model_config.export_model_type = export_model_type
    parallel_config.block_size = block_size

    model_config.group_size = group_size
    load_config.model_path = model_path
    model_config.use_rmsnorm = use_rmsnorm
    parallel_config.msg_queue_id = msg_queue_id
    additional_config.use_fake_parameter = use_fake_parameter
    model_config.num_key_value_heads = num_key_value_heads
    model_config.use_stop_seqs = use_stop_seqs
    tmp_config.cache_quant_dtype = cache_quant_dtype
    tmp_config.has_zero_point = config.get("has_zero_point", False)
    tmp_config.is_channel_wise = config.get("is_channel_wise", False),
    speculative_config.speculate_method = speculate_method
    speculative_config.speculate_max_draft_token_num = speculate_max_draft_token_num
    model_config.return_all_hidden_states = return_all_hidden_states
    speculative_config.draft_type = draft_type
    model_config.start_layer_index = start_layer_index
    model_config.use_moe = use_moe
    if use_moe:
        moe_config.use_moe = use_moe
        moe_config.num_experts = config.get("moe_num_experts", None)
        moe_config.moe_intermediate_size = config.get("moe_intermediate_size",
                                                    None)
        moe_config.moe_use_gate_correction_bias = config.get(
            "moe_use_gate_correction_bias", True)
        moe_config.moe_every2 = config.get("moe_every2", False)
        moe_config.moe_topk = config.get("moe_topk", 8)
        moe_config.moe_num_shared_experts = config.get("moe_num_shared_experts", 0)
        moe_config.moe_layer_start_index = config.get("moe_layer_start_index", 0)
        moe_config.moe_use_ffn_shared_weight_and_bias = config.get(
            "moe_use_ffn_shared_weight_and_bias", False)
        moe_config.use_moe = use_moe
        moe_config.moe_group = config.get("moe_group", False)
        moe_config.moe_quant_type = moe_quant_type
        if top_k > 0:
            moe_config.top_k = top_k
    parallel_config.use_ep = use_ep
    additional_config.ep_just_for_test = ep_just_for_test
    model_config.generation_phase = generation_phase
    parallel_config.use_micro_batch = use_micro_batch
    tmp_config.weight_block_size = config.get("weight_block_size", [-1, -1])
    load_config.scale_dir = scale_dir
    model_config.output_via_mq = output_via_mq


    decoding_config.bos_token_id = tokenizer.bos_token_id
    decoding_config.pad_token_id = tokenizer.pad_token_id
    decoding_config.temperature = temperature
    decoding_config.forced_eos_token_id = tokenizer.eos_token_id
    model_config.ori_vocab_size = ori_vocab_size
    decoding_config.max_dec_len = max_dec_len
    decoding_config.min_dec_len = min_dec_len
    additional_config.fake_server_p = fake_server_p
    decoding_config.decode_strategy = decode_strategy
    speculative_config.speculate_max_candidate_len = speculate_max_candidate_len
    speculative_config.speculate_verify_window = speculate_verify_window

    weight_dtype, act_dtype, cachekv_dtype = parser_quant_type(
        export_model_type)
    logger.info(
        f"quant_type: weight[{weight_dtype}], act[{act_dtype}], cachekv[{cachekv_dtype}]"
    )
    model_config.weight_dtype = weight_dtype
    model_config.act_dtype = act_dtype

    if weight_dtype == "int8" and act_dtype in ["bfloat16", "float16"]:
        quant_cls = get_quantization_config("weight_only")
        quant_config = quant_cls.from_config({
            "weight_only_linear_arch": None,
            "algo": "weight_only_int8"
        })
        quant_config.quant_max_bound = 0
        quant_config.quant_min_bound = 0
        quant_config.quant_round_type = 0
        model_config.use_smooth_quant = False
    elif weight_dtype == "int4" and act_dtype in ["bfloat16", "float16"]:
        quant_cls = get_quantization_config("weight_only")
        quant_config = quant_cls.from_config({
            "weight_only_linear_arch": None,
            "algo": "weight_only_int4"
        })
        quant_config.quant_max_bound = 0
        quant_config.quant_min_bound = 0
        quant_config.quant_round_type = 0
        model_config.use_smooth_quant = False
    elif tmp_config.weight_block_size[0] != -1:
        quant_cls = get_quantization_config("block_wise")
        quant_config = quant_cls.from_config(
            {"weight_block_size": tmp_config.weight_block_size})
        quant_config.quant_max_bound = 448
        quant_config.quant_min_bound = -448
        quant_config.quant_round_type = 1
        model_config.use_smooth_quant = False
    elif weight_dtype == "int4" and act_dtype == "float8_e4m3fn":
        quant_cls = get_quantization_config("w4afp8")
        quant_config = quant_cls.from_config({
            "weight_scale_dict": {},
            "act_scale_dict": {}
        })
        quant_config.quant_max_bound = 448
        quant_config.quant_min_bound = -448
        quant_config.quant_round_type = 1
        model_config.use_smooth_quant = False
    elif weight_dtype == "int8" and act_dtype == weight_dtype:
        quant_cls = get_quantization_config("w8a8")
        quant_config = quant_cls.from_config({
            "weight_scale_dict": {},
            "act_scale_dict": {},
            "use_gemm_dequant": False
        })
        quant_config.quant_max_bound = 127
        quant_config.quant_min_bound = -127
        quant_config.quant_round_type = 0
        model_config.use_smooth_quant = True
    elif weight_dtype == "float8_e4m3fn" and act_dtype == weight_dtype:
        quant_cls = get_quantization_config("wfp8afp8")
        quant_config = quant_cls.from_config({
            "weight_scale_dict": {},
            "act_scale_dict": {}
        })
        quant_config.quant_max_bound = 448
        quant_config.quant_min_bound = -448
        quant_config.quant_round_type = 1
        model_config.use_smooth_quant = False
    else:
        quant_config = None

    llm_config = LLMConfig(
        model_config=model_config,
        parallel_config=parallel_config,
        speculative_config=speculative_config,
        device_config=device_config,
        additional_config=additional_config,
        load_config=load_config,
        tmp_config=tmp_config,
        moe_config=moe_config,
        decoding_config=decoding_config,
        quant_config=quant_config,
    )

    with context:
        model_cls = ModelRegistry.get_class(model_config.architectures[0])
        model = model_cls(llm_config)

    model.eval()

    if use_fake_parameter:
        return config, tokenizer, model
    elif not use_moe:
        for k, v in state_dict.items():
            if convert_dtype(v.dtype) == dtype:
                continue
            elif convert_dtype(v.dtype) == "float32":
                continue
            state_dict[k] = convert_ndarray_dtype(v, dtype)

    paddle.device.cuda.empty_cache()
    assert state_dict is not None
    model.set_state_dict(state_dict)
    if use_ep and generation_phase == GenerationPhase.DECODER:
        logger.info("Reloading model...")
        reconstruct_memory(model)
    logger.info(f"{runtime_timer.log()}")

    if sharing_state_dicts is not None:
        for k in list(sharing_state_dicts):
            sharing_state_dicts.pop(k)

    possible_state_dict = state_dict if return_state_dicts else None
    return config, tokenizer, model, possible_state_dict
