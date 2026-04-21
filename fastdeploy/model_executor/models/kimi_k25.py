from __future__ import annotations

import re

import paddle
from paddleformers.transformers import PretrainedModel
from paddleformers.utils.log import logger

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.models.model_base import (ModelCategory, ModelRegistry)
from fastdeploy.model_executor.layers.moe.moe import FusedMoE
from fastdeploy.model_executor.models.deepseek_v3 import DeepseekV3ForCausalLM


@ModelRegistry.register_model_class(
    architecture="KimiK25ForConditionalGeneration",
    module_name="kimi_k25",
    category=ModelCategory.TEXT_GENERATION,
    primary_use=ModelCategory.TEXT_GENERATION,
)
class KimiK25ForConditionalGeneration(DeepseekV3ForCausalLM):
    def __init__(self, fd_config: FDConfig):
        super().__init__(fd_config)

    @classmethod
    def name(cls):
        return "KimiK25ForConditionalGeneration"
    
    @paddle.no_grad()
    def load_weights(self, weights_iterator) -> None:
        if self.fd_config.model_config.is_quantized:
            from fastdeploy.model_executor.utils import (
                default_weight_loader,
                get_tensor,
                process_weights_after_loading,
                slice_fn,
            )
            
            stacked_params_mapping = [
                ("up_gate_proj",                    "gate_proj",                    "gate"),
                ("up_gate_proj",                    "up_proj",                      "up"),
                ("embed_tokens.embeddings",         "embed_tokens",                 None),
                ("lm_head.linear",                  "language_model.lm_head",       None),
                ("experts.gate_correction_bias",    "gate.e_score_correction_bias", None),
                ("qkv_a_proj_with_mqa",             "q_a_proj",                     "q_a"),
                ("qkv_a_proj_with_mqa",             "kv_a_proj_with_mqa",           "kv_a"),
            ]
            params_dict = dict(self.named_parameters())
            process_weights_after_loading_fn = process_weights_after_loading(dict(self.named_sublayers()), self.fd_config)
            for loaded_weight_name, loaded_weight in weights_iterator:
                logger.debug(f"Loading weight: {loaded_weight_name}")
                loaded_weight_name = loaded_weight_name.replace("language_model.model", "model")
            
                # 第一层：stacked_params_mapping
                handled = False
                for param_name, weight_name, shard_id in stacked_params_mapping:
                    if weight_name not in loaded_weight_name:
                        continue
                    if "mlp.experts." in loaded_weight_name:
                        continue

                    model_param_name = loaded_weight_name.replace(weight_name, param_name)

                    if model_param_name not in params_dict:
                        continue

                    param = params_dict[model_param_name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader(self.fd_config))
                    weight_loader(param, loaded_weight, shard_id)

                    handled = True
                    break

                # 第二层：直接写入 experts 下的 stacked parameter
                if not handled and ".experts." in loaded_weight_name:
                    for proj_name in ("gate_proj", "up_proj", "down_proj"):
                        for suffix in ("weight_packed", "weight_scale"):
                            proj_match = re.match(
                                rf"^(.*\.experts\.)(\d+)\.{proj_name}\.{suffix}$",
                                loaded_weight_name,
                            )
                            if not proj_match:
                                continue

                            model_param_name = f"{proj_match.group(1)}{proj_name}_{suffix}"
                            expert_id = int(proj_match.group(2))

                            if model_param_name in params_dict:
                                param = params_dict[model_param_name]
                                if not param._is_initialized():
                                    param.initialize()
                                weight = get_tensor(loaded_weight)

                                # For TP loading, gate/up are sharded on the penultimate dim,
                                # while down is sharded on the last dim.
                                tp_size = self.fd_config.parallel_config.tensor_parallel_size
                                if tp_size > 1 and not self.fd_config.load_config.is_pre_sharded:
                                    split_on_last_dim = proj_name == "down_proj"
                                    split_dim = -1 if split_on_last_dim else 0
                                    split_size = weight.shape[split_dim]
                                    if split_size % tp_size != 0:
                                        raise ValueError(
                                            f"Cannot split {loaded_weight_name} for TP: "
                                            f"size={split_size}, tp_size={tp_size}"
                                        )
                                    block_size = split_size // tp_size
                                    tp_rank = self.fd_config.parallel_config.tensor_parallel_rank
                                    shard_start = tp_rank * block_size
                                    shard_end = (tp_rank + 1) * block_size
                                    weight = slice_fn(weight, split_on_last_dim, shard_start, shard_end)

                                expert_param = param[expert_id]
                                if expert_param.shape != weight.shape:
                                    raise ValueError(
                                        f"Shape mismatch when loading {loaded_weight_name}: "
                                        f"loaded={weight.shape}, param={expert_param.shape}"
                                    )
                                if expert_param.dtype != weight.dtype:
                                    weight = weight.cast(expert_param.dtype)
                                expert_param.set_value(weight)
                                handled = True
                            break

                # 第三层：默认逻辑
                if not handled:
                    model_param_name = loaded_weight_name

                    if model_param_name not in params_dict:
                        continue

                    param = params_dict[model_param_name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader(self.fd_config))
                    weight_loader(param, loaded_weight)
        
                model_sublayer_name = re.sub(r"\.(up_gate_proj_weight|down_proj_weight|weight)$", "", model_param_name)
                if "kv_b_proj" in model_sublayer_name:
                    kv_model_sublayer_name = model_sublayer_name.replace("kv_b_proj", "kv_b_proj_bmm")
                    process_weights_after_loading_fn(kv_model_sublayer_name)
                process_weights_after_loading_fn(model_sublayer_name, param)
        else:
            from fastdeploy.model_executor.utils import (
                default_weight_loader,
                process_weights_after_loading,
            )
            
            stacked_params_mapping = [
                # (param_name, shard_name, shard_id)
                ("up_gate_proj", "gate_proj", "gate"),
                ("up_gate_proj", "up_proj", "up"),
                ("embed_tokens.embeddings", "embed_tokens", None),
                ("lm_head.linear", "language_model.lm_head", None),
                ("experts.gate_correction_bias", "gate.e_score_correction_bias", None),
                ("qkv_a_proj_with_mqa", "q_a_proj", "q_a"),
                ("qkv_a_proj_with_mqa", "kv_a_proj_with_mqa", "kv_a"),
            ]
            # (param_name, weight_name, expert_id, shard_id)
            expert_params_mapping = FusedMoE.make_expert_params_mapping(
                num_experts=self.fd_config.model_config.n_routed_experts,
                ckpt_gate_proj_name="gate_proj",
                ckpt_down_proj_name="down_proj",
                ckpt_up_proj_name="up_proj",
                param_gate_up_proj_name="experts.up_gate_proj_",
                param_down_proj_name="experts.down_proj_",
            )
            params_dict = dict(self.named_parameters())
            process_weights_after_loading_fn = process_weights_after_loading(dict(self.named_sublayers()), self.fd_config)
            for loaded_weight_name, loaded_weight in weights_iterator:
                logger.debug(f"Loading weight: {loaded_weight_name}")
                loaded_weight_name = loaded_weight_name.replace("language_model.model", "model")
                for param_name, weight_name, shard_id in stacked_params_mapping:
                    if weight_name not in loaded_weight_name:
                        continue
                    if "mlp.experts." in loaded_weight_name:
                        continue
                    model_param_name = loaded_weight_name.replace(weight_name, param_name)

                    if model_param_name not in params_dict:
                        continue

                    param = params_dict[model_param_name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader(self.fd_config))
                    weight_loader(param, loaded_weight, shard_id)
                    break
                else:
                    for mapping in expert_params_mapping:
                        param_name, weight_name, expert_id, shard_id = mapping
                        if weight_name not in loaded_weight_name:
                            continue
                        model_param_name = loaded_weight_name.replace(weight_name, param_name)
                        if model_param_name not in params_dict:
                            continue
                        param = params_dict[model_param_name]
                        weight_loader = param.weight_loader
                        weight_loader(param, loaded_weight, shard_id=shard_id, expert_id=expert_id)
                        break
                    else:
                        model_param_name = loaded_weight_name
                        if model_param_name not in params_dict:
                            continue
                        param = params_dict[model_param_name]
                        weight_loader = getattr(param, "weight_loader", default_weight_loader(self.fd_config))
                        weight_loader(param, loaded_weight)

                model_sublayer_name = re.sub(r"\.(up_gate_proj_weight|down_proj_weight|weight)$", "", model_param_name)
                if "kv_b_proj" in model_sublayer_name:
                    kv_model_sublayer_name = model_sublayer_name.replace("kv_b_proj", "kv_b_proj_bmm")
                    process_weights_after_loading_fn(kv_model_sublayer_name)
                process_weights_after_loading_fn(model_sublayer_name, param)


class KimiK25PretrainedModel(PretrainedModel):
    config_class = FDConfig

    def _init_weight(self, layer):
        return None

    @classmethod
    def arch_name(self):
        return "KimiK25ForConditionalGeneration"