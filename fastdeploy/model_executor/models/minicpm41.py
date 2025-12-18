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

from typing import Dict, Optional, Tuple

import paddle
import paddle.nn.functional as F
from paddle import nn

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.forward_meta import ForwardMeta
from fastdeploy.model_executor.layers.attention.attention import Attention
from fastdeploy.model_executor.layers.embeddings import VocabParallelEmbedding
from fastdeploy.model_executor.layers.linear import RowParallelLinear
from fastdeploy.model_executor.layers.lm_head import ParallelLMHead
from fastdeploy.model_executor.layers.normalization import RMSNorm
from fastdeploy.model_executor.models.model_base import (
    ModelCategory,
    ModelForCasualLM,
    ModelRegistry,
)


@ModelRegistry.register_model_class(
    architecture="MiniCPMForCausalLM",
    module_name="minicpm41",
    category=ModelCategory.TEXT_GENERATION,
    primary_use=ModelCategory.TEXT_GENERATION,
)
class MiniCPMForCausalLM(ModelForCasualLM):
    """
    MiniCPM4.1-8B model for FastDeploy

    This model implements the MiniCPM4.1-8B architecture with:
    - Trainable sparse attention mechanism (InfLLM v2)
    - Mixed reasoning mode support
    - Long context support (up to 64K tokens)
    - Various quantization support
    """

    def __init__(self, fd_config: FDConfig):
        super().__init__(fd_config)
        self.fd_config = fd_config
        self.model_config = fd_config.model_config

        # Pre-validate and fix model configuration before initializing layers
        self._fix_model_config()
        # breakpoint()
        # Embedding layer
        self.embed_tokens = VocabParallelEmbedding(
            fd_config=fd_config,
            num_embeddings=self.model_config.vocab_size,
            embedding_dim=self.model_config.hidden_size,
            prefix="model.embed_tokens",
            general=True,
        )
        # Decoder layers
        self.layers = nn.LayerList(
            [MiniCPM41DecoderLayer(fd_config, layer_id=i) for i in range(self.model_config.num_hidden_layers)]
        )

        # Normalization layer
        self.norm = RMSNorm(
            fd_config,
            hidden_size=self.model_config.hidden_size,
            eps=self.model_config.rms_norm_eps,
            prefix="model.norm",
            begin_norm_axis=-1,
        )

        # LM head for causal LM
        self.lm_head = ParallelLMHead(
            fd_config=fd_config,
            embedding_dim=self.model_config.hidden_size,
            num_embeddings=self.model_config.vocab_size,
            prefix="lm_head",
        )
        # Sparse attention configuration
        self.sparse_config = getattr(self.model_config, "sparse_config", None)

        # Rope scaling configuration
        self.rope_scaling = getattr(self.model_config, "rope_scaling", None)

        self.config = fd_config.model_config

    def _fix_model_config(self):
        """Fix model configuration to match MiniCPM4.1-8B config.json BEFORE initializing layers"""
        print("🔧 Pre-initializing Model Configuration:")

        # Key configuration parameters from config.json
        config_params = {
            "vocab_size": 73448,
            "hidden_size": 4096,
            "intermediate_size": 16384,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "num_key_value_heads": 2,
            "max_position_embeddings": 65536,
            "rms_norm_eps": 1e-06,
            "tie_word_embeddings": False,
            "rope_theta": 10000.0,
            "scale_emb": 12,
            "scale_depth": 1.4,
            "mup_denominator": 32,
            "dim_model_base": 256,
        }

        # Update all model config parameters
        for param_name, expected_value in config_params.items():
            actual_value = getattr(self.model_config, param_name, None)
            if actual_value != expected_value:
                setattr(self.model_config, param_name, expected_value)
                print(f"   ✓ Set {param_name}: {expected_value}")
            else:
                print(f"   ✓ {param_name}: {actual_value}")

        # Add rope_scaling configuration
        rope_scaling = {
            "rope_type": "longrope",
            "long_factor": [0.9982316082870437, 1.033048153422584, 1.0749920956484724],
            "short_factor": [0.9982316082870437, 1.033048153422584, 1.0749920956484724],
            "original_max_position_embeddings": 65536,
        }
        setattr(self.model_config, "rope_scaling", rope_scaling)
        print("   ✓ Added rope_scaling: longrope")

        print("   🎉 Model configuration fixed for MiniCPM4.1-8B!")

    @classmethod
    def name(cls) -> str:
        """ """
        return "MiniCPMForCausalLM"

    def forward(
        self,
        input_ids: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        """
        Forward pass of MiniCPM4.1-8B model

        Args:
            input_ids: Token input IDs
            attention_mask: Attention mask
            position_ids: Position IDs
            past_key_values: Past key-value cache
            inputs_embeds: Input embeddings (alternative to input_ids)
            use_cache: Whether to use key-value cache
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return dict format
        """
        # breakpoint()
        return_dict = return_dict if return_dict is not None else True
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # Handle embeddings
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_len, hidden_size = inputs_embeds.shape

        # Handle position IDs if not provided
        if position_ids is None:
            position_ids = paddle.arange(seq_len, dtype="int64").expand([batch_size, seq_len])

        # Initialize outputs
        hidden_states = inputs_embeds
        presents = [] if use_cache else None
        all_hidden_states = [] if output_hidden_states else None
        all_self_attentions = [] if output_attentions else None

        # Process through decoder layers
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            # Get past key-value for current layer
            layer_past = past_key_values[idx] if past_key_values is not None else None

            # Forward through decoder layer
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=layer_past,
                use_cache=use_cache,
                output_attentions=output_attentions,
                sparse_config=self.sparse_config,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                presents.append(layer_outputs[1])

            if output_attentions:
                all_self_attentions.append(layer_outputs[2])

        # Final normalization
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        # Compute logits
        logits = self.lm_head(hidden_states)

        if not return_dict:
            return tuple(
                v
                for v in [
                    logits,
                    presents,
                    all_hidden_states,
                    all_self_attentions,
                ]
                if v is not None
            )

        return ForwardMeta(
            last_hidden_state=hidden_states,
            logits=logits,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    def compute_logits(self, hidden_states: paddle.Tensor, **kwargs) -> paddle.Tensor:
        """Compute logits from hidden states"""
        return self.lm_head(hidden_states)

    def prepare_inputs_for_generation(
        self,
        input_ids: paddle.Tensor,
        past_key_values: Optional[Tuple] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        **kwargs,
    ) -> Dict[str, paddle.Tensor]:
        """
        Prepare inputs for generation step
        """
        # Get the past length
        past_length = 0
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

        # If only one token is generated, prepare for next step
        if attention_mask is not None and input_ids.shape[1] > 1:
            # Trim attention mask for next token
            attention_mask = attention_mask[:, -1:]

        if position_ids is None:
            # Calculate position IDs based on past length
            if past_key_values is not None:
                position_ids = paddle.full((input_ids.shape[0], 1), past_length, dtype="int64")
            else:
                position_ids = paddle.arange(input_ids.shape[1], dtype="int64").expand(
                    [input_ids.shape[0], input_ids.shape[1]]
                )

        # Return prepared inputs
        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
        }

    def load_weights(self, weights_iterator):
        """
        Load weights from iterator with precise mapping for MiniCPM4.1-8B

        Args:
            weights_iterator: Iterator yielding (param_name, param_value) tuples
        """
        # Precise parameter mapping based on model.safetensors.index.json
        param_mapping = {
            # Attention layers - separate Q, K, V projections
            "model.layers.{i}.self_attn.q_proj.weight": "layers.{i}.self_attn.q_proj.weight",
            "model.layers.{i}.self_attn.k_proj.weight": "layers.{i}.self_attn.k_proj.weight",
            "model.layers.{i}.self_attn.v_proj.weight": "layers.{i}.self_attn.v_proj.weight",
            "model.layers.{i}.self_attn.o_proj.weight": "layers.{i}.self_attn.o_proj.weight",
            # MLP layers
            "model.layers.{i}.mlp.gate_proj.weight": "layers.{i}.mlp.gate_proj.weight",
            "model.layers.{i}.mlp.up_proj.weight": "layers.{i}.mlp.up_proj.weight",
            "model.layers.{i}.mlp.down_proj.weight": "layers.{i}.mlp.down_proj.weight",
            # Layer normalization
            "model.layers.{i}.input_layernorm.weight": "layers.{i}.input_layernorm.weight",
            "model.layers.{i}.post_attention_layernorm.weight": "layers.{i}.post_attention_layernorm.weight",
            # Embeddings and output layers - VocabParallelEmbedding and ParallelLMHead have different internal structure
            "model.embed_tokens.weight": "embed_tokens.embeddings.weight",
            "model.norm.weight": "norm.weight",
            "lm_head.weight": "lm_head.linear.weight",
        }

        # Debug model structure (configuration already fixed in __init__)
        self._debug_model_paths()

        loaded_count = 0
        failed_params = []
        # breakpoint()
        for name, loaded_weight in weights_iterator:
            # Special handling for embed_tokens and lm_head - try multiple paths
            if name == "model.embed_tokens.weight":
                # Check and fix vocabulary size mismatch
                actual_vocab_size = loaded_weight.shape[0]
                expected_vocab_size = self.model_config.vocab_size

                if actual_vocab_size != expected_vocab_size:
                    print("⚠️  Vocabulary size mismatch:")
                    print(f"   Expected: {expected_vocab_size}")
                    print(f"   Actual: {actual_vocab_size}")

                    # Update model config to match weights
                    self.model_config.vocab_size = actual_vocab_size
                    print(f"   ✓ Updated model vocab_size to {actual_vocab_size}")

                # Try different possible paths for embed_tokens
                paths_to_try = ["embed_tokens.embeddings.weight", "embed_tokens.weight"]
                loaded_successfully = False
                for target_name in paths_to_try:
                    param = self._get_parameter_by_path(target_name)
                    if param is not None:
                        try:
                            param.set_value(loaded_weight)
                            loaded_count += 1
                            print(f"✓ Loaded embed_tokens weight as {target_name} (shape: {loaded_weight.shape})")
                            loaded_successfully = True
                            break
                        except Exception as e:
                            print(f"✗ Error setting embed_tokens weight as {target_name}: {e}")

                if not loaded_successfully:
                    print("✗ Failed to load embed_tokens weight")
                    failed_params.append(name)
                continue

            elif name == "lm_head.weight":
                # Fix lm_head weight transpose issue
                # Weight files often store embeddings as [vocab_size, hidden_size]
                # but linear layers expect [hidden_size, vocab_size]
                print(f"📝 Processing lm_head weight (original shape: {loaded_weight.shape})")

                # Check if we need to transpose
                lm_head_param = self._get_parameter_by_path("lm_head.linear.weight")
                if lm_head_param is not None:
                    expected_shape = lm_head_param.shape
                    print(f"   Expected shape: {expected_shape}")

                    # Transpose if needed: [vocab_size, hidden_size] -> [hidden_size, vocab_size]
                    if loaded_weight.shape != expected_shape:
                        print("   🔄 Transposing lm_head weight")
                        loaded_weight = loaded_weight.transpose([1, 0])
                        print(f"   ✓ New shape: {loaded_weight.shape}")

                # Try different possible paths for lm_head
                paths_to_try = ["lm_head.linear.weight", "lm_head.weight"]
                loaded_successfully = False
                for target_name in paths_to_try:
                    param = self._get_parameter_by_path(target_name)
                    if param is not None:
                        try:
                            param.set_value(loaded_weight)
                            loaded_count += 1
                            print(f"✓ Loaded lm_head weight as {target_name} (shape: {loaded_weight.shape})")
                            loaded_successfully = True
                            break
                        except Exception as e:
                            print(f"✗ Error setting lm_head weight as {target_name}: {e}")

                if not loaded_successfully:
                    print("✗ Failed to load lm_head weight")
                    failed_params.append(name)
                continue

            # Map parameter name from safetensors to FastDeploy format
            target_name = self._map_param_name(name, param_mapping)

            # Get parameter by path
            param = self._get_parameter_by_path(target_name)

            if param is not None:
                try:
                    param.set_value(loaded_weight)
                    loaded_count += 1
                except Exception as e:
                    print(f"Error setting weight for {target_name}: {e}")
                    failed_params.append(target_name)
            else:
                print(f"Warning: Parameter {target_name} not found")
                failed_params.append(target_name)

        print("\n📊 Weight Loading Summary:")
        print(f"   ✓ Successfully loaded: {loaded_count} parameters")
        if failed_params:
            print(f"   ❌ Failed to load: {len(failed_params)} parameters")
            print("   Failed parameters:")
            for param in failed_params[:5]:  # Show first 5 failed parameters
                print(f"     - {param}")
            if len(failed_params) > 5:
                print(f"     ... and {len(failed_params) - 5} more")
        else:
            print("   🎉 All parameters loaded successfully!")

        # Final compatibility check
        print("\n🔍 Final Compatibility Check:")
        print(f"   Model vocab_size: {self.model_config.vocab_size}")
        if hasattr(self, "embed_tokens") and hasattr(self.embed_tokens, "embeddings"):
            print(f"   embed_tokens shape: {self.embed_tokens.embeddings.weight.shape}")
        if hasattr(self, "lm_head") and hasattr(self.lm_head, "linear"):
            print(f"   lm_head linear shape: {self.lm_head.linear.weight.shape}")
        print(f"   Total layers: {len(self.layers)}")
        # breakpoint()
        # Critical: If any parameters failed to load, raise an error to prevent segfault
        if failed_params:
            critical_params = [p for p in failed_params if "embed_tokens" in p or "lm_head" in p]
            if critical_params:
                print(f"\n❌ CRITICAL ERROR: Failed to load critical parameters: {critical_params}")
                print("   This may cause segmentation faults during inference.")
                print("   Please check model configuration and weight file compatibility.")
                raise RuntimeError(f"Failed to load critical parameters: {critical_params}")
            else:
                print("\n⚠️  WARNING: Some non-critical parameters failed to load.")
                print("   Model may still function with degraded performance.")

    def _map_param_name(self, orig_name: str, param_mapping: dict) -> str:
        """Map parameter name from safetensors format to FastDeploy format"""
        import re

        # Handle layer-specific parameters
        for pattern, template in param_mapping.items():
            if "{i}" in pattern:
                # Replace {i} with regex pattern to match layer numbers
                regex_pattern = pattern.replace("{i}", r"(\d+)")
                match = re.fullmatch(regex_pattern, orig_name)
                if match:
                    layer_id = match.group(1)
                    return template.replace("{i}", layer_id)
            elif orig_name == pattern:
                return template

        # Default behavior: remove "model." prefix if present
        if orig_name.startswith("model."):
            return orig_name[6:]

        return orig_name

    def _get_parameter_by_path(self, param_path: str):
        """Get parameter object by navigating through the model structure"""
        parts = param_path.split(".")
        current = self

        try:
            for part in parts:
                # Handle list access like "layers[0]"
                if "[" in part and part.endswith("]"):
                    attr_name = part.split("[")[0]
                    index = int(part.split("[")[1].split("]")[0])
                    current = getattr(current, attr_name)[index]
                else:
                    current = getattr(current, part)

            # Return the weight parameter
            if hasattr(current, "weight"):
                return current.weight
            elif hasattr(current, "set_value"):
                return current
            else:
                return None

        except (AttributeError, IndexError, KeyError):
            return None

    def _debug_model_paths(self):
        """Debug function to check actual parameter paths in the model"""
        print("=== Model Parameter Paths Debug ===")

        # Check embed_tokens
        if hasattr(self, "embed_tokens"):
            print(f"embed_tokens type: {type(self.embed_tokens)}")
            if hasattr(self.embed_tokens, "embeddings"):
                print(f"embed_tokens.embeddings weight shape: {self.embed_tokens.embeddings.weight.shape}")
            elif hasattr(self.embed_tokens, "weight"):
                print(f"embed_tokens weight shape: {self.embed_tokens.weight.shape}")

        # Check lm_head
        if hasattr(self, "lm_head"):
            print(f"lm_head type: {type(self.lm_head)}")
            if hasattr(self.lm_head, "linear"):
                print(f"lm_head.linear weight shape: {self.lm_head.linear.weight.shape}")
            elif hasattr(self.lm_head, "weight"):
                print(f"lm_head weight shape: {self.lm_head.weight.shape}")

        # Check norm
        if hasattr(self, "norm"):
            print(f"norm type: {type(self.norm)}")
            print(f"norm weight shape: {self.norm.weight.shape}")

    def set_state_dict(self, state_dict: Dict[str, paddle.Tensor]) -> None:
        """
        Set state dict with weight mapping (required by ModelForCasualLM base class)

        Args:
            state_dict: Dictionary mapping parameter names to tensors
        """

        # Convert state dict to iterator format and reuse load_weights logic
        def state_dict_iterator():
            for name, tensor in state_dict.items():
                yield name, tensor

        self.load_weights(state_dict_iterator())

    @property
    def dtype(self):
        """Get the dtype of the model"""
        return next(self.parameters()).dtype


class MiniCPM41DecoderLayer(nn.Layer):
    """
    MiniCPM4.1-8B Decoder Layer with attention and feed-forward
    """

    def __init__(self, fd_config: FDConfig, layer_id: int, prefix: str = ""):
        super().__init__()

        self.fd_config = fd_config
        self.model_config = fd_config.model_config
        self.layer_id = layer_id
        self.prefix = prefix

        # Self-attention
        self.self_attn = MiniCPM41Attention(fd_config, layer_id=layer_id, prefix=f"{prefix}.self_attn")

        # Input layernorm (pre-attention norm)
        self.input_layernorm = RMSNorm(
            fd_config,
            hidden_size=self.model_config.hidden_size,
            eps=self.model_config.rms_norm_eps,
            prefix=f"{prefix}.input_layernorm",
            begin_norm_axis=-1,
        )

        # MLP
        self.mlp = MiniCPM41MLP(fd_config, layer_id=layer_id, prefix=f"{prefix}.mlp")

        # Post-attention layernorm
        self.post_attention_layernorm = RMSNorm(
            fd_config,
            hidden_size=self.model_config.hidden_size,
            eps=self.model_config.rms_norm_eps,
            prefix=f"{prefix}.post_attention_layernorm",
            begin_norm_axis=-1,
        )

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_value: Optional[Tuple] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        sparse_config: Optional[Dict] = None,
    ) -> Tuple[paddle.Tensor, Optional[Tuple], Optional[paddle.Tensor]]:
        """
        Forward pass of decoder layer

        Args:
            hidden_states: Input hidden states
            attention_mask: Attention mask
            position_ids: Position IDs
            past_key_value: Past key-value cache
            use_cache: Whether to use cache
            output_attentions: Whether to output attention weights
            sparse_config: Sparse attention configuration
        """

        residual = hidden_states

        # Pre-attention layernorm
        hidden_states = self.input_layernorm(hidden_states)

        # Self-attention
        outputs = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
            sparse_config=sparse_config,
        )

        if use_cache:
            hidden_states, present_key_value, attention_weights = outputs
        else:
            hidden_states, attention_weights = outputs
            present_key_value = None

        # Residual connection
        hidden_states = residual + hidden_states

        residual = hidden_states

        # Post-attention layernorm
        hidden_states = self.post_attention_layernorm(hidden_states)

        # MLP
        hidden_states = self.mlp(hidden_states)

        # Residual connection
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if use_cache:
            outputs += (present_key_value,)

        if output_attentions:
            outputs += (attention_weights,)

        return outputs


class MiniCPM41Attention(nn.Layer):
    """
    MiniCPM4.1-8B Attention with sparse attention support
    """

    def __init__(self, fd_config: FDConfig, layer_id: int, prefix: str = ""):
        super().__init__()

        self.fd_config = fd_config
        self.model_config = fd_config.model_config
        self.layer_id = layer_id
        self.prefix = prefix

        # Compute head dimensions
        self.head_dim = self.model_config.hidden_size // self.model_config.num_attention_heads

        # Check for GQA (Grouped Query Attention) configuration
        self.num_key_value_heads = getattr(
            self.model_config, "num_key_value_heads", self.model_config.num_attention_heads
        )
        self.kv_dim = self.head_dim * self.num_key_value_heads

        # Separate Q, K, V linear projections (matching MiniCPM4.1-8B structure)
        # Q projection uses full hidden_size (all heads)
        self.q_proj = RowParallelLinear(
            fd_config,
            prefix=f"{prefix}.q_proj",
            input_size=self.model_config.hidden_size,
            output_size=self.model_config.hidden_size,
            layer_id=layer_id,
        )

        # K projection uses kv_dim (grouped query attention)
        self.k_proj = RowParallelLinear(
            fd_config,
            prefix=f"{prefix}.k_proj",
            input_size=self.model_config.hidden_size,
            output_size=self.kv_dim,
            layer_id=layer_id,
        )

        # V projection uses kv_dim (grouped query attention)
        self.v_proj = RowParallelLinear(
            fd_config,
            prefix=f"{prefix}.v_proj",
            input_size=self.model_config.hidden_size,
            output_size=self.kv_dim,
            layer_id=layer_id,
        )

        # Output projection
        self.o_proj = RowParallelLinear(
            fd_config,
            prefix=f"{prefix}.o_proj",
            input_size=self.model_config.hidden_size,
            output_size=self.model_config.hidden_size,
            layer_id=layer_id,
        )

        # Attention mechanism
        self.attn = Attention(
            fd_config,
            layer_id=layer_id,
            prefix=prefix,
            use_neox_rotary_style=True,
        )

        # Optional Q/K normalization (if enabled in config)
        self.q_norm = None
        self.k_norm = None
        if getattr(self.model_config, "use_qk_norm", False):
            self.q_norm = RMSNorm(
                fd_config,
                hidden_size=self.head_dim,
                eps=self.model_config.rms_norm_eps,
                prefix=f"{prefix}.q_norm",
                begin_norm_axis=-1,
            )
            self.k_norm = RMSNorm(
                fd_config,
                hidden_size=self.head_dim,
                eps=self.model_config.rms_norm_eps,
                prefix=f"{prefix}.k_norm",
                begin_norm_axis=-1,
            )

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_value: Optional[Tuple] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        sparse_config: Optional[Dict] = None,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[paddle.Tensor]]:
        """
        Forward pass of attention
        """

        # Separate Q, K, V projections (matching MiniCPM4.1-8B structure)
        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        batch_size, seq_len, hidden_size = queries.shape

        # Q reshape: [batch_size, seq_len, num_q_heads, head_dim]
        queries = queries.reshape([batch_size, seq_len, self.model_config.num_attention_heads, self.head_dim])

        # K, V reshape: [batch_size, seq_len, num_kv_heads, head_dim]
        keys = keys.reshape([batch_size, seq_len, self.num_key_value_heads, self.head_dim])
        values = values.reshape([batch_size, seq_len, self.num_key_value_heads, self.head_dim])

        # Transpose to [batch_size, num_heads, seq_len, head_dim]
        queries = queries.transpose([0, 2, 1, 3])
        keys = keys.transpose([0, 2, 1, 3])
        values = values.transpose([0, 2, 1, 3])

        # Apply Q/K normalization if enabled
        if self.q_norm is not None and self.k_norm is not None:
            queries = self.q_norm(queries)
            keys = self.k_norm(keys)

        # Apply attention
        attn_output = self.attn(
            queries,
            keys,
            values,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            sparse_config=sparse_config,
        )

        if use_cache:
            attn_output, present_key_value = attn_output
        else:
            present_key_value = None

        # Reshape and project output
        attn_output = attn_output.transpose([0, 2, 1, 3])  # [batch_size, seq_len, num_heads, head_dim]
        attn_output = attn_output.reshape([batch_size, seq_len, -1])

        # Output projection
        output = self.o_proj(attn_output)

        outputs = (output,)

        if use_cache:
            outputs += (present_key_value,)

        if output_attentions:
            # For simplicity, we don't return attention weights by default
            outputs += (None,)

        return outputs


class MiniCPM41MLP(nn.Layer):
    """
    MiniCPM4.1-8B MLP (SwiGLU activation)
    """

    def __init__(self, fd_config: FDConfig, layer_id: int, prefix: str = ""):
        super().__init__()

        self.fd_config = fd_config
        self.model_config = fd_config.model_config
        self.layer_id = layer_id
        self.prefix = prefix

        # Get intermediate_size from config or default to 4 * hidden_size
        intermediate_size = getattr(self.model_config, "intermediate_size", 4 * self.model_config.hidden_size)

        # Gate projection (for SwiGLU)
        self.gate_proj = RowParallelLinear(
            fd_config,
            prefix=f"{prefix}.gate_proj",
            input_size=self.model_config.hidden_size,
            output_size=intermediate_size,
            layer_id=layer_id,
        )

        # Up projection
        self.up_proj = RowParallelLinear(
            fd_config,
            prefix=f"{prefix}.up_proj",
            input_size=self.model_config.hidden_size,
            output_size=intermediate_size,
            layer_id=layer_id,
        )

        # Down projection
        self.down_proj = RowParallelLinear(
            fd_config,
            prefix=f"{prefix}.down_proj",
            input_size=intermediate_size,
            output_size=self.model_config.hidden_size,
            layer_id=layer_id,
        )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """Forward pass with SwiGLU activation"""
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)
