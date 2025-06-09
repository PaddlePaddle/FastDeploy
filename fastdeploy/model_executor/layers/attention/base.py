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

import os

import paddle
from paddle import nn

import fastdeploy


class Attention(nn.Layer):
    """
    Attention Layer
    """

    def __init__(
        self,
        inference_args,
        prefix,
        out_scale=-1,
        use_neox_rotary_style=False,
        rope_theta=10000.0,
        rope_3d=False,
        qkv_scale=None,
        qkv_bias=None,
        linear_shift=None,
        linear_smooth=None,
    ):
        """
        Initialize the attention layer with various parameters.

        Args:
            inference_args (dict or object): Contains arguments for inference, including
                number of key-value heads, weight data type, activation data type, etc.
            prefix (str): The name of the attention layer for identification purposes.
            out_scale (float, optional): Output scale factor. Defaults to -1.
            use_neox_rotary_style (bool, optional): Whether to use the NeoX rotary position
                encoding style. Defaults to False.
            rope_theta (float, optional): Theta value for the rope position encoding. Defaults to 10000.0.
            qkv_scale (float or None, optional): Quantization scale for QKV weights.
                Used only for certain quantization configurations. Defaults to None.
            qkv_bias (Tensor or None, optional): Bias for QKV linear layer. Defaults to None.
            linear_shift (float or None, optional): Linear shift factor used in
                quantization. Used only for certain quantization configurations.
                Defaults to None.
            linear_smooth (float or None, optional): Linear smooth factor used in
                quantization. Used only for certain quantization configurations.
                Defaults to None.
        """
        super().__init__()
        self.inference_args = inference_args
        self.nranks = inference_args.mp_size
        self.kv_num_heads = inference_args.num_key_value_heads // self.nranks
        self.head_dim = self.inference_args.head_dim
        self.prefix = prefix
        self.cache_k_scale_name = prefix + ".cachek_matmul.activation_quanter"
        self.cache_v_scale_name = prefix + ".cachev_matmul.activation_quanter"
        self.out_scale = out_scale

        self.cache_k_zp_name = self.cache_k_scale_name + ".zero_point"
        self.cache_v_zp_name = self.cache_v_scale_name + ".zero_point"

        self.use_neox_rotary_style = use_neox_rotary_style
        self.rope_theta = rope_theta
        self.rope_3d = rope_3d

        self._dtype = self._helper.get_default_dtype()
        if self._dtype == "bfloat16":
            self._fuse_kernel_compute_dtype = "bf16"
        elif self._dtype == "float16":
            self._fuse_kernel_compute_dtype = "fp16"
        elif self._dtype == "float32":
            self._fuse_kernel_compute_dtype = "fp32"
        else:
            raise ValueError(f"Just support float32, float16 and \
                    bfloat16 as default dtype, but received {self._dtype}")

        self.cache_scale_dtype = (
            self._dtype if self.inference_args.use_append_attn else "float32")

        self.qkv_bias = qkv_bias
        if inference_args.weight_dtype == "int8" and inference_args.act_dtype == "int8":
            self.qkv_scale = qkv_scale
            self.linear_shift = linear_shift
            self.linear_smooth = linear_smooth
        if (inference_args.cachekv_dtype == "int8"
                or inference_args.cachekv_dtype == "int4"
                or inference_args.cachekv_dtype == "float8_e4m3fn"):
            self.set_cachekv_scale()
        # qkv_bias fused with attention only when W8A8
        if not (inference_args.weight_dtype == "int8"
                and inference_args.act_dtype == "int8"):
            self.qkv_bias = None

    def set_cachekv_scale(self):
        """
        Set cache key (K) and value (V) scaling factors.

        This method initializes and sets the scaling factors for cache key (K) and value (V)
        tensors, which are used in attention mechanisms to adjust the scale of the cache
        representations. Additionally, it calculates and sets the inverse of these scaling
        factors for the output cache K and V tensors.

        Args:
            None - This method does not take any explicit arguments as it relies on the
                instance variables of the class, such as `self.kv_num_heads`,
                `self.cache_k_scale_name`, `self.cache_v_scale_name`, and
                `self.inference_args.cachekv_scale_dict` for its functionality.

        Returns:
            None - This method modifies the instance variables directly and does not return
                any values.
        """
        self.cache_k_scale = self.create_parameter(
            shape=([self.kv_num_heads *
                    self.head_dim] if self.inference_args.is_channel_wise else
                   [self.kv_num_heads]),
            dtype=self.cache_scale_dtype,
            is_bias=False,
        )
        self.cache_v_scale = self.create_parameter(
            shape=([self.kv_num_heads *
                    self.head_dim] if self.inference_args.is_channel_wise else
                   [self.kv_num_heads]),
            dtype=self.cache_scale_dtype,
            is_bias=False,
        )
        self.cache_k_out_scale = self.create_parameter(
            shape=([self.kv_num_heads *
                    self.head_dim] if self.inference_args.is_channel_wise else
                   [self.kv_num_heads]),
            attr=None,
            dtype=self.cache_scale_dtype,
            is_bias=False,
        )
        self.cache_v_out_scale = self.create_parameter(
            shape=([self.kv_num_heads *
                    self.head_dim] if self.inference_args.is_channel_wise else
                   [self.kv_num_heads]),
            attr=None,
            dtype=self.cache_scale_dtype,
            is_bias=False,
        )

        if self.cache_k_scale_name in self.inference_args.cachekv_scale_dict:
            cache_k_scale = paddle.cast(
                paddle.to_tensor(self.inference_args.cachekv_scale_dict[
                    self.cache_k_scale_name]),
                self.cache_scale_dtype,
            )
            cache_k_out_scale = 1.0 / cache_k_scale
        else:
            if os.getenv("EP_DECODER_PERF_TEST", "False") == "True":
                cache_k_scale = paddle.zeros(self.cache_k_scale.shape,
                                             self.cache_k_scale.dtype)
                cache_k_out_scale = paddle.zeros(self.cache_k_out_scale.shape,
                                                 self.cache_k_out_scale.dtype)
            else:
                raise KeyError(
                    f"{self.cache_k_scale_name} not found in scale dict")

        if self.cache_v_scale_name in self.inference_args.cachekv_scale_dict:
            cache_v_scale = paddle.cast(
                paddle.to_tensor(self.inference_args.cachekv_scale_dict[
                    self.cache_v_scale_name]),
                self.cache_scale_dtype,
            )
            cache_v_out_scale = 1.0 / cache_v_scale
        else:
            if os.getenv("EP_DECODER_PERF_TEST", "False") == "True":
                cache_v_scale = paddle.zeros(self.cache_v_scale.shape,
                                             self.cache_v_scale.dtype)
                cache_v_out_scale = paddle.zeros(self.cache_v_out_scale.shape,
                                                 self.cache_v_out_scale.dtype)
            else:
                raise KeyError(
                    f"{self.cache_v_scale_name} not found in scale dict")

        self.cache_k_scale.set_value(cache_k_scale)
        self.cache_v_scale.set_value(cache_v_scale)
        self.cache_k_out_scale.set_value(cache_k_out_scale)
        self.cache_v_out_scale.set_value(cache_v_out_scale)

        if self.inference_args.has_zero_point:
            self.cache_k_zp = self.create_parameter(
                shape=([self.kv_num_heads *
                        self.head_dim] if self.inference_args.is_channel_wise
                       else [self.kv_num_heads]),
                dtype=self.cache_scale_dtype,
                is_bias=False,
            )
            self.cache_v_zp = self.create_parameter(
                shape=([self.kv_num_heads *
                        self.head_dim] if self.inference_args.is_channel_wise
                       else [self.kv_num_heads]),
                dtype=self.cache_scale_dtype,
                is_bias=False,
            )
            if self.cache_k_zp_name in self.inference_args.cachekv_scale_dict:
                cache_k_zp = paddle.cast(
                    paddle.to_tensor(self.inference_args.cachekv_scale_dict[
                        self.cache_k_zp_name]),
                    self.cache_scale_dtype,
                )
            else:
                cache_k_zp = paddle.zeros(
                    ([self.kv_num_heads *
                      self.head_dim] if self.inference_args.is_channel_wise
                     else [self.kv_num_heads]),
                    dtype=self.cache_scale_dtype,
                )
            if self.cache_v_zp_name in self.inference_args.cachekv_scale_dict:
                cache_v_zp = paddle.cast(
                    paddle.to_tensor(self.inference_args.cachekv_scale_dict[
                        self.cache_v_zp_name]),
                    self.cache_scale_dtype,
                )
            else:
                cache_v_zp = paddle.zeros(
                    ([self.kv_num_heads *
                      self.head_dim] if self.inference_args.is_channel_wise
                     else [self.kv_num_heads]),
                    dtype=self.cache_scale_dtype,
                )
            self.cache_k_zp.set_value(cache_k_zp)
            self.cache_v_zp.set_value(cache_v_zp)

    def forward(
        self,
        qkv,
        input_ids,
        rotary_embs,
        rotary_emb_dims,
        key_cache,
        value_cache,
        pre_key_cache,
        pre_value_cache,
        pre_caches_length,
        attn_mask,
        kv_signal_data,
        **kwargs,
    ):
        """
            Compute the attention for a single time step.

        Args:
            qkv (Tensor): The output of the linear transformation of query, key and value.
                Shape: [batch_size, num_heads, seq_len, embed_dim // num_heads].
            padding_offset (Tensor): The offset to be added to the sequence length when computing
                the attention mask. Shape: [batch_size, 1].
            input_ids (Tensor, optional): The input ids of the batch. Used for computing the
                attention mask. Default: None. Shape: [batch_size, max_sequence_length].
            rotary_embs (Tensor, optional): The rotary position embeddings. Default: None.
                Shape: [num_heads, rotary_emb_dims].
            rotary_emb_dims (int, optional): The dimension of the rotary position embeddings.
                Default: None.
            caches (List[Tensor], optional): The cache tensors used in the computation of the
                attention. Default: None.
            pre_caches (List[Tensor], optional): The pre-computed cache tensors used in the
                computation of the attention. Default: None.
            pre_caches_length (int, optional): The length of the pre-computed cache tensors.
                Default: None.
            attn_mask (Tensor, optional): The attention mask. Default: None.
                Shape: [batch_size, max_sequence_length].
            **kwargs (dict, optional): Additional keyword arguments passed along.

        Returns:
            Tensor: The output of the linear transformation after applying the attention.
                Shape: [batch_size, embed_dim // num_heads].

        Raises:
            None.
        """
        k_quant_scale = kwargs.get("k_quant_scale", None)
        v_quant_scale = kwargs.get("v_quant_scale", None)
        k_dequant_scale = kwargs.get("k_dequant_scale", None)
        v_dequant_scale = kwargs.get("v_dequant_scale", None)

        if not self.inference_args.use_dynamic_cachekv_quant:
            k_quant_scale = getattr(self, "cache_k_scale", None)
            v_quant_scale = getattr(self, "cache_v_scale", None)
            k_dequant_scale = getattr(self, "cache_k_out_scale", None)
            v_dequant_scale = getattr(self, "cache_v_out_scale", None)
            cache_quant_type_str = self.inference_args.cache_quant_type
        else:
            cache_quant_type_str = "none"

        if self.inference_args.use_append_attn:
            out = fastdeploy.model_executor.ops.gpu.append_attention(
                qkv,
                key_cache,
                value_cache,
                kwargs.get("seq_lens_encoder", None),
                kwargs.get("seq_lens_decoder", None),
                kwargs.get("seq_lens_this_time", None),
                kwargs.get("padding_offsets", None),
                kwargs.get("cum_offsets", None),
                kwargs.get("block_tables", None),
                kwargs.get("encoder_batch_ids", None),
                kwargs.get("encoder_tile_ids_per_batch", None),
                kwargs.get("encoder_num_blocks", None),
                kwargs.get("kv_batch_ids", None),
                kwargs.get("kv_tile_ids_per_batch", None),
                kwargs.get("kv_num_blocks", None),
                kwargs.get("decoder_batch_ids", None),
                kwargs.get("decoder_tile_ids_per_batch", None),
                kwargs.get("decoder_num_blocks", None),
                kwargs.get("set_max_lengths", None),
                kwargs.get("max_len_kv", None),
                rotary_embs,
                attn_mask,
                getattr(self, "qkv_bias", None),
                getattr(self, "qkv_scale", None),
                k_quant_scale,
                v_quant_scale,
                k_dequant_scale,
                v_dequant_scale,
                getattr(self, "cache_k_zp", None),  # cache_k_zp
                getattr(self, "cache_v_zp", None),  # cache_v_zp
                getattr(self, "linear_shift", None),  # out_shifts
                getattr(self, "linear_smooth", None),  # out_smooths
                kv_signal_data,
                self._fuse_kernel_compute_dtype,
                cache_quant_type_str,  # cache_quant_type
                self.use_neox_rotary_style,
                self.rope_3d,
                kwargs.get("max_input_length", -1),
                self.inference_args.quant_max_bound,
                self.inference_args.quant_min_bound,
                self.out_scale,  # out_linear_in_scale
                kwargs.get("encoder_block_shape_q", 64),
                kwargs.get("decoder_block_shape_q", 16),
                kwargs.get("max_partition_size", 32768),
                kwargs.get("encoder_max_partition_size", 32768),
                self.inference_args.speculate_max_draft_token_num +
                1,  # speculate_max_draft_token_num
                True,  # causal
                self.inference_args.speculate_method
                is not None,  # speculate_decoder
            )[0]
        else:
            out = paddle.incubate.nn.functional.block_multihead_attention(
                qkv,
                key_cache,
                value_cache,
                kwargs.get("seq_lens_encoder", None),
                kwargs.get("seq_lens_decoder", None),
                kwargs.get("seq_lens_this_time", None),
                kwargs.get("padding_offsets", None),
                kwargs.get("cum_offsets", None),
                kwargs.get("cu_seqlens_q", None),
                kwargs.get("cu_seqlens_k", None),
                kwargs.get("block_tables", None),
                pre_key_cache,
                pre_value_cache,
                k_quant_scale,
                v_quant_scale,
                k_dequant_scale,
                v_dequant_scale,
                getattr(self, "qkv_scale", None),
                getattr(self, "qkv_bias", None),
                getattr(self, "linear_shift", None),
                getattr(self, "linear_smooth", None),
                kwargs.get("max_enc_len_this_time", None),
                kwargs.get("max_dec_len_this_time", None),
                rotary_embs,
                attn_mask,
                None,  # tgt_mask
                kwargs.get("max_input_length", -1),
                kwargs.get("block_size", 64),
                self.use_neox_rotary_style,
                self.inference_args.use_dynamic_cachekv_quant,
                quant_round_type=self.inference_args.quant_round_type,
                quant_max_bound=self.inference_args.quant_max_bound,
                quant_min_bound=self.inference_args.quant_min_bound,
                out_scale=self.out_scale,
                compute_dtype=self._fuse_kernel_compute_dtype,
                rope_theta=self.rope_theta,
            )[0]

        return out
