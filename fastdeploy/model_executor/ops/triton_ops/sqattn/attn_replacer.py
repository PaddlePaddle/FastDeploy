import copy
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import paddle
import paddleformers
import seaborn as sns
from .attn_triton_mix_relative import attn_hierarchical_relative_window
from .fake_quant import (FloatQuantizer, IntegerQuantizer)
from .sdpa_attention import sdpa_attention_forward


@paddle.no_grad()
def replace_sdpa_for_block_with_attn_weights(
    module: paddle.nn.Layer,
    blockidx: int,
    args,
    bit8_window_sizes=None,
    bit4_window_sizes=None,
    sink_window_size=0,
):
    if isinstance(module, paddleformers.transformers.qwen2.modeling.Qwen2DecoderLayer):
        pass
        attn_fn = delayed_sdpa_wrapper_with_attn_weights(
            blockidx,
            args=args,
            bit8_window_sizes=bit8_window_sizes,
            bit4_window_sizes=bit4_window_sizes,
            sink_window_size=sink_window_size,
        )
        impl_name = f"sparsequantattn_{blockidx}"
        paddleformers.transformers.qwen2.modeling.ALL_ATTENTION_FUNCTIONS[
            impl_name
        ] = attn_fn
        module.self_attn.config = copy.deepcopy(module.self_attn.config)
        module.self_attn.config._attn_implementation = impl_name


@paddle.no_grad()
def replace_sdpa_for_block(
    module: paddle.nn.Layer,
    blockidx: int,
    args,
    bit8_window_sizes=None,
    bit4_window_sizes=None,
    sink_window_size=0,
    use_sageattn=False,
):
    if isinstance(module, paddleformers.transformers.qwen2.modeling.Qwen2DecoderLayer):
        pass
        attn_fn = sdpa_attention_forward
       
        impl_name = f"sparsequantattn_{blockidx}"
        paddleformers.transformers.qwen2.modeling.ALL_ATTENTION_FUNCTIONS[
            impl_name
        ] = attn_fn
        module.self_attn.config = copy.deepcopy(module.self_attn.config)
        module.self_attn.config._attn_implementation = impl_name

@paddle.no_grad()
def replace_mp_triton_for_block(
    module: paddle.nn.Layer,
    blockidx: int,
    args,
    bit8_window_sizes=None,
    bit4_window_sizes=None,
    sink_window_size=0,
    use_sageattn=False,
):
    if isinstance(module, paddleformers.transformers.qwen2.modeling.Qwen2DecoderLayer):
        pass
        attn_fn = mp_triton_wrapper(
            blockidx,
            args=args,
            bit8_window_sizes=bit8_window_sizes,
            bit4_window_sizes=bit4_window_sizes,
            sink_window_size=sink_window_size,
        )
        impl_name = f"sparsequantattn_{blockidx}"
        paddleformers.transformers.qwen2.modeling.ALL_ATTENTION_FUNCTIONS[
            impl_name
        ] = attn_fn
        module.self_attn.config = copy.deepcopy(module.self_attn.config)
        module.self_attn.config._attn_implementation = impl_name


def delayed_sdpa_wrapper(
    layer_idx, bit8_window_sizes=0, bit4_window_sizes=0, sink_window_size=0, args=None
):
    is_per_head = isinstance(bit8_window_sizes, list) or isinstance(
        bit4_window_sizes, list
    )
    if is_per_head:
        if isinstance(bit8_window_sizes, list):
            num_heads = len(bit8_window_sizes)
        else:
            num_heads = len(bit4_window_sizes)
        if isinstance(bit8_window_sizes, int):
            bit8_window_sizes = [bit8_window_sizes] * num_heads
        if isinstance(bit4_window_sizes, int):
            bit4_window_sizes = [bit4_window_sizes] * num_heads
        if bit8_window_sizes is None:
            bit8_window_sizes = [0] * num_heads
        if bit4_window_sizes is None:
            bit4_window_sizes = [0] * num_heads
    else:
        if isinstance(bit8_window_sizes, list):
            bit8_window_sizes = bit8_window_sizes[0] if bit8_window_sizes else 0
        if isinstance(bit4_window_sizes, list):
            bit4_window_sizes = bit4_window_sizes[0] if bit4_window_sizes else 0
        bit8_window_sizes = bit8_window_sizes or 0
        bit4_window_sizes = bit4_window_sizes or 0

    def construct_mix_bit_mask_per_head(
        seq_len, bit8_window_sizes, bit4_window_sizes, sink_window_size, device="cuda"
    ):
        """
        返回 shape = (num_heads, L, L) 的 bool mask，True表示可以attend。
        """
        num_heads = len(bit8_window_sizes)
        q_idx = paddle.arange(seq_len, device=device).unsqueeze(0).unsqueeze(2)
        kv_idx = paddle.arange(seq_len, device=device).unsqueeze(0).unsqueeze(1)
        q_idx = q_idx.expand(num_heads, seq_len, 1)
        kv_idx = kv_idx.expand(num_heads, 1, seq_len)
        bit8_window_sizes = paddle.tensor(bit8_window_sizes, device=device).view(
            -1, 1, 1
        )
        bit4_window_sizes = paddle.tensor(bit4_window_sizes, device=device).view(
            -1, 1, 1
        )
        causal_mask = kv_idx <= q_idx
        bit8_window_mask = kv_idx > q_idx - bit8_window_sizes
        sink_mask = kv_idx < sink_window_size
        fp8_mask = causal_mask & (sink_mask | bit8_window_mask)
        kv_idx_4bit = kv_idx - seq_len // 2
        bit8_window_mask_4bit = kv_idx_4bit <= q_idx - bit8_window_sizes
        bit4_window_mask = kv_idx_4bit > q_idx - bit8_window_sizes - bit4_window_sizes
        bit4_no_sink_mask = kv_idx_4bit >= sink_window_size
        int4_mask = bit4_no_sink_mask & bit8_window_mask_4bit & bit4_window_mask
        is_bit8_part = kv_idx < seq_len // 2
        final_mask = is_bit8_part & fp8_mask | ~is_bit8_part & int4_mask
        return final_mask[:, : seq_len // 2, :]

    def construct_mix_bit_mask(
        seq_len, bit8_window_size, bit4_window_size, sink_window_size, device="cuda"
    ):
        """
        返回 shape = (L, L) 的 mask，True表示可以attend，False表示mask掉。
        """
        q_idx = paddle.arange(seq_len, device=device).unsqueeze(1)
        kv_idx = paddle.arange(seq_len, device=device).unsqueeze(0)
        causal_mask = kv_idx <= q_idx
        bit8_window_mask = kv_idx > q_idx - bit8_window_size
        sink_mask = kv_idx < sink_window_size
        fp8_mask = causal_mask & (sink_mask | bit8_window_mask)
        kv_idx_4bit = kv_idx - seq_len // 2
        bit8_window_mask_4bit = kv_idx_4bit <= q_idx - bit8_window_size
        bit4_window_mask = kv_idx_4bit > q_idx - bit8_window_size - bit4_window_size
        bit4_no_sink_mask = kv_idx_4bit >= sink_window_size
        int4_mask = bit4_no_sink_mask & bit8_window_mask_4bit & bit4_window_mask
        is_bit8_part = kv_idx < seq_len // 2
        final_mask = is_bit8_part & fp8_mask | ~is_bit8_part & int4_mask
        return final_mask[: seq_len // 2, :]

    def attention_fn(
        module,
        q,
        k,
        v,
        attn_mask,
        dropout=0.0,
        scaling=1.0,
        sliding_window=None,
        **kwargs,
    ):
        if args.quant:
            q_len, dim = q.shape[2], q.shape[3]
            kv_len = k.shape[2]
            km = k.mean(dim=2, keepdim=True)
            k = k - km
            if args.qk_qtype == "int":
                bit8_qk_quantizer = IntegerQuantizer(8, True, "per_token")
                bit4_qk_quantizer = IntegerQuantizer(4, True, "per_token")
            elif args.qk_qtype == "e4m3":
                bit8_qk_quantizer = FloatQuantizer("e4m3", True, "per_token")
                bit4_qk_quantizer = IntegerQuantizer(4, True, "per_token")
            elif args.qk_qtype == "e5m2":
                bit8_qk_quantizer = FloatQuantizer("e5m2", True, "per_token")
                bit4_qk_quantizer = IntegerQuantizer(4, True, "per_token")
            else:
                raise ValueError(f"Invalid quantization type: {args.qk_qtype}")
            if args.v_qtype == "int":
                bit8_v_quantizer = IntegerQuantizer(8, False, "per_channel")
            elif args.v_qtype == "e4m3":
                bit8_v_quantizer = FloatQuantizer("e4m3", True, "per_channel")
            elif args.v_qtype == "e5m2":
                bit8_v_quantizer = FloatQuantizer("e5m2", True, "per_channel")
            else:
                raise ValueError(f"Invalid quantization type: {args.v_qtype}")
            q_bit8 = bit8_qk_quantizer.fake_quant_tensor(q)
            q_bit4 = bit4_qk_quantizer.fake_quant_tensor(q)
            k_bit8 = bit8_qk_quantizer.fake_quant_tensor(k)
            k_bit4 = bit4_qk_quantizer.fake_quant_tensor(k)
            v_bit8 = bit8_v_quantizer.fake_quant_tensor(v)
            q_bit8_bit8 = q_bit8
            q_bit8_bit4 = paddle.cat([q_bit8, q_bit4], dim=2)
            k_bit8_bit4 = paddle.cat([k_bit8, k_bit4], dim=2)
            v_bit8_bit8 = paddle.cat([v_bit8, v_bit8], dim=2)
            if is_per_head:
                mask = construct_mix_bit_mask_per_head(
                    seq_len=kv_len * 2,
                    bit8_window_sizes=bit8_window_sizes,
                    bit4_window_sizes=bit4_window_sizes,
                    sink_window_size=sink_window_size,
                )
            else:
                mask = construct_mix_bit_mask(
                    seq_len=kv_len * 2,
                    bit8_window_size=bit8_window_sizes,
                    bit4_window_size=bit4_window_sizes,
                    sink_window_size=sink_window_size,
                )
            if is_per_head:
                mask = mask[:, -q_len:, :]
            else:
                mask = mask[-q_len:, :]
            attn_output, attn_weights = sdpa_attention_forward(
                module,
                q_bit8,
                k_bit8,
                v_bit8,
                attention_mask=mask,
                dropout=dropout,
                scaling=scaling,
                sliding_window=sliding_window,
                **kwargs,
            )
            import pdb

            pdb.set_trace()
            return attn_output[:, :q_len, :, :], attn_weights
        else:
            if args.vis_attn:

                def cal_attn_weight(query, key, is_causal=True, attn_mask=None):
                    key = repeat_kv(key, 6)
                    query = query.contiguous()
                    key = key.contiguous()
                    L, S = query.size(-2), key.size(-2)
                    scale_factor = 1 / math.sqrt(query.size(-1))
                    attn_bias = paddle.zeros(
                        L, S, dtype=query.dtype, device=query.place
                    )
                    if is_causal:
                        assert attn_mask is None
                        temp_mask = paddle.ones(
                            L, S, dtype=paddle.bool, device=query.place
                        ).tril(diagonal=0)
                        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
                        attn_bias.to(query.dtype)
                    if attn_mask is not None:
                        if attn_mask.dtype == paddle.bool:
                            attn_bias.masked_fill_(
                                attn_mask.logical_not(), float("-inf")
                            )
                        else:
                            attn_bias = attn_mask + attn_bias
                    attn_weight = query @ key.transpose(-2, -1) * scale_factor
                    attn_weight += attn_bias
                    attn_weight = paddle.softmax(attn_weight, dim=-1)
                    attn_weight_pool = paddle.nn.functional.max_pool2d(
                        x=attn_weight, kernel_size=(10, 10), stride=(10, 10)
                    )
                    return attn_weight_pool

                attn_weights = cal_attn_weight(q, k)
                os.makedirs(
                    f"attn_vis_softmax_max_pool/layer_{layer_idx}", exist_ok=True
                )
                attn_map = attn_weights.detach().to(paddle.float32).cpu().mean(dim=0)
                for h in range(attn_map.shape[0]):
                    plt.imshow(attn_map[h], cmap="coolwarm", aspect="auto")
                    plt.colorbar()
                    plt.title(f"Layer {layer_idx} Head {h}")
                    plt.savefig(
                        f"attn_vis_softmax_max_pool/layer_{layer_idx}/head_{h}.png"
                    )
                    plt.close()
            return sdpa_attention_forward(
                module,
                q,
                k,
                v,
                attention_mask=attn_mask,
                dropout=dropout,
                scaling=scaling,
                sliding_window=sliding_window,
                **kwargs,
            )

    return attention_fn


def delayed_sdpa_wrapper_with_attn_weights(
    layer_idx, bit8_window_sizes=0, bit4_window_sizes=0, sink_window_size=0, args=None
):
    def attention_fn(
        module,
        query,
        key,
        value,
        attention_mask,
        dropout=0.0,
        scaling=1.0,
        sliding_window=None,
        **kwargs,
    ):
        attn_weights = cal_attn_weight(module, query, key)
        args.current_attention = attn_weights
        return sdpa_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask=attention_mask,
            dropout=dropout,
            scaling=scaling,
            sliding_window=sliding_window,
            **kwargs,
        )

    return attention_fn


def cal_attn_weight(module, query, key, is_causal=True, attn_mask=None):
    if hasattr(module, "num_key_value_groups"):
        key = repeat_kv(key, module.num_key_value_groups)
        query = query.permute(0, 2, 1, 3)
    query = query.contiguous()
    key = key.contiguous()
    L, S = query.shape[-2], key.shape[-2]
    scale_factor = 1 / math.sqrt(query.shape[-1])
    attn_bias = paddle.zeros(L, S, dtype=query.dtype, device=query.place)
    if is_causal:
        assert attn_mask is None
        temp_mask = paddle.ones(L, S, dtype=paddle.bool, device=query.place).tril(
            diagonal=0
        )
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)
    if attn_mask is not None:
        if attn_mask.dtype == paddle.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = paddle.softmax(attn_weight, dim=-1)
    return attn_weight


def repeat_kv(hidden_states: paddle.Tensor, n_rep: int) -> paddle.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    # batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    batch, slen, num_key_value_heads, head_dim = hidden_states.shape
    hidden_states = hidden_states.permute(0, 2, 1, 3)  # [batch, num_key_value_heads, slen, head_dim]
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def mp_triton_wrapper(
    layer_idx, bit8_window_sizes=0, bit4_window_sizes=0, sink_window_size=0, args=None
):
    is_per_head = isinstance(bit8_window_sizes, list) or isinstance(
        bit4_window_sizes, list
    )
    if is_per_head:
        if isinstance(bit8_window_sizes, list):
            num_heads = len(bit8_window_sizes)
        else:
            num_heads = len(bit4_window_sizes)
        if isinstance(bit8_window_sizes, int):
            bit8_window_sizes = [bit8_window_sizes] * num_heads
        if isinstance(bit4_window_sizes, int):
            bit4_window_sizes = [bit4_window_sizes] * num_heads
        if bit8_window_sizes is None:
            bit8_window_sizes = [0] * num_heads
        if bit4_window_sizes is None:
            bit4_window_sizes = [0] * num_heads
    else:
        if isinstance(bit8_window_sizes, list):
            bit8_window_sizes = bit8_window_sizes[0] if bit8_window_sizes else 0
        if isinstance(bit4_window_sizes, list):
            bit4_window_sizes = bit4_window_sizes[0] if bit4_window_sizes else 0
        bit8_window_sizes = bit8_window_sizes or 0
        bit4_window_sizes = bit4_window_sizes or 0

    def attention_fn(
        module,
        query,
        key,
        value,
        attention_mask,
        dropout=0.0,
        scaling=1.0,
        sliding_window=None,
        **kwargs,
    ):
        if hasattr(module, "num_key_value_groups"):
            key = repeat_kv(key, module.num_key_value_groups)
            value = repeat_kv(value, module.num_key_value_groups)
            query = query.permute(0, 2, 1, 3)
        km = key.mean(dim=2, keepdim=True)
        key = key - km
        if args.qk_qtype == "int":
            bit8_qk_quantizer = IntegerQuantizer(8, True, "per_token")
            bit4_qk_quantizer = IntegerQuantizer(4, True, "per_token")
        elif args.qk_qtype == "e4m3":
            bit8_qk_quantizer = FloatQuantizer("e4m3", True, "per_token")
            bit4_qk_quantizer = IntegerQuantizer(4, True, "per_token")
        elif args.qk_qtype == "e5m2":
            bit8_qk_quantizer = FloatQuantizer("e5m2", True, "per_token")
            bit4_qk_quantizer = IntegerQuantizer(4, True, "per_token")
        else:
            raise ValueError(f"Invalid quantization type: {args.qk_qtype}")
        if args.v_qtype == "int":
            bit8_v_quantizer = IntegerQuantizer(8, True, "per_channel")
        elif args.v_qtype == "e4m3":
            bit8_v_quantizer = FloatQuantizer("e4m3", True, "per_channel")
        elif args.v_qtype == "e5m2":
            bit8_v_quantizer = FloatQuantizer("e5m2", True, "per_channel")
        else:
            raise ValueError(f"Invalid quantization type: {args.v_qtype}")
        q_bit8 = bit8_qk_quantizer.fake_quant_tensor(query)
        q_bit4 = bit4_qk_quantizer.fake_quant_tensor(query)
        k_bit8 = bit8_qk_quantizer.fake_quant_tensor(key)
        k_bit4 = bit4_qk_quantizer.fake_quant_tensor(key)
        v_bit8 = bit8_v_quantizer.fake_quant_tensor(value)
        seq_len = query.shape[2]
        if seq_len == 1:
            k_len = key.shape[2]
            batch_size, num_heads, _, head_dim = key.shape
            k_reconstructed = paddle.zeros_like(key)
            for head_idx in range(num_heads):
                if head_idx < len(bit8_window_sizes):
                    bit8_window = (
                        math.floor(bit8_window_sizes[head_idx] * k_len)
                    )
                else:
                    bit8_window = 0
                if head_idx < len(bit4_window_sizes):
                    bit4_window = (
                        math.floor(bit4_window_sizes[head_idx] * k_len)
                    )
                else:
                    bit4_window = 0
                sink_end = min(sink_window_size, k_len)
                if sink_end > 0:
                    k_reconstructed[:, head_idx, :sink_end, :] = k_bit8[
                        :, head_idx, :sink_end, :
                    ]
                if bit4_window > 0:
                    bit4_start = max(sink_end, k_len - bit4_window - bit8_window)
                    bit4_end = k_len - bit8_window
                    if bit4_end > bit4_start and bit4_start >= 0 and bit4_end <= k_len:
                        k_reconstructed[:, head_idx, bit4_start:bit4_end, :] = k_bit4[
                            :, head_idx, bit4_start:bit4_end, :
                        ]
                if bit8_window > 0:
                    bit8_start = k_len - bit8_window
                    bit8_end = k_len
                    if bit8_start >= 0 and bit8_start < bit8_end:
                        k_reconstructed[:, head_idx, bit8_start:bit8_end, :] = k_bit8[
                            :, head_idx, bit8_start:bit8_end, :
                        ]
            return sdpa_attention_forward(
                module,
                query,
                key,
                value,
                attention_mask=attention_mask,
                dropout=dropout,
                scaling=scaling,
                sliding_window=sliding_window,
                tensor_layout="HND",
                **kwargs,
            )
        attn_output = attn_hierarchical_relative_window(
            q_bit8,
            k_bit8,
            q_bit4,
            k_bit4,
            v_bit8,
            int8_window_ratios=bit8_window_sizes,
            int4_window_ratios=bit4_window_sizes,
            sink_size=sink_window_size,
            tensor_layout="HND",
            output_dtype=paddle.bfloat16,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        bs, seq_len, head, head_dim = attn_output.shape
        attn_output = attn_output.view(bs, seq_len, head * head_dim)
        return attn_output, None

    return attention_fn
