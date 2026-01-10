from typing import Optional, Tuple
import paddle


def repeat_kv(hidden_states: paddle.Tensor, n_rep: int) -> paddle.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def sdpa_attention_forward(
    module: paddle.nn.Layer,
    query: paddle.Tensor,
    key: paddle.Tensor,
    value: paddle.Tensor,
    attention_mask: Optional[paddle.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    tensor_layout: str = "NHD",
    **kwargs
) -> Tuple[paddle.Tensor, None]:    
    if tensor_layout == "NHD":
        query = query.permute(0, 2, 1, 3)  # (B, N, H, D) -> (B, H, N, D)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)
    if hasattr(module, "num_key_value_groups") and key.shape[1] != query.shape[1]:
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)
    causal_mask = attention_mask
    if attention_mask is not None and causal_mask.ndim == 4:
        causal_mask = causal_mask[:, :, :, : key.shape[-2]]
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    if is_causal is None:
        is_causal = query.shape[2] > 1 and causal_mask is None
    if paddle.is_tensor(is_causal):
        is_causal = is_causal.item()
    
    query = query * scaling
    # if tensor_layout == "NHD":
    query = query.permute(0, 2, 1, 3)   # (B, H, N, D) -> (B, N, H, D)
    key = key.permute(0, 2, 1, 3)       # (B, H, N, D) -> (B, N, H, D)
    value = value.permute(0, 2, 1, 3)   # (B, H, N, D) -> (B, N, H, D)
    # import pdb; pdb.set_trace()
    attn_output = paddle.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=causal_mask,
        dropout_p=dropout,
        is_causal=is_causal,
    )
    # attn_output = attn_output.transpose(1, 2).contiguous()
    bs, seq_len, head, head_dim = attn_output.shape
    attn_output = attn_output.view(bs, seq_len, head * head_dim)
    # attn_output = attn_output.contiguous()
    return attn_output, None
