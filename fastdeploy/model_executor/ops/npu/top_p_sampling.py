import paddle

def top_p_sampling_npu(probs, top_p):
    sorted_probs = paddle.sort(probs, descending=True)
    sorted_indices = paddle.argsort(probs, descending=True)
    cumulative_probs = paddle.cumsum(sorted_probs, axis=-1)

    # Remove tokens with cumulative probs above the top_p, But keep at
    # least min_tokens_to_keep tokens
    sorted_indices_to_remove = cumulative_probs > top_p

    # Keep the first token
    sorted_indices_to_remove = paddle.cast(
        sorted_indices_to_remove, dtype='int64'
    )

    sorted_indices_to_remove = paddle.static.setitem(
        sorted_indices_to_remove,
        (slice(None), slice(1, None)),
        sorted_indices_to_remove[:, :-1].clone(),
    )
    sorted_indices_to_remove = paddle.static.setitem(
        sorted_indices_to_remove, (slice(None), 0), 0
    )

    # Scatter sorted tensors to original indexing
    batch_size = probs.shape[0]
    vocab_size = probs.shape[-1]
    
    # Create flat indices for scatter operation
    batch_offsets = paddle.arange(batch_size).unsqueeze(-1) * vocab_size
    flat_sorted_indices = (sorted_indices + batch_offsets).flatten()
    
    # Perform scatter operation
    condition = paddle.scatter(
        paddle.zeros(batch_size * vocab_size, dtype=sorted_indices_to_remove.dtype),
        flat_sorted_indices,
        sorted_indices_to_remove.flatten(),
    )
    condition = paddle.cast(condition, 'bool').reshape(probs.shape)
    probs = paddle.where(condition, paddle.full_like(probs, 0.0), probs)
    next_tokens = paddle.multinomial(probs)
    next_scores = paddle.index_sample(probs, next_tokens)
    return next_scores, next_tokens
