import triton
import triton.language as tl

import paddle

@triton.jit()
def compute_kernel(
    weight,
    query_ptr, indexer_cache,
    block_tables, 
    cu_seqlen_q, seq_lens_decoder,
    output, output_padding_len,
    MAX_BLOCKS_PER_SEQ: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    Q_NUM_HEAD: tl.constexpr,
    K_NUM_HEAD: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr):

    page_id = tl.program_id(axis=0)
    batch_id = tl.program_id(axis=1)
    q_head_id = tl.program_id(axis=2)

    kv_len_this_batch = tl.load(seq_lens_decoder+batch_id)
    if kv_len_this_batch <= 0:
        return
    else:
        kv_len_this_batch =kv_len_this_batch + 1
    cu_len_this_batch = tl.load(cu_seqlen_q+batch_id)

    if page_id * PAGE_SIZE >= kv_len_this_batch:
        return 

    q_weight = tl.load(weight + cu_len_this_batch * Q_NUM_HEAD + q_head_id)

    offset_m = tl.arange(0, BLOCK_SIZE_M)
    offset_k = tl.arange(0, HEAD_DIM)
    offset_n = tl.arange(0, PAGE_SIZE)

    read_query_ptr = query_ptr + cu_len_this_batch * Q_NUM_HEAD * HEAD_DIM
    read_query_ptr += (q_head_id + offset_m[:,None]) * HEAD_DIM + offset_k[None,:]
    mask = (offset_m[:, None] < 1)

    query = tl.load(read_query_ptr, mask=mask, other=0.0)
    # query = query * q_weight * (Q_NUM_HEAD ** -0.5) * (HEAD_DIM ** -0.5)
 
    block_id = tl.load(block_tables + batch_id * MAX_BLOCKS_PER_SEQ + page_id)
    read_key_ptr = indexer_cache + block_id * K_NUM_HEAD * PAGE_SIZE * HEAD_DIM
    read_key_ptr += q_head_id * PAGE_SIZE * HEAD_DIM
    read_key_ptr += offset_k[:,None] + offset_n[None,:] * HEAD_DIM

    key = tl.load(read_key_ptr)

    accumulator = tl.dot(query, key)
    accumulator = tl.where(accumulator >= 0, accumulator, 0)
    accumulator = accumulator * q_weight[:,None] * (Q_NUM_HEAD ** -0.5) * (HEAD_DIM ** -0.5)
 
    write_ptr = output + cu_len_this_batch * Q_NUM_HEAD * output_padding_len 
    write_ptr += q_head_id * output_padding_len + page_id * PAGE_SIZE
    write_ptr += offset_m[:,None] * output_padding_len + offset_n[None,:]

    mask2 = (page_id * PAGE_SIZE + offset_n[None,:]) < kv_len_this_batch
    tl.store(write_ptr, accumulator, mask=mask & mask2)
    # tl.store(write_ptr, accumulator, mask=mask)
 
def indexer_mha_page_logits(query, indexer_cache, weight, block_tables, cu_seqlen_q, seq_lens_decoder, max_model_len, select_first_tokens=64):
    assert query.dtype == paddle.bfloat16, f"{query.dtype} != bfloat16"
    assert query.is_contiguous(), "query is not contiguous"
    assert indexer_cache.dtype == paddle.bfloat16, f"{indexer_cache.dtype} != bfloat16"
    assert weight.dtype == paddle.bfloat16, f"{weight.dtype} != bfloat16"
    assert len(indexer_cache.shape) == 4, f"{indexer_cache.shape} != 4"
    k_num_head = indexer_cache.shape[1]
    block_size = indexer_cache.shape[2]
    head_dim = indexer_cache.shape[3]
    assert block_size == 64, f"{block_size}!= 64"

    assert query.shape[-1] % head_dim == 0, f"query.shape[-1] = {query.shape[-1]}, head_dim = {head_dim}"
    q_num_head = query.shape[-1] // head_dim
    assert q_num_head == k_num_head, f"(q_num_head{q_num_head} != k_num_head{k_num_head})"
    assert len(weight.shape) == 2, f"{weight.shape} != 2"
    assert weight.shape[0] == query.shape[0], f"(weight.shape[0]{weight.shape[0]} != query.shape[0]{query.shape[0]})"
    assert weight.shape[1] == q_num_head, f"(weight.shape[1]{weight.shape[1]} != q_num_head{q_num_head})"

    token_num = query.shape[0]
    real_bs = seq_lens_decoder.shape[0]
    max_bs = block_tables.shape[0]
    # print("real_bs",real_bs)
    # print("token_num",token_num)
    # assert token_num <= real_bs
    assert real_bs + 1 == cu_seqlen_q.shape[0], f"(real_bs + 1, cu_seqlen_q.shape[0]) = ({real_bs + 1}, {cu_seqlen_q.shape[0]})"
    assert real_bs <= max_bs, f"(real_bs, max_bs) = ({real_bs}, {max_bs})"

    # 记住要+1
    output_padding_len = max_model_len #paddle.max(seq_lens_decoder).item() + 1
    output_padding_len = (output_padding_len + block_size - 1) // block_size * block_size

    output = paddle.full([token_num*q_num_head, output_padding_len], - 1e10 , dtype='bfloat16')

    grid = (output_padding_len // block_size, real_bs, q_num_head)
    
    
    compute_kernel[grid](
        weight,
        query,
        indexer_cache,
        block_tables,
        cu_seqlen_q,
        seq_lens_decoder,
        output,
        output_padding_len,
        MAX_BLOCKS_PER_SEQ=block_tables.shape[1],
        PAGE_SIZE=block_size,
        HEAD_DIM=head_dim,
        Q_NUM_HEAD=q_num_head,
        K_NUM_HEAD=k_num_head,
        BLOCK_SIZE_M=16,
    )
    output[:, :select_first_tokens] = 1e10
    output[:, -1] = 1e10
    return output

def indexer_mha_page_logits_naive(query, indexer_cache, weight, block_tables, cu_seqlen_q, seq_lens_decoder):
    # breakpoint()
    assert query.dtype == paddle.bfloat16
    assert indexer_cache.dtype == paddle.bfloat16
    assert weight.dtype == paddle.bfloat16

    assert len(indexer_cache.shape) == 4
    k_num_head = indexer_cache.shape[1]
    block_size = indexer_cache.shape[2]
    head_dim = indexer_cache.shape[3]
    assert block_size == 64

    assert query.shape[-1] % head_dim == 0
    q_num_head = query.shape[-1] // head_dim
    assert q_num_head == k_num_head

    assert len(weight.shape) == 2
    assert weight.shape[0] == query.shape[0]
    assert weight.shape[1] == q_num_head

    token_num = query.shape[0]
    real_bs = seq_lens_decoder.shape[0]
    max_bs = block_tables.shape[0]
    # print("real_bs",real_bs)
    # print("token_num",token_num)
    # assert token_num <= real_bs
    assert real_bs + 1 == cu_seqlen_q.shape[0]
    assert real_bs <= max_bs

    # 记住要+1
    output_padding_len = paddle.max(seq_lens_decoder).item() + 1
    output_padding_len = (output_padding_len + block_size - 1) // block_size * block_size

    # query = query.reshape([-1,q_num_head,head_dim]).transpose([1,0,2]).reshape(-1,head_dim)

    retrived_k = paddle.zeros([token_num, k_num_head, output_padding_len, head_dim], dtype=query.dtype)

    for i in range(real_bs):
        this_k_len = seq_lens_decoder[i].item()
        if this_k_len <= 0:
            continue
        this_k_len += 1
        token_id = cu_seqlen_q[i].item()
        for j in range(0, this_k_len, block_size):

            start = j
            end = j+block_size

            block_id = block_tables[i, j // block_size].numpy().item()

            retrived_k[token_id,:,start:end,:] = indexer_cache[block_id,:,:,:]
    # breakpoint()
    query = query.reshape([token_num*q_num_head,head_dim])
    weight_naive = weight.reshape([-1,1])
    query = query * weight_naive
    retrived_k.reshape_([token_num*k_num_head, output_padding_len, head_dim])
    # breakpoint()
    baseline = paddle.einsum('ik, ilk->il', query,retrived_k)

    # baseline = paddle.nn.functional.relu(baseline)
    # breakpoint()
    # assert ((baseline - output).abs().max().item()) == 0

    return baseline 



# indexer_q = paddle.load("/root/paddlejob/workspace/env_run/output/changwenbin/Sparse_GQA/baidu/paddle_internal/test_data/indexer_q")
# indexer_cache = paddle.load("/root/paddlejob/workspace/env_run/output/changwenbin/Sparse_GQA/baidu/paddle_internal/test_data/indexer_cache")
# weights = paddle.load("/root/paddlejob/workspace/env_run/output/changwenbin/Sparse_GQA/baidu/paddle_internal/test_data/weights")
# block_tables = paddle.load("/root/paddlejob/workspace/env_run/output/changwenbin/Sparse_GQA/baidu/paddle_internal/test_data/block_tables")
# cu_seqlens_q = paddle.load("/root/paddlejob/workspace/env_run/output/changwenbin/Sparse_GQA/baidu/paddle_internal/test_data/cu_seqlens_q")
# seq_lens_decoder = paddle.load("/root/paddlejob/workspace/env_run/output/changwenbin/Sparse_GQA/baidu/paddle_internal/test_data/seq_lens_decoder")

# out = indexer_mha_page_logits(indexer_q, indexer_cache, weights, block_tables, cu_seqlens_q, seq_lens_decoder)
# # breakpoint()
# out1 = indexer_mha_page_logits_naive(indexer_q, indexer_cache, weights, block_tables, cu_seqlens_q, seq_lens_decoder)

# paddle.set_printoptions(precision=4, threshold=160, edgeitems=40, sci_mode=None, linewidth=80)
# print(out-out1)
# breakpoint()
# print(((out - out1).abs().max().item()))