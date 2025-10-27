
from typing import Optional
import paddle
import paddle.nn.functional as F
from paddleformers.utils.log import logger
import triton
import pprint

# 导入你已经准备好的底层 Triton JIT Kernels
from .minimax_mamba_kernels import (
    _fwd_diag_kernel,
    _fwd_kv_parallel,
    _fwd_kv_reduce,
    _fwd_none_diag_kernel,
    _linear_attn_decode_kernel,
)

# 保留打印函数，用于调试
def print_tensor_stats(tensor, name):
    """打印Paddle张量的统计信息 (强制 float32)"""
    if tensor is None:
        logger.info(f"DEBUG_OPS_FD: {name} is None")
        return
    with paddle.no_grad():
        stats = {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        if tensor.numel() > 0:
            tensor_float = tensor.astype('float32')
            tensor_cpu = tensor_float.cpu()

            # ==================== 新增检查 ====================
            has_nan = paddle.any(paddle.isnan(tensor_cpu)).item()
            has_inf = paddle.any(paddle.isinf(tensor_cpu)).item()
            stats["has_nan"] = has_nan
            stats["has_inf"] = has_inf
            # ================================================

            # 只有在没有 nan/inf 的情况下才计算统计值
            if not has_nan and not has_inf:
                stats["max"] = f"{tensor_cpu.max().item():.6f}"
                stats["min"] = f"{tensor_cpu.min().item():.6f}"
                stats["mean"] = f"{tensor_cpu.mean().item():.6f}"
                stats["std"] = f"{tensor_cpu.std().item():.6f}"
            else:
                stats["max"] = "NaN/Inf Present"
                stats["min"] = "NaN/Inf Present"
                stats["mean"] = "NaN/Inf Present"
                stats["std"] = "NaN/Inf Present"

            # if tensor_float.ndim == 2:
            #     flat_data = tensor_cpu.numpy()[0, :5]
            # else:
            flat_data = tensor_cpu.flatten().numpy()[:5]
            stats["first_5_values"] = flat_data
        logger.info(f"\n--- [FD OPS DEBUG] {name} ---\n{pprint.pformat(stats, indent=2)}\n--------------------------\n")

# 这是一个基于 paddle.autograd.Function 的包装类，用于调用 Triton kernels
# 它的作用类似于 PyTorch 中的 torch.autograd.Function
class _Attention(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, q, k, v, s, kv_history_in):
        # 确保输入张量在内存中是连续的
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        s = s.contiguous()

        # ==================== 核心修复：在 PyLayer 内部创建副本 ====================
        # 创建一个与输入 kv_history 形状和类型相同的、用于计算的张量。
        # 这是为了避免原地修改传入的叶子节点张量。
        kv_history_compute = paddle.clone(kv_history_in)
        # =====================================================================

        # 获取输入维度
        b, h, n, d = q.shape
        e = v.shape[-1]

        # 初始化输出张量
        o = paddle.empty(shape=[b, h, n, e], dtype=q.dtype)

        # --- [后续所有 Triton Kernel 调用逻辑保持不变] ---
        # ... (设置 BLOCK, CBLOCK, 计算 k_decay 等) ...

        BLOCK = 256
        NUM_BLOCK = triton.cdiv(n, BLOCK)
        CBLOCK_DIAG = 32
        NUM_CBLOCK_DIAG = BLOCK // CBLOCK_DIAG
        assert BLOCK % CBLOCK_DIAG == 0
        array = paddle.arange(0, BLOCK) + 1
        array_float = array.astype("float32")
        k_decay = paddle.exp(-s * (BLOCK - array_float.reshape([1, -1])))

        # Step 1
        grid_diag = (b * h * NUM_BLOCK, NUM_CBLOCK_DIAG)
        _fwd_diag_kernel[grid_diag](
            q, k, v, o, s,
            b=b, h=h, n=n, d=d, e=e,
            BLOCK=BLOCK, NUM_BLOCK=NUM_BLOCK, CBLOCK=CBLOCK_DIAG,
        )
        print_tensor_stats(o, "PyLayer_After_DiagKernel")

        # Step 2
        NUM_FBLOCK = 1
        D_FBLOCK = d // NUM_FBLOCK
        E_FBLOCK = e // NUM_FBLOCK
        CBLOCK_KV_AND_NON_DIAG = 64
        NUM_CBLOCK_KV_AND_NON_DIAG = BLOCK // CBLOCK_KV_AND_NON_DIAG
        kv = paddle.empty(shape=[b, h, NUM_BLOCK, d, e], dtype="float32")
        grid_kv_parallel = (b * h, NUM_BLOCK)
        _fwd_kv_parallel[grid_kv_parallel](
            k, v, k_decay, kv,
            b=b, h=h, n=n, d=d, e=e,
            BLOCK=BLOCK, NUM_BLOCK=NUM_BLOCK,
            D_FBLOCK=D_FBLOCK, E_FBLOCK=E_FBLOCK, NUM_FBLOCK=NUM_FBLOCK,
            CBLOCK=CBLOCK_KV_AND_NON_DIAG, NUM_CBLOCK=NUM_CBLOCK_KV_AND_NON_DIAG,
        )
        print_tensor_stats(kv, "PyLayer_After_KVParallelKernel")

        # Step 3: 将新创建的 kv_history_compute 传入
        grid_kv_reduce = (b * h, NUM_FBLOCK)
        print_tensor_stats(kv_history_compute, "PyLayer_Before_KVReduceKernel_History")
        _fwd_kv_reduce[grid_kv_reduce](
            s, kv, kv_history_compute,  # <--- 使用副本
            b=b, h=h, n=n, d=d, e=e,
            BLOCK=BLOCK, NUM_BLOCK=NUM_BLOCK,
            D_FBLOCK=D_FBLOCK, E_FBLOCK=E_FBLOCK,
        )
        print_tensor_stats(kv, "PyLayer_After_KVReduceKernel_KV") 
        print_tensor_stats(kv_history_compute, "PyLayer_After_KVReduceKernel_History")


        # Step 4
        grid_none_diag = (b * h, NUM_BLOCK * NUM_CBLOCK_KV_AND_NON_DIAG)
        _fwd_none_diag_kernel[grid_none_diag](
            q, o, s, kv,
            b=b, h=h, n=n, d=d, e=e,
            BLOCK=BLOCK, NUM_BLOCK=NUM_BLOCK, E_FBLOCK=E_FBLOCK,
            CBLOCK=CBLOCK_KV_AND_NON_DIAG, NUM_CBLOCK=NUM_CBLOCK_KV_AND_NON_DIAG,
        )
        print_tensor_stats(o, "PyLayer_After_NoneDiagKernel")

        # 返回计算结果和被更新后的 kv_history 副本
        return o, kv_history_compute

    @staticmethod
    def backward(ctx, grad_output, grad_kv_history):
        raise NotImplementedError("Backward pass for lightning_attention is not implemented")

def lightning_attention(
    q: paddle.Tensor,
    k: paddle.Tensor,
    v: paddle.Tensor,
    slope_rate: paddle.Tensor, # 在 vLLM 中叫 ed
    kv_history: Optional[paddle.Tensor] = None,
    is_profiling: bool = False, 
    block_size: int = 256, 
) -> tuple[paddle.Tensor, paddle.Tensor]:

    # if is_profiling:
    #     logger.warning("<<<<< RUNNING in PROFILING MODE for LIGHTNING ATTENTION! >>>>>")
    #     logger.warning("<<<<< Bypassing actual computation and returning dummy tensors. >>>>>")
    #     dummy_output = paddle.zeros_like(v)
    #     dummy_kv_state = paddle.zeros_like(kv_history) if kv_history is not None else paddle.zeros(shape=[q.shape[0], q.shape[1], q.shape[3], v.shape[3]], dtype=v.dtype)
    #     return dummy_output, dummy_kv_state
    
    # if is_profiling:
    #     logger.warning("<<<<< RUNNING in PROFILING MODE for LIGHTNING ATTENTION! >>>>>")
    #     logger.warning("<<<<< Bypassing computation by returning input tensors directly. >>>>>")
        
    #     # --- [最终解决方案] ---
    #     # 1. 直接返回 v 作为 dummy_output。它的形状和类型都与期望输出完全一致。
    #     #    不进行任何计算，这是最安全的。
    #     dummy_output = v
        
    #     # 2. 检查 kv_history
    #     if kv_history is None:
    #         # 这段代码不应该被执行，但我们保留它作为最后一道防线。
    #         raise RuntimeError(
    #             "In profiling/CUDAGraph mode, kv_history cannot be None. "
    #             "The caller must provide a pre-allocated tensor."
    #         )
        
    #     # 3. 直接返回传入的 kv_history 作为 dummy_kv_state。
    #     #    它的形状和类型也与期望的输出状态完全一致。
    #     dummy_kv_state = kv_history
            
    #     return dummy_output, dummy_kv_state
    #     # --- [结束解决方案] ---

    logger.info("<<<<< RUNNING TRITON KERNEL FOR LIGHTNING ATTENTION! >>>>>")
    print_tensor_stats(q, "Kernel_Wrapper_Input_Q")
    print_tensor_stats(k, "Kernel_Wrapper_Input_K")
    print_tensor_stats(v, "Kernel_Wrapper_Input_V")

    d = q.shape[-1]
    e = v.shape[-1]

    if slope_rate.dim() == 1:
        slope_rate = slope_rate.reshape([1, -1, 1, 1])

    m = 128 if d >= 128 else 64
    if d % m != 0:
        raise ValueError(f"Head dimension d ({d}) must be divisible by chunk size m ({m})")

    arr = [m * i for i in range(d // m + 1)]
    if arr[-1] != d:
        arr.append(d)

    num_chunks = len(arr) - 1
    output = 0

    if kv_history is None:
        kv_history_for_loop = paddle.zeros(
            shape=[q.shape[0], q.shape[1], d, e], dtype="float32"
        )
    else:
        kv_history_for_loop = paddle.clone(kv_history).contiguous()

    logger.info(f">>> [DEBUG] Starting chunked attention computation. Total chunks: {num_chunks}")

    final_kv_state = None
    for i in range(num_chunks):
        s = arr[i]
        e_chunk = arr[i + 1]

        q_chunk = q[..., s:e_chunk]
        k_chunk = k[..., s:e_chunk]

        # 你的 _Attention.apply 需要能够处理切片后的 Q, K
        # 假设它返回的是一个完整的、更新后的 kv_history

        logger.info(f">>> [DEBUG] Processing chunk {i}: head_dim slice [{s}:{e_chunk}]")
        print_tensor_stats(q_chunk, f"Kernel_Chunk_{i}_Input_Q")
        print_tensor_stats(k_chunk, f"Kernel_Chunk_{i}_Input_K")
        print_tensor_stats(kv_history_for_loop, f"Kernel_Chunk_{i}_Input_KV_History")

        o_chunk, updated_full_kv_history = _Attention.apply(q_chunk, k_chunk, v, slope_rate, kv_history_for_loop)

        print_tensor_stats(o_chunk, f"Kernel_Chunk_{i}_Output_O")

        output = output + o_chunk

        # 更新 kv_history 以便下一次循环使用
        kv_history_for_loop = updated_full_kv_history
        final_kv_state = updated_full_kv_history

    print_tensor_stats(output, "Kernel_Final_Aggregated_Output_O")

    # 返回最终的累加输出和最后一次更新后的完整 kv 状态
    return output.astype(k.dtype), final_kv_state

# ----------------------------------------------------------------------------------
#  decode 函数保持不变，因为它已经使用了 Triton Kernel
# ----------------------------------------------------------------------------------
def linear_decode_forward_triton(
    q: paddle.Tensor,
    k: paddle.Tensor,
    v: paddle.Tensor,
    kv_caches: paddle.Tensor,
    slope_rate: paddle.Tensor,
    slot_idx: paddle.Tensor,
    BLOCK_SIZE: int = 32,
) -> paddle.Tensor:
    B, H, _, D = q.shape
    assert tuple(k.shape) == (B, H, 1, D), f"Shape of k is {k.shape}, expected {(B, H, 1, D)}"
    assert tuple(v.shape) == (B, H, 1, D), f"Shape of v is {v.shape}, expected {(B, H, 1, D)}"
    from einops import rearrange
    output = paddle.empty_like(q)
    grid = (B, H, triton.cdiv(D, BLOCK_SIZE))

    # ==================== 核心修复 ====================
    qkv_b_stride, qkv_h_stride = q.strides[0], q.strides[1]
    cache_b_stride, cache_h_stride, cache_d0_stride, cache_d1_stride = (kv_caches.strides[0], kv_caches.strides[1], kv_caches.strides[2], kv_caches.strides[3])
    # ================================================

    _linear_attn_decode_kernel[grid](q, k, v, kv_caches, slope_rate, slot_idx, output, D=D, qkv_b_stride=qkv_b_stride, qkv_h_stride=qkv_h_stride, cache_b_stride=cache_b_stride, cache_h_stride=cache_h_stride, cache_d0_stride=cache_d0_stride, cache_d1_stride=cache_d1_stride, BLOCK_SIZE=BLOCK_SIZE)
    output = rearrange(output, "b h n d -> b n (h d)")
    return output.squeeze(1).contiguous()