#!/usr/bin/env python3
"""
直接测试 dsmla_write_cache 函数
"""


import paddle

# 设置GPU设备
paddle.set_device("gpu")

print("=" * 60)
print("DS MLA WriteCache 直接测试")
print("=" * 60)

# 直接导入并测试
import fastdeploy.model_executor.ops.gpu as gpu_ops

print("检查可用的writecache函数:")
for item in dir(gpu_ops):
    if "write" in item.lower() and "cache" in item.lower():
        print(f"  - {item}")

# 检查具体函数存在性
print("\n检查 dsmla_write_cache 函数:")
if hasattr(gpu_ops, "dsmla_write_cache"):
    func = gpu_ops.dsmla_write_cache
    print(f"✓ dsmla_write_cache 存在: {type(func)}")

    # 检查文档
    try:
        doc = func.__doc__
        if doc:
            print(f"  文档: {doc}")
        else:
            print("  无文档")
    except:
        print("  无法访问文档")
else:
    print("✗ dsmla_write_cache 不存在")

    # 检查备用名称
    print("\n检查可能的备用名称:")
    for item in dir(gpu_ops):
        if "ds" in item.lower() and "mla" in item.lower() and "write" in item.lower():
            print(f"  - {item}: {type(getattr(gpu_ops, item))}")

# 创建测试数据
print("\n创建测试数据...")
batch_size = 2
num_tokens = 16
max_num_blocks = 100
block_size = 16
kv_lora_rank = 512
pe_dim = 64
entry_size = kv_lora_rank + 16 + pe_dim * 2

compressed_kv = paddle.randn([num_tokens, kv_lora_rank], dtype="bfloat16")
k_pe = paddle.randn([num_tokens, pe_dim], dtype="bfloat16")
latent_cache = paddle.zeros([max_num_blocks, 1, block_size, entry_size], dtype="uint8")
slot_mapping = paddle.randint(0, max_num_blocks * block_size, [num_tokens], dtype="int64")
seq_lens_this_time = paddle.to_tensor([num_tokens // 2] * batch_size, dtype="int32")
seq_lens_decoder = paddle.zeros([batch_size], dtype="int32")
batch_id_per_token = paddle.concat(
    [paddle.zeros([num_tokens // 2], dtype="int32"), paddle.ones([num_tokens // 2], dtype="int32")]
)
cu_seqlens_q = paddle.to_tensor([0, num_tokens // 2, num_tokens], dtype="int32")
block_tables = paddle.randint(0, max_num_blocks, [batch_size, 10], dtype="int32")
scale = paddle.randn([num_tokens, 1], dtype="float32")

print("✓ 测试数据创建完成")

# 直接调用函数（如果存在）
if hasattr(gpu_ops, "dsmla_write_cache"):
    print("\n测试 dsmla_write_cache 调用...")
    try:
        result = gpu_ops.dsmla_write_cache(
            compressed_kv,
            k_pe,
            latent_cache,
            slot_mapping,
            seq_lens_this_time,
            seq_lens_decoder,
            batch_id_per_token,
            cu_seqlens_q,
            block_tables,
            None,  # kv_signal_data
            scale,
            "fp8_ds_mla",  # 根据之前测试，这是正确值
            4096,  # max_seq_len
            True,  # is_prefill
        )
        print("✓ dsmla_write_cache 调用成功！")
        print(f"  返回形状: {result.shape}")
        print(f"  返回类型: {type(result)}")

    except Exception as e:
        print(f"✗ dsmla_write_cache 调用失败: {e}")
        import traceback

        traceback.print_exc()

# 测试其他相关的writecache函数
print("\n测试其他writecache函数...")

if hasattr(gpu_ops, "prefill_mla_write_cache"):
    print("\n测试 prefill_mla_write_cache...")
    try:
        result2 = gpu_ops.prefill_mla_write_cache(
            compressed_kv,
            k_pe,
            latent_cache,
            seq_lens_this_time,  # 注意参数顺序不同
            seq_lens_decoder,
            batch_id_per_token,
            cu_seqlens_q,
            block_tables,
            None,  # kv_signal_data
            "none",  # scale_fmt
            4096,  # max_input_length
        )
        print("✓ prefill_mla_write_cache 调用成功")
    except Exception as e:
        print(f"✗ prefill_mla_write_cache 调用失败: {e}")

if hasattr(gpu_ops, "decode_mla_write_cache"):
    print("\n测试 decode_mla_write_cache...")
    try:
        result3 = gpu_ops.decode_mla_write_cache(
            compressed_kv[:2],
            k_pe[:2],
            latent_cache,
            paddle.to_tensor([1, 1], dtype="int32"),
            seq_lens_decoder,
            batch_id_per_token[:2],
            paddle.to_tensor([0, 1, 2], dtype="int32"),
            block_tables,
            "none",
            4096,
            False,
        )
        print("✓ decode_mla_write_cache 调用成功")
    except Exception as e:
        print(f"✗ decode_mla_write_cache 调用失败: {e}")

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)
