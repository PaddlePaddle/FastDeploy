export LD_LIBRARY_PATH=/root/paddlejob/share-storage/gpfs/system-public/lizhenyun/miniconda3/envs/lzyun_h_attn_new/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH


ncu --clock-control=reset

# ncu  --target-processes all --set full -f -o attn_v35_t1_t2 -k regex:"multi_query_append_attention_warp1_4_kernel|decode_unified_attention_c16_kernel|merge_multi_chunks_decoder_kernel|merge_multi_chunks_v2_kernel|merge_chunks_kernel" python tests/operators/attention/benchmark_decode_attention_c16.py --op both


ncu  --target-processes all --set full -f -o b_c8_v7 -k regex:"multi_query_append_attention_c8_warp1_4_kernel|decode_unified_attention_c8_kernel|merge_multi_chunks_decoder_kernel|merge_multi_chunks_v2_kernel|merge_chunks_kernel" python tests/operators/attention/benchmark_decode_attention_c8.py

# ncu  --target-processes all --set full -f -o attn_c16_v6 -k regex:"multi_query_append_attention_warp1_4_kernel|decode_unified_attention_c16_kernel" python tests/operators/attention/benchmark_decode_attention_c16.py --op both
