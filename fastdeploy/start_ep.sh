
source /root/paddlejob/workspace/env_run/output/changwenbin/miniconda3/bin/activate /root/paddlejob/workspace/env_run/output/changwenbin/miniconda3/envs/cwb_full_310

MODEL_PATH=/root/paddlejob/workspace/models/DeepSeek-V3.2-Exp-BF16

export FD_DISABLE_CHUNKED_PREFILL=1
export FD_ATTENTION_BACKEND="MLA_ATTN"
export FLAGS_flash_attn_version=3


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export FD_ENABLE_MULTI_API_SERVER=1
python -m fastdeploy.entrypoints.openai.multi_api_server \
       --ports "8091,8092,8093,8094,8095,8096,8097,8098" \
       --num-servers 8 \
       --args --model "MODEL_PATH" \
       --ips "10.95.246.141,10.95.246.79" \
       --no-enable-prefix-caching \
       --quantization block_wise_fp8 \
       --disable-sequence-parallel-moe \
       --tensor-parallel-size 1 \
       --num-gpu-blocks-override 1024 \
       --data-parallel-size 16 \
       --max-model-len 16384 \
       --enable-expert-parallel \
       --max-num-seqs 20 \
       --graph-optimization-config '{"use_cudagraph":true}' \
