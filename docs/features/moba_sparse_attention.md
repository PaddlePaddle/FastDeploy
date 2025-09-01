# moba_sparse_attention

## Introduction

We propose Lite MoBA and improve it based on MoBA. Specifically, we still draw on the MoE structure to divide KV into multiple blocks, introduce a learnable MLP layer to adaptively select important blocks. We use Full Attention's 1D Max Pooling Attention Map as Ground Truth. Then, we employ KLDivLoss to distill and train the MLP layer weights. Lite MoBA can be directly applied to post - training, where only the weights of the MLP are learnable and the weights of the original model remain unchanged.

Compared to NSA or MoBA, our Lite MoBA is more scalable and pluggable, without the need to change traditional attention architectures or interfere with model weight training in the Pre - training and Post - training stages. It only requires a small amount of training on the MLP layer in the final stage of the model to achieve almost lossless accuracy. Since MoBA updates the weights of the entire model, even when Full Attention is automatically invoked for inputs shorter than BlockSize x BlockNum, it still cannot avoid the impact of model updates on the model's effectiveness in text processing. In contrast, our pluggable Lite MoBA can achieve Full Attention that is truly equivalent to that of the original model in short text scenarios.

Compared with MoBA, in terms of effectiveness, its use of Average Pooling to represent inter - block relationships appears relatively limited and has poor handling of outlier representations. Our ablation experiments also demonstrated that the effectiveness of Average Pooling is inferior to that of the learnable MLP. In terms of training performance, since only the MLP weights need to be updated and the model weights do not need to be updated, a large amount of video memory will be saved during training (which needs to be tested). In terms of inference performance, when the input length is 128K, Block Size = 1024, and Block Num = 16, the performance is improved by 322% compared to Flash Attention 3.

## Usage

```bash
export FD_ATTENTION_BACKEND="MOBA_ATTN"

python -m fastdeploy.entrypoints.openai.api_server
    --model baidu/ERNIE-4.5-300B-A47B-Paddle  \
    --port 8188 \
    --tensor-parallel-size 4 \
    --quantization wint4 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 8192 \
    --max-model-len 131072 \
    --max-num-seqs 32 \
    --moba-attention-config '{"moba_encoder_top_k_left": 60, "moba_encoder_top_k_right": 80, "moba_decoder_top_k_left": 100, "moba_decoder_top_k_right": 120}'
```
## Environmental Variables Description

* Setting `FD_ATTENTION_BACKEND="MOBA_ATTN"` enables MOBA sparse attention.
* `moba_encoder_top_k_left=60, moba_encoder_top_k_right=80` indicates that the range of top - k is between 80 and 100 when the encoder is sparse.
* `moba_decoder_top_k_left=100, moba_decoder_top_k_right=100` indicates that the range of top - k is between 120 and 140 when the decoder is sparse.
