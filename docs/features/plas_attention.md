[简体中文](../zh/features/plas_attention.md)

# PLAS

## Introduction

We propose **PLAS (Pluggable Lightweight Attention for Sparsity)**, an improvement over MoBA. Specifically, we adopt an MoE-inspired structure that partitions KV into multiple blocks and introduces a learnable MLP layer to adaptively select important blocks. PLAS can be directly applied during post-training, where only the MLP weights are learnable, and the original model weights remain unchanged.

Compared to NSA/MoBA, our PLAS offers greater scalability and pluggability. It does not require modifying the traditional attention architecture or interfering with model weight training during pre-training or post-training. Only a small amount of training for the MLP layer is needed at the final stage to achieve nearly lossless accuracy. Since NSA/MoBA updates the entire model weights, it inevitably affects performance on short texts—even though it automatically switches to full attention when the input length is shorter than BlockSize × Top-K. In contrast, our PLAS can achieve truly equivalent full attention to the original model in short-text scenarios.

In terms of training efficiency, the training cost is very low because only the MLP weight needs to be updated. For inference performance, when the input length is 128K, Block Size = 128, and Top-K = 55, PLAS achieves a **386% speedup** compared to Flash Attention 3.

## Method

### Training

Following the approaches of NSA and MoBA, we partition the KV into multiple blocks. During both the prefill and decode stages, instead of performing attention computation over all KV, we dynamically select the top-K blocks with the highest attention scores for each query token, thereby enabling efficient sparse attention computation.

<div align="center">
<img src="./images/plas_training_distill.png" alt="Attention Gate Module" width="60%">
</div>

* **Attention Gate Module**: As illustrated in the figure above, to estimate the importance of each block with low computational overhead, we design a lightweight attention gate module. This module first compresses each K block via a MLP layer to generate a representative low-dimensional representation: $K_c^T=W_{kp}K^T$, where $W_{kp}$ denotes the MLP layer weights. Compared to directly applying mean pooling, the learnable MLP can more effectively capture semantic relationships and importance distributions among different tokens, thereby providing a refined representation of each block. After obtaining the compressed representation $K_c$, the importance of each query token with respect to each block is estimated via: $Softmax(Q\cdot K_c^T)$. To enhance the discriminative ability of the MLP layer, we use the full attention result after 1D max pooling $1DMaxPooling(Softmax(Q \cdot K^T))$ as the ground truth. By minimizing the distribution divergence between the two, the MLP layer is guided to learn feature representations that better align with the true attention distribution.
* **Training Data**: Benefiting from the efficiency of both the model architecture and the training paradigm, our approach achieves near-lossless precision with only 1B tokens used for training. The training data is sourced from an internally constructed mixed corpus containing both long and short texts, thereby enhancing the module’s adaptability to varying sequence lengths.
* **Other**: We observe that the final decode layer has a significant impact on the overall model accuracy. Therefore, during training, we exclude this layer from sparse attention computation and revert to full attention for this layer during inference.

### Inference

During sparse attention computation, each query token may dynamically select different KV blocks, leading to highly irregular memory access patterns in HBM. It is feasible to simply process each query token separately, but it will lead to excessively fine-grained computing, which cannot make full use of the tensor core, thus significantly reducing the GPU computing efficiency.

<div align="center">
<img src="./images/plas_inference_union.png" alt="Token/Head Union" width="60%">
</div>

To optimize performance in both the prefill and decode stages, we design a special joint strategy to adapt to their respective characteristics:

* **Prefill Token Union**: We observe that adjacent query tokens tend to select similar key blocks. Leveraging this locality, we take the union of the key blocks selected by consecutive 128 query tokens and jointly compute sparse attention for these tokens.
* **Decode Head Union**: Given the widespread adoption of GQA in modern models, we find that different heads within the same group often select overlapping key blocks. Thus, we combine the key blocks selected by all query heads within a group into a unified set and jointly calculate sparse attention. This way also reduces memory access overhead and further improves decoding efficiency.
* **Top-K Selection**: Conventional top-k algorithms based on sorting or direct calls to the cub library introduce significant runtime overhead. To mitigate this, we implemented an approximate top-k selection algorithm using binary search, which significantly reduces latency while maintaining accuracy, ultimately achieving significantly improved performance.

## Evaluation

### Experiments

We evaluated the precision of full attention and sparse attention on LongBenchV2 and Ruler (with context lengths of 32K, 64K, and 128K).

<table style="border-collapse: collapse; width: 100%;">
    <tr>
        <td rowspan="4" style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;">
            <strong>Model</strong>
        </td>
        <td colspan="8" style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;">
            <strong>Precision</strong>
        </td>
    </tr>
    <tr>
        <td colspan="4" style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;">
            <strong>FullAttention</strong>
        </td>
        <td colspan="4" style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;">
            <strong>SparseAttention</strong>
        </td>
    </tr>
    <tr>
        <td rowspan="2" style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;">
            <strong>LongBenchV2</strong>
        </td>
        <td colspan="3" style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;">
            <strong>Ruler</strong>
        </td>
        <td rowspan="2" style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;">
            <strong>LongBenchV2</strong>
        </td>
        <td colspan="3" style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;">
            <strong>Ruler</strong>
        </td>
    </tr>
    <tr>
        <td style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;">
            <strong>32K</strong>
        </td>
        <td style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;">
            <strong>64K</strong>
        </td>
        <td style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;">
            <strong>128K</strong>
        </td>
        <td style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;">
            <strong>32K</strong>
        </td>
        <td style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;">
            <strong>64K</strong>
        </td>
        <td style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;">
            <strong>128K</strong>
        </td>
    </tr>
    <tr>
        <td style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;">
            <strong>ERNIE-4.5-21B-A3B</strong>
        </td>
        <td style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;">31.48</td>
        <td style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;">76.74</td>
        <td style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;">56.40</td>
        <td style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;">25.48</td>
        <td style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;">31.45</td>
        <td style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;">75.93</td>
        <td style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;">55.38</td>
        <td style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;">25.05</td>
    </tr>
    <tr>
        <td style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;">
            <strong>ERNIE-4.5-300B-A47B</strong>
        </td>
        <td style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;">41.02</td>
        <td style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;">94.70</td>
        <td style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;">83.56</td>
        <td style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;">58.18</td>
        <td style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;">41.05</td>
        <td style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;">94.50</td>
        <td style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;">82.32</td>
        <td style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;">57.85</td>
    </tr>
</table>

### Performance

We selected a subset (longbook_sum_eng) from InfiniteBench as the performance evaluation dataset. For inputs exceeding 128K in length, we truncate the sequence by keeping the first 64K and the last 64K tokens.

<table style="border-collapse: collapse; width: 100%;">
    <tr>
        <td style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;"></td>
        <td style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;"></td>
        <td style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;"><strong>QPS</strong></td>
        <td style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;"><strong>Decode Speed (token/s)</strong></td>
        <td style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;"><strong>Time to First token(s)</strong></td>
        <td style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;"><strong>Time per Output Token(ms)</strong></td>
        <td style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;"><strong>End-to-End Latency(s)</strong></td>
        <td style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;"><strong>Mean Input<br>Length</strong></td>
        <td style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;"><strong>Mean Output Length</strong></td>
    </tr>
    <tr>
        <td rowspan="2" style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;">
            <strong>ERNIE-4.5-21B-A3B</strong>
        </td>
        <td style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;">
            <strong>FullAttention</strong>
        </td>
        <td style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;">0.101</td>
        <td style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;">13.32</td>
        <td style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;">8.082</td>
        <td style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;">87.05</td>
        <td style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;">61.400</td>
        <td style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;">113182.32</td>
        <td style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;">627.76</td>
    </tr>
    <tr>
        <td style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;">
            <strong>SparseAttention</strong>
        </td>
        <td style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;">0.150(+48%)</td>
        <td style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;">18.12(+36%)</td>
        <td style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;">5.466(-48%)</td>
        <td style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;">66.35(-31%)</td>
        <td style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;">42.157(-46%)</td>
        <td style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;">113182.32</td>
        <td style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;">590.23</td>
    </tr>
    <tr>
        <td rowspan="2" style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;">
            <strong>ERNIE-4.5-300B-A47B</strong>
        </td>
        <td style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;">
            <strong>FullAttention</strong>
        </td>
        <td style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;">0.066</td>
        <td style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;">5.07</td>
        <td style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;">13.812</td>
        <td style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;">206.70</td>
        <td style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;">164.704</td>
        <td style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;">113182.32</td>
        <td style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;">725.97</td>
    </tr>
    <tr>
        <td style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;">
            <strong>SparseAttention</strong>
        </td>
        <td style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;">0.081(+23%)</td>
        <td style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;">6.75(+33%)</td>
        <td style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;">10.584(-30%)</td>
        <td style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;">154.84(-34%)</td>
        <td style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;">132.745(-24%)</td>
        <td style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;">113182.32</td>
        <td style="border: 1px solid #dcdde0; padding: 8px; text-align: center; vertical-align: middle;">748.25</td>
    </tr>
</table>

## Usage

```
export FD_ATTENTION_BACKEND="PLAS_ATTN"

python -m fastdeploy.entrypoints.openai.api_server
    --model baidu/ERNIE-4.5-300B-A47B-Paddle  \
    --port 8188 \
    --tensor-parallel-size 4 \
    --quantization wint4 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 8192 \
    --max-model-len 131072 \
    --max-num-seqs 32 \
    --plas-attention-config '{"plas_encoder_top_k_left": 50, "plas_encoder_top_k_right": 60, "plas_decoder_top_k_left": 100, "plas_decoder_top_k_right": 120}'
```

**Note**: If sparse attention is enabled, the system will automatically load the MLP weights from `plas_attention_mlp_weight.safetensors` in the weight directory. If the MLP weight file is not found, mean pooling will be applied to the key representations.

**Parameter Description:**

* Setting `FD_ATTENTION_BACKEND="PLAS_ATTN"` enables PLAS sparse attention.
* `plas_encoder_top_k_left=50, plas_encoder_top_k_right=60` indicates that the range of top-k is between 50 and 60 when the encoder is sparse.
* `plas_decoder_top_k_left=100, plas_decoder_top_k_right=120` indicates that the range of top-k is between 100 and 120 when the decoder is sparse.
