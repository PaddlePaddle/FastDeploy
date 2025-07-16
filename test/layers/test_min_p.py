# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import matplotlib.pyplot as plt
import numpy as np
import paddle
import paddle.nn.functional as F
import torch
from tqdm import tqdm

from fastdeploy.model_executor.ops.gpu import min_p_sampling

sample_time = 1000000
vocab_size = 1000
min_p_value = 0.5
batch_size = 3
batch_min_p_values = [0.1, 0.5, 0.9]
batch_min_p_values2=[0,3,0,0,0.4]


def compress(data):
    new_data = np.array([0, 0, 0], dtype=float)
    new_data[0] = data[0]
    new_data[1] = data[1]
    new_data[2] = np.sum(data[2:])
    return new_data


def plot_bar_chart(data1, data2, data3, title, request_idx=None):
    plt.figure(figsize=(6, 6))
    bar_width = 0.2
    idx = np.arange(len(data1)).astype(float)

    bars1 = plt.bar(idx - bar_width, data1, width=bar_width, color='salmon', label='Original Probability', alpha=0.9)
    bars2 = plt.bar(idx, data2, width=bar_width, color='skyblue', label='Sampled Probability', alpha=0.9)
    bars3 = plt.bar(idx + bar_width, data3, width=bar_width, color='orange', label='Normalized Original Probability', alpha=0.9)

    plt.bar_label(bars1, label_type='edge', padding=3, fmt='%.3f', fontsize=5, color='black')
    plt.bar_label(bars2, label_type='edge', padding=3, fmt='%.3f', fontsize=5, color='red')
    plt.bar_label(bars3, label_type='edge', padding=3, fmt='%.3f', fontsize=5, color='blue')

    full_title = title if request_idx is None else f"{title} (min_p={batch_min_p_values[request_idx]})"
    plt.title(full_title, fontsize=14)
    plt.xlabel("Index", fontsize=12)
    plt.ylabel("Probability", fontsize=12)
    plt.ylim(0, 1.1)
    plt.xlim(-1, 3)
    plt.xticks(range(0, 3, 1))
    plt.legend(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    output_path = f"{title.replace(' ', '_')}{'' if request_idx is None else f'_req{request_idx}'}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.clf()

def plot_low_prob_curve(low_prob_token_probs, sample_time, title, request_idx=None):
    plt.figure(figsize=(6, 6))
    plt.plot(np.arange(0, sample_time), low_prob_token_probs, marker='', linestyle='-', linewidth=1, color='blue')
    plt.xlabel('Sample Times')
    plt.ylabel('Probability')
    full_title = 'Probability of Low-Probability Tokens' if request_idx is None else f"Low-Probability Tokens (min_p={batch_min_p_values[request_idx]})"
    plt.title(full_title)
    plt.grid(alpha=0.3)
    output_path = f"{title.replace(' ', '_')}_low_prob{'' if request_idx is None else f'_req{request_idx}'}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.clf()

# min_p:0.5：FastDeploy
def fastdeploy_min_p_sampling():
    logits = paddle.ones(shape=[1, vocab_size], dtype="float32")
    logits[0][0] = 10
    logits[0][1] = 8
    low_prob_tensor = paddle.linspace(2.0, 0.0, vocab_size - 2)
    logits[0][2:] = low_prob_tensor

    probs = F.softmax(logits)
    min_p = paddle.to_tensor([min_p_value], dtype="float32")

    max_prob = probs.max().item()
    threshold = max_prob * min_p.item()
    allowed_tokens = paddle.where(probs[0] >= threshold)[0].numpy()

    sample_freq = [0] * vocab_size
    low_prob_token_times = 0
    low_prob_token_probs = []

    for i in tqdm(range(sample_time), desc="FastDeploy Sampling"):
        ids = min_p_sampling(probs, min_p, seed=-1)
        sample_freq[ids.item()] += 1
        if ids.item() >= 2:
            low_prob_token_times += 1
        low_prob_token_probs.append(low_prob_token_times / (i + 1))

    sample_freq = np.array(sample_freq, dtype=float) / sample_time
    low_prob_token_probs = np.array(low_prob_token_probs, dtype=float)

    ori_data1 = probs.numpy().reshape(-1)
    data1 = compress(ori_data1)
    data2 = compress(sample_freq)

    allowed_probs = probs[0, allowed_tokens].numpy()
    norm_scale = np.sum(allowed_probs)
    data3 = np.zeros_like(data1)
    for idx in allowed_tokens:
        if idx < 2:
            data3[idx] = ori_data1[idx] / norm_scale
        else:
            data3[2] += ori_data1[idx] / norm_scale

    plot_bar_chart(data1, data2, data3, "FastDeploy[min_p_sampling]")
    plot_low_prob_curve(low_prob_token_probs, sample_time, "FastDeploy[min_p_sampling]")

    return data2, data3


# batch:[0.1.0,5,0.9]：FastDeploy
def fastdeploy_batch_min_p_sampling(batch_size, min_p_values):
    logits = paddle.ones(shape=[batch_size, vocab_size], dtype="float32")
    for b in range(batch_size):
        logits[b][0] = 10
        logits[b][1] = 8
        logits[b][2:] = paddle.linspace(2.0, 0.0, vocab_size - 2)

    probs = F.softmax(logits, axis=-1)
    min_p_arr = paddle.to_tensor(min_p_values, dtype="float32")

    allowed_tokens_list = []
    for b in range(batch_size):
        max_prob = probs[b].max().item()
        threshold = max_prob * min_p_values[b]
        allowed_tokens = paddle.where(probs[b] >= threshold)[0].numpy()
        allowed_tokens_list.append(allowed_tokens)

    sample_freq = [np.zeros(vocab_size, dtype=float) for _ in range(batch_size)]
    low_prob_token_times = [0] * batch_size
    low_prob_token_probs = [[] for _ in range(batch_size)]

    for i in tqdm(range(sample_time), desc="FastDeploy Batch Sampling"):
        ids = min_p_sampling(probs, min_p_arr, seed=-1)
        for b in range(batch_size):
            sample_freq[b][ids[b].item()] += 1
            if ids[b].item() >= 2:
                low_prob_token_times[b] += 1
            low_prob_token_probs[b].append(low_prob_token_times[b] / (i + 1))

    data2_list = []
    data3_list = []
    for b in range(batch_size):
        sample_freq_b = sample_freq[b] / sample_time
        low_prob_token_probs[b] = np.array(low_prob_token_probs[b], dtype=float)

        ori_data1 = probs[b].numpy()
        data1 = compress(ori_data1)
        data2 = compress(sample_freq_b)
        data2_list.append(data2)

        allowed_probs = probs[b, allowed_tokens_list[b]].numpy()
        norm_scale = np.sum(allowed_probs)
        data3 = np.zeros_like(data1)
        for idx in allowed_tokens_list[b]:
            if idx < 2:
                data3[idx] = ori_data1[idx] / norm_scale
            else:
                data3[2] += ori_data1[idx] / norm_scale
        data3_list.append(data3)

        plot_bar_chart(data1, data2, data3, "FastDeploy[min_p_batch_sampling]", b)
        plot_low_prob_curve(low_prob_token_probs[b], sample_time, "FastDeploy[min_p_batch_sampling]", b)

    return data2_list, data3_list


def main():
    print("Running single min_p sampling (min_p=0.5)...")
    data2_fastdeploy, data3_fastdeploy = fastdeploy_min_p_sampling()

    print("\nFastDeploy Single Request Results:")
    print(f"Sampled Probability: {data2_fastdeploy}")
    print(f"Theoretical Normalized Probability: {data3_fastdeploy}")

    print("\nRunning batch min_p sampling (min_p=[0.1, 0.5, 0.9])...")
    data2_fd_batch, data3_fd_batch = fastdeploy_batch_min_p_sampling(batch_size, batch_min_p_values)

    data2_fd_batch,data3_fd_batch = fastdeploy_batch_min_p_sampling(batch_size,batch_min_p_values2)

    for b in range(batch_size):
        print(f"\nBatch Request {b} (min_p={batch_min_p_values[b]}):")
        print(f"FastDeploy - Sampled: {data2_fd_batch[b]}, Normalized: {data3_fd_batch[b]}")

if __name__ == "__main__":
    if paddle.device.is_compiled_with_cuda() and torch.cuda.is_available():
        main()
