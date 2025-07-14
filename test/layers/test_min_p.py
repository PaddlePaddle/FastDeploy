import flashinfer.sampling
import matplotlib.pyplot as plt
import numpy as np
import paddle
import paddle.nn.functional as F
import torch
from tqdm import tqdm

from fastdeploy.model_executor.ops.gpu import min_p_sampling

# 设置测试参数
sample_time = 1000000
vocab_size = 1000
min_p_value = 0.5

# 定义压缩函数
def compress(data):
    new_data = np.array([0, 0, 0], dtype=float)
    new_data[0] = data[0]
    new_data[1] = data[1]
    new_data[2] = np.sum(data[2:])
    return new_data

# FastDeploy 采样函数
def fastdeploy_min_p_sampling():
    # 创建 logits
    logits = paddle.ones(shape=[1, vocab_size], dtype="float32")
    logits[0][0] = 10
    logits[0][1] = 8
    low_prob_tensor = paddle.linspace(2.0, 0.0, vocab_size - 2)
    logits[0][2:] = low_prob_tensor

    # 计算概率
    probs = F.softmax(logits)
    min_p = paddle.to_tensor([min_p_value], dtype="float32")

    # 计算允许的 token
    max_prob = probs.max().item()
    threshold = max_prob * min_p.item()
    allowed_tokens = paddle.where(probs[0] >= threshold)[0].numpy()

    # 初始化统计变量
    sample_freq = [0] * vocab_size
    low_prob_token_times = 0
    low_prob_token_probs = []

    # 执行采样
    for i in tqdm(range(sample_time), desc="FastDeploy Sampling"):
        ids = min_p_sampling(probs, min_p, seed=-1)
        sample_freq[ids.item()] += 1
        if ids.item() >= 2:
            low_prob_token_times += 1
        low_prob_token_probs.append(low_prob_token_times / (i + 1))

    # 处理采样结果
    sample_freq = np.array(sample_freq, dtype=float) / sample_time
    low_prob_token_probs = np.array(low_prob_token_probs, dtype=float)

    # 压缩数据
    ori_data1 = probs.numpy().reshape(-1)
    data1 = compress(ori_data1)  # 原始概率
    data2 = compress(sample_freq)  # 采样概率

    # 计算理论归一化概率
    allowed_probs = probs[0, allowed_tokens].numpy()
    norm_scale = np.sum(allowed_probs)
    data3 = np.zeros_like(data1)
    for idx in allowed_tokens:
        if idx < 2:
            data3[idx] = ori_data1[idx] / norm_scale
        else:
            data3[2] += ori_data1[idx] / norm_scale

    # 绘制柱状图
    plot_bar_chart(data1, data2, data3, "FastDeploy[min_p_sampling]")
    # 绘制曲线图
    plot_low_prob_curve(low_prob_token_probs, sample_time, "FastDeploy[min_p_sampling]")

    return data2, data3

# vLLM (FlashInfer) 采样函数
def vllm_min_p_sampling():
    # 创建 logits
    logits = torch.ones((1, vocab_size), dtype=torch.float32).cuda()
    logits[0][0] = 10
    logits[0][1] = 8
    low_prob_tensor = torch.linspace(2.0, 0.0, vocab_size - 2).cuda()
    logits[0][2:] = low_prob_tensor

    # 计算概率
    probs = torch.softmax(logits, dim=-1)
    min_p = torch.tensor([min_p_value], dtype=torch.float32).cuda()

    # 计算允许的 token
    max_prob = probs.max().item()
    threshold = max_prob * min_p.item()
    allowed_tokens = torch.where(probs[0] >= threshold)[0].cpu().numpy()

    # 初始化统计变量
    sample_freq = [0] * vocab_size
    low_prob_token_times = 0
    low_prob_token_probs = []

    # 执行采样
    for i in tqdm(range(sample_time), desc="FlashInfer Sampling"):
        ids = flashinfer.sampling.min_p_sampling_from_probs(probs, min_p, deterministic=False)
        sample_freq[ids.item()] += 1
        if ids.item() >= 2:
            low_prob_token_times += 1
        low_prob_token_probs.append(low_prob_token_times / (i + 1))

    # 处理采样结果
    sample_freq = np.array(sample_freq, dtype=float) / sample_time
    low_prob_token_probs = np.array(low_prob_token_probs, dtype=float)

    # 压缩数据
    ori_data1 = probs.cpu().numpy().reshape(-1)
    data1 = compress(ori_data1)  # 原始概率
    data2 = compress(sample_freq)  # 采样概率

    # 计算理论归一化概率
    allowed_probs = probs[0, allowed_tokens].cpu().numpy()
    norm_scale = np.sum(allowed_probs)
    data3 = np.zeros_like(data1)
    for idx in allowed_tokens:
        if idx < 2:
            data3[idx] = ori_data1[idx] / norm_scale
        else:
            data3[2] += ori_data1[idx] / norm_scale

    # 绘制柱状图
    plot_bar_chart(data1, data2, data3, "vLLM[min_p_sampling]")
    # 绘制曲线图
    plot_low_prob_curve(low_prob_token_probs, sample_time, "vLLM[min_p_sampling]")

    return data2, data3

# 绘制柱状图的函数
def plot_bar_chart(data1, data2, data3, title):
    plt.figure(figsize=(6, 6))
    bar_width = 0.2
    idx = np.arange(len(data1)).astype(float)

    bars1 = plt.bar(idx - bar_width, data1, width=bar_width, color='salmon', label='原始概率', alpha=0.9)
    bars2 = plt.bar(idx, data2, width=bar_width, color='skyblue', label='采样概率', alpha=0.9)
    bars3 = plt.bar(idx + bar_width, data3, width=bar_width, color='orange', label='归一化原始概率', alpha=0.9)

    plt.bar_label(bars1, label_type='edge', padding=3, fmt='%.3f', fontsize=5, color='black')
    plt.bar_label(bars2, label_type='edge', padding=3, fmt='%.3f', fontsize=5, color='red')
    plt.bar_label(bars3, label_type='edge', padding=3, fmt='%.3f', fontsize=5, color='blue')

    plt.title(title, fontsize=14)
    plt.xlabel("索引", fontsize=12)
    plt.ylabel("概率", fontsize=12)
    plt.ylim(0, 1.1)
    plt.xlim(-1, 3)
    plt.xticks(range(0, 3, 1))
    plt.legend(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    output_path = f"{title.replace(' ', '_')}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.clf()

# 绘制低概率 token 概率曲线的函数
def plot_low_prob_curve(low_prob_token_probs, sample_time, title):
    plt.figure(figsize=(6, 6))
    plt.plot(np.arange(0, sample_time), low_prob_token_probs, marker='', linestyle='-', linewidth=1, color='blue')
    plt.xlabel('采样次数')
    plt.ylabel('概率')
    plt.title('低概率 token 的概率')
    plt.grid(alpha=0.3)
    output_path = f"{title.replace(' ', '_')}_low_prob.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.clf()

# 主函数
def main():
    print("运行 FastDeploy 采样...")
    data2_fastdeploy, data3_fastdeploy = fastdeploy_min_p_sampling()
    print("运行 vLLM (FlashInfer) 采样...")
    data2_vllm, data3_vllm = vllm_min_p_sampling()

    # 计算误差
    error_fastdeploy = np.abs(data2_fastdeploy - data3_fastdeploy)
    error_vllm = np.abs(data2_vllm - data3_vllm)

    # 输出对比结果
    print("\nFastDeploy 对比结果：")
    print(f"采样概率 (data2): {data2_fastdeploy}")
    print(f"理论归一化概率 (data3): {data3_fastdeploy}")
    print(f"误差: {error_fastdeploy}")
    print(f"最大误差: {np.max(error_fastdeploy)}, 是否小于 1e-5: {np.max(error_fastdeploy) < 1e-5}")

    print("\nvLLM (FlashInfer) 对比结果：")
    print(f"采样概率 (data2): {data2_vllm}")
    print(f"理论归一化概率 (data3): {data3_vllm}")
    print(f"误差: {error_vllm}")
    print(f"最大误差: {np.max(error_vllm)}, 是否小于 1e-5: {np.max(error_vllm) < 1e-5}")

    # 检查误差是否满足要求
    if np.max(error_fastdeploy) < 1e-5 and np.max(error_vllm) < 1e-5:
        print("\n结论：FastDeploy 和 vLLM 的采样结果均与理论值一致，误差小于 1e-5。")
    else:
        print("\n结论：存在误差大于 1e-5 的情况，建议增加采样次数或检查实现细节。")

# 运行程序
if __name__ == "__main__":
    if paddle.device.is_compiled_with_cuda() and torch.cuda.is_available():
        main()
    else:
        print("GPU 不可用，请检查环境配置（需要支持 PaddlePaddle 和 PyTorch 的 CUDA）。")
