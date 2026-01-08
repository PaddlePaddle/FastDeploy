import glob
import os
import re
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet

num_ranks = dist.get_world_size()
print("gaoziyuan test :", num_ranks)
rank_id = dist.get_rank()

strategy = fleet.DistributedStrategy()
strategy.hybrid_configs = {"dp_degree": 1, "mp_degree": num_ranks, "pp_degree": 1}
fleet.init(is_collective=True, strategy=strategy)

ep_group = fleet.get_hybrid_communicate_group().get_model_parallel_group()


def tail_lines(path, n=2000):
    """快速获取最后 n 行"""
    result = subprocess.run(["tail", f"-n{n}", path], stdout=subprocess.PIPE, text=True, check=True)
    return result.stdout.splitlines()


def parse_global_topk_from_log(content):
    """解析日志里的 global_topk_indices"""
    nums = []
    collecting = False
    for line in content:
        if "global_topk_indices" in line:
            collecting = True
            continue
        if collecting:
            arr = re.findall(r"\d+", line)
            nums.extend(map(int, arr))
            if "]" in line:  # 结束
                break
    return np.array(nums, dtype=np.int64)


def plt_histogram(all_expert_topk, prefix_name, save_dir):
    experts = np.arange(1, all_expert_topk.shape[0] + 1)  # Expert ID 1~N
    plt.figure(figsize=(14, 6))
    plt.bar(experts, all_expert_topk, color="skyblue")
    plt.xlabel("Expert ID")
    plt.ylabel("Top-K Selection Count")
    plt.title(f"{prefix_name} Top-K Selection Frequency of {all_expert_topk.shape[0]} MoE Experts")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"moe_topk_counts_{prefix_name}.png"))
    plt.close()


def plt_sorted(all_expert_topk, prefix_name, save_dir):
    sorted_counts = np.sort(all_expert_topk)[::-1]  # 从高到低
    plt.figure(figsize=(12, 6))
    plt.plot(sorted_counts, marker="o")
    plt.xlabel("Expert Rank (by Top-K frequency)")
    plt.ylabel("Selection Count")
    plt.title(f"{prefix_name} Sorted Top-K Selection Frequency of {all_expert_topk.shape[0]} Experts")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"moe_topk_counts_sorted_{prefix_name}.png"))
    plt.close()


def status_analyse(all_expert_topk, prefix_name, save_dir):
    total_count = all_expert_topk.sum()
    mean_val = all_expert_topk.mean()
    var_val = all_expert_topk.var()
    std_val = all_expert_topk.std()
    min_val = all_expert_topk.min()
    max_val = all_expert_topk.max()
    median_val = np.median(all_expert_topk)
    q25, q50, q75, q90 = np.percentile(all_expert_topk, [25, 50, 75, 90])
    num_zeros = np.sum(all_expert_topk == 0)

    save_file = os.path.join(save_dir, f"moe_topk_distribution_{prefix_name}.txt")
    with open(save_file, "w") as f:
        print(f"Total Top-K selections: {total_count}", file=f)
        print(f"Mean: {mean_val:.3f}, Std: {std_val:.3f}, Variance: {var_val:.3f}", file=f)
        print(f"Min: {min_val:.3f}, Max: {max_val:.3f}, Median: {median_val:.3f}", file=f)
        print(f"P25: {q25}, P50: {q50}, P75: {q75}, P90: {q90}", file=f)
        imbalance_ratio = max_val / mean_val if mean_val > 0 else float("inf")
        print(f"imbalance_ratio: {imbalance_ratio:.3f}. num_zeros: {num_zeros}", file=f)


def print_top_n_experts(expert_counts, top_n=10, prefix_name="all_expert", save_dir=None):
    """
    打印并保存前 N 个最热专家的 ID、选择次数、自身占比、累计占比。

    Args:
        expert_counts: np.array, shape [num_experts], 每个专家被选中的总次数
        top_n: int, 显示前 N 个专家
        prefix_name: str, 用于保存文件名前缀
        save_dir: str, 保存路径（可选）
    """
    total = expert_counts.sum()
    if total == 0:
        print("⚠️  Total count is zero, skip analysis.")
        return

    sorted_indices = np.argsort(expert_counts)[::-1]  # 从高到低排序索引
    top_indices = sorted_indices[:top_n]
    top_values = expert_counts[top_indices]

    cumulative_sum = 0
    lines = []
    header = f"{'HotRank':<8} {'ExpertID':<10} {'Count':<12} {'Self %':<10} {'Cumul %':<10}"
    separator = "-" * len(header)

    print(f"\n🔥 Top-{top_n} Experts for '{prefix_name}' (Total selections: {total:,})")
    print(header)
    print(separator)

    for i, (idx, val) in enumerate(zip(top_indices, top_values)):
        self_percent = val / total * 100
        cumulative_sum += val
        cum_percent = cumulative_sum / total * 100
        line = f"{i+1:<8} {idx:<10} {val:<12,} {self_percent:<9.2f}% {cum_percent:<9.2f}%"
        print(line)
        lines.append(line)

    # 可选：保存到文件
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"top_{top_n}_experts_{prefix_name}.txt")
        with open(save_path, "w") as f:
            f.write(f"Top-{top_n} Experts for '{prefix_name}' (Total selections: {total:,})\n")
            f.write(header + "\n")
            f.write(separator + "\n")
            for line in lines:
                f.write(line + "\n")
        print(f"📊 Top-{top_n} experts saved to: {save_path}")


import heapq


def balance_experts_across_ep(expert_loads, num_ep_ranks=8, save_dir=None):
    """
    将专家按负载均衡分配到各个 EP rank 上。

    Args:
        expert_loads: np.array, shape [num_experts], 每个专家的总负载（选择次数）
        num_ep_ranks: int, EP 并行数（即 mp_degree，rank 数量）
        save_dir: str, 保存分配结果的路径（可选）

    Returns:
        expert_to_ep: list[int], 长度=num_experts，expert_to_ep[i] = 该专家应分配到的 EP rank ID
        ep_loads: list[int], 每个 EP rank 的总负载
        ep_assignments: list[list[int]], 每个 EP rank 分配到的专家列表
    """
    num_experts = len(expert_loads)

    # 创建专家索引按负载降序排列
    sorted_expert_indices = np.argsort(expert_loads)[::-1]  # 最热专家在前

    # 初始化最小堆：每个 EP 初始负载为 0
    # 堆元素: (当前负载, ep_rank_id, 专家列表)
    heap = [(0, ep_id, []) for ep_id in range(num_ep_ranks)]
    heapq.heapify(heap)

    # 分配专家
    for expert_id in sorted_expert_indices:
        load = expert_loads[expert_id]
        # 取出当前负载最小的 EP
        min_load, ep_id, expert_list = heapq.heappop(heap)
        # 分配专家到该 EP
        expert_list.append(expert_id)
        new_load = min_load + load
        # 放回堆中
        heapq.heappush(heap, (new_load, ep_id, expert_list))

    # 整理结果
    ep_assignments = [[] for _ in range(num_ep_ranks)]
    ep_loads = [0] * num_ep_ranks
    expert_to_ep = [0] * num_experts  # expert_to_ep[i] = 分配到的 ep_id

    # 从堆中提取最终分配
    while heap:
        total_load, ep_id, experts = heapq.heappop(heap)
        ep_assignments[ep_id] = sorted(experts)  # 排序便于阅读
        ep_loads[ep_id] = total_load
        for eid in experts:
            expert_to_ep[eid] = ep_id

    # 打印分配结果
    print(f"\n⚖️  Expert Load Balancing across {num_ep_ranks} EP Ranks:")
    print("=" * 60)
    for ep_id in range(num_ep_ranks):
        experts = ep_assignments[ep_id]
        load = ep_loads[ep_id]
        avg_load = load / len(experts) if experts else 0
        print(
            f"EP Rank {ep_id:<2} | 负载: {load:>8,} | 专家数: {len(experts):>3} | 平均负载: {avg_load:>8.1f} | 专家列表: {experts}"
        )

    imbalance_ratio = max(ep_loads) / min(ep_loads) if min(ep_loads) > 0 else float("inf")
    print(f"\n📊 负载不均衡度 (Max/Min): {imbalance_ratio:.3f}")
    print(f"✅ 平均每 EP 负载: {np.mean(ep_loads):,.1f}")

    # 保存到文件
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "expert_ep_assignment.txt")
        with open(save_path, "w") as f:
            f.write(f"Expert Load Balancing across {num_ep_ranks} EP Ranks\n")
            f.write("=" * 60 + "\n")
            for ep_id in range(num_ep_ranks):
                experts = ep_assignments[ep_id]
                load = ep_loads[ep_id]
                avg_load = load / len(experts) if experts else 0
                f.write(
                    f"EP Rank {ep_id:<2} | 负载: {load:>8,} | 专家数: {len(experts):>3} | 平均负载: {avg_load:>8.1f} | 专家列表: {experts}\n"
                )
            f.write(f"\n📊 负载不均衡度 (Max/Min): {imbalance_ratio:.3f}\n")
            f.write(f"✅ 平均每 EP 负载: {np.mean(ep_loads):,.1f}\n")
            f.write("\n# expert_id -> ep_rank 映射表:\n")
            for i, ep in enumerate(expert_to_ep):
                f.write(f"expert_{i} -> ep_{ep}\n")
        print(f"💾 专家分配方案已保存至: {save_path}")

    return expert_to_ep, ep_loads, ep_assignments


if __name__ == "__main__":
    log_dir = "/root/paddlejob/workspace/env_run/output/gaoziyuan/moe_analyser/"
    save_dir = "/root/paddlejob/workspace/env_run/output/gaoziyuan/moe_analyser/save_ep0.bak"

    # ========== 配置参数 ==========
    MOE_EXPERTS = 128  # MoE 专家总数
    LAYYERS_NUM = 37  # MoE 层总数
    sample_layer = [1, 5, 10, 30]  # 采样分析的层 ID

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 加载本地文件
    my_files = sorted(glob.glob(os.path.join(log_dir, "moe_expert_rank*")))
    local_vals = []
    for f in my_files:
        arr = paddle.load(f, return_numpy=True)
        local_vals.append(arr)

    # 安全检查：确保数据形状符合预期
    if len(local_vals) > 0:
        assert local_vals[0].shape == (
            LAYYERS_NUM,
            MOE_EXPERTS,
        ), f"Data shape mismatch! Expected: ({LAYYERS_NUM}, {MOE_EXPERTS}), Got: {local_vals[0].shape}"

    # 转成 tensor 并 stack: [num_files_per_rank, LAYYERS_NUM, MOE_EXPERTS]
    local_tensor = paddle.stack([paddle.to_tensor(v) for v in local_vals])
    print("local tesnor")
    print(local_tensor)
    print(local_tensor.shape)
    # all_gather 收集所有 rank 数据
    all_tensor_list = []
    paddle.distributed.all_gather(all_tensor_list, local_tensor, group=ep_group)

    if rank_id == 0:
        # 合并成大 tensor: [num_ranks, num_files_per_rank, LAYYERS_NUM, MOE_EXPERTS]
        all_tensor = paddle.stack(all_tensor_list)
        # 展平 worker 维度: [N, LAYYERS_NUM, MOE_EXPERTS]
        merged = all_tensor.numpy().reshape(-1, LAYYERS_NUM, MOE_EXPERTS)
        print("mrege tesnor")
        print(merged)
        print(merged.shape)
        print(f"共收集到 {merged.shape[0]} 个 workerlog, 每个 topk 长度={merged.shape[1:]}")

        # 按层+专家维度求和: [LAYYERS_NUM, MOE_EXPERTS]
        all_expert_topk = np.sum(merged, axis=0)
        print("all xeprt -otpk")
        print(all_expert_topk)
        print(all_expert_topk.shape)
        # 保存为标准 .npy 格式
        np.save(os.path.join(save_dir, "all_expert_topk.npy"), all_expert_topk)

        # 分析指定层
        for layer_id in sample_layer:
            if layer_id >= LAYYERS_NUM:
                print(f"⚠️  Warning: layer_id {layer_id} >= LAYYERS_NUM {LAYYERS_NUM}, skipping.")
                continue
            data = all_expert_topk[layer_id, :]  # [MOE_EXPERTS]
            plt_histogram(data, f"layer_{layer_id}", save_dir)
            plt_sorted(data, f"layer_{layer_id}", save_dir)
            status_analyse(data, f"layer_{layer_id}", save_dir)

        # ✅ 修复：分析所有专家（跨层聚合）→ 每个专家在所有层被选总次数
        expert_sum_across_layers = np.sum(all_expert_topk, axis=0)  # [128,]

        plt_histogram(expert_sum_across_layers, "all_expert", save_dir)
        plt_sorted(expert_sum_across_layers, "all_expert", save_dir)
        status_analyse(expert_sum_across_layers, "all_expert", save_dir)

        # ✅ 保留：分析每层专家选择总和（可选，用于看哪层路由最活跃）
        layer_sum_across_experts = np.sum(all_expert_topk, axis=1)  # [37,]
        plt_histogram(layer_sum_across_experts, "expert_sum_per_layer", save_dir)
        plt_sorted(layer_sum_across_experts, "expert_sum_per_layer", save_dir)
        status_analyse(layer_sum_across_experts, "expert_sum_per_layer", save_dir)

        print("✅ 分析完成，图表和统计已保存至:", save_dir)

        # 🆕 新增：打印 Top-N 热门专家统计
        TOP_N = 128  # 你可以改成 5, 20, 32 等
        print_top_n_experts(expert_sum_across_layers, top_n=TOP_N, prefix_name="all_expert", save_dir=save_dir)

        # 可选：也对每层做 Top-N 分析
        # for layer_id in sample_layer:
        #     if layer_id >= LAYYERS_NUM:
        #         continue
        #     data = all_expert_topk[layer_id, :]
        #     print_top_n_experts(data, top_n=TOP_N, prefix_name=f"layer_{layer_id}", save_dir=save_dir)

        # ========== 🧩 专家重分配：负载均衡到 EP ==========
        print(f"\n🔄 正在根据专家热度重新分配专家到 {num_ranks} 个 EP 设备以实现负载均衡...")
        expert_to_ep, ep_loads, ep_assignments = balance_experts_across_ep(
            expert_sum_across_layers, num_ep_ranks=num_ranks, save_dir=save_dir
        )
