#!/usr/bin/env python

import argparse
import glob
import json
import os
import sys
import time
import uuid
import multiprocessing as mp
from pathlib import Path
from threading import Thread, Event
from operator import itemgetter

import tiktoken
from tqdm import tqdm
from pymxsml.mxsml_extension import *

import sys
import traceback

import paddle.profiler as profiler
# 全局变量：记录处理失败的文件
error_files = []

# ======================
# 1. 基础工具与全局配置
# ======================

# encoding = tiktoken.get_encoding("cl100k_base")

def get_curr_time():
    return time.perf_counter()

# ======================
# 2. Worker 进程逻辑
# ======================
def worker_loop(worker_idx, gpu_id, config_path, device, shared_task_q, result_q):
    """
    每个 Worker 进程初始化一次 Pipeline，随后抢占式获取任务。
    """
    try:
        # 隔离当前进程可见的 GPU 资源
        os.environ["MACA_VISIBLE_DEVICES"] = str(gpu_id)
        from paddlex import create_pipeline
        
        # 初始化 Pipeline (假定为 FastDeploy 后端)
        pipeline = create_pipeline(config_path, device=device)
        # pipeline = create_pipeline(config_path, device="cpu")
        result_q.put(("READY", worker_idx))


    except Exception as e:
        result_q.put(("ERR", worker_idx, f"Initialization Failed: {repr(e)}"))
        return

    # Phase-stagger workers so their per-task submission waves interleave instead of
    # aligning. Aligned waves overload the VL server in bursts (queue spikes) then leave
    # it idle (GPU troughs); interleaving keeps the server steadily fed. Off by default.
    stagger = float(os.environ.get("FD_WORKER_STAGGER", "0") or "0")
    if stagger > 0:
        time.sleep(worker_idx * stagger)

    while True:
        msg = shared_task_q.get()
        if msg is None:  # 收到 None 代表所有任务完成
            break
        
        tag, job_id, batch_paths = msg
        start_t = get_curr_time()
        
        try:
            # 执行推理：直接传入路径列表
            results = list(pipeline.predict(input=batch_paths))
            print(f"Batch {job_id} processed {len(results)} pages")
            markdown_list = []
            for res in results:
                # 提取 Markdown 文本
                res_dict = res._to_markdown(pretty=False)
                markdown_list.append(res_dict["markdown_texts"])
            
            full_markdown = "\n\n".join(markdown_list)
            end_t = get_curr_time()
            

            # 返回详细结果数据用于主进程统计
            task_info = {
                "id": job_id,
                "start_time": start_t,
                "end_time": end_t,
                "successful": True,
                "processed_pages": len(results),
                # "generated_tokens": len(encoding.encode(full_markdown)),
                "markdown": full_markdown
            }
            result_q.put(("DONE", task_info))
            
        except Exception as e:
            # 方式1：获取完整的栈信息字符串（推荐）
            error_trace = traceback.format_exc()
            result_q.put(("DONE", {
                "id": job_id,
                "start_time": start_t,
                "end_time": get_curr_time(),
                "successful": False,
                "error": str(e),
                "traceback": error_trace
            }))

# ======================
# 3. 监控与任务分发
# ======================
def monitor_device(gpu_ids, gpu_metrics_list, stop_event):
    """
    独立线程：采样所有指定 GPU 的性能指标。
    """
    try:
        mxSmlExInit()
        handles = [mxSmlExDeviceGetHandleByIndex(gid) for gid in gpu_ids]
        while not stop_event.is_set():
            sample = {"ts": time.time(), "per_gpu": []}
            for gid, handle in zip(gpu_ids, handles):
                util = mxSmlExDeviceGetUtilizationRates(handle).gpu
                mem = mxSmlExDeviceGetMemoryInfo(handle).used
                sample["per_gpu"].append({"id": gid, "util": util, "mem": mem})
            gpu_metrics_list.append(sample)
            time.sleep(0.5)
    except Exception as e:
        print(f"Monitor Thread Error: {e}")
    finally:
        mxSmlExShutdown()

def task_producer(all_paths, batch_size, shared_task_q):
    """
    生产者线程：将路径列表切片并塞入任务队列。
    """
    for i in range(0, len(all_paths), batch_size):
        batch = all_paths[i : i + batch_size]
        # 任务元组：(TAG, 唯一ID, 路径列表)
        shared_task_q.put(("RUN", uuid.uuid4().hex, batch))

# ======================
# 4. 主程序 (统计与协调)
# ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-GPU Pipeline Parallel Benchmark")
    parser.add_argument("--input_dirs", type=str, default="/external/ai2/models/squad3/model/dataset/omni1_5/pdfs")
    parser.add_argument("-b", "--batch_size", type=int, default=0,
                        help="Files per task. <=0 means auto (ceil(total/num_workers), one big batch per worker). "
                             "A small value (e.g. 4) creates many more tasks than workers so workers pull "
                             "dynamically, desync their pipeline phases, and feed the VL server a smoother stream.")
    parser.add_argument("-o", "--output_path", type=str, default="benchmark_results.json")
    parser.add_argument("--device", type=str, default="metax_gpu")
    parser.add_argument("--paddlex_config_path", type=str, default="PaddleOCR-VL-1_5_fastdeploy.yaml")
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=[0])
    parser.add_argument("--process_per_gpu", type=int, default=3, help="每个 GPU 启动的并发进程数")
    args = parser.parse_args()

    # 扫描输入路径 (支持单路径及递归匹配)
    all_input_paths = []
    input_dirs = [args.input_dirs] if isinstance(args.input_dirs, str) else args.input_dirs
    for d in input_dirs:
        # 只选取 PDF 文件，避免目录或其它杂质干扰 pipeline
        all_input_paths += glob.glob(os.path.join(d, "*.pdf"))
    all_input_paths.sort()
    print(f"Found {len(all_input_paths)} PDF files: {all_input_paths}")
    if not all_input_paths:
        print(f"Error: No PDF files found in {args.input_dirs}")
        sys.exit(1)

    total_files = len(all_input_paths)
    num_workers = len(args.gpu_ids) * args.process_per_gpu
    # batch_size<=0 => auto (one big batch per worker, original behavior).
    # A small explicit batch_size decouples task granularity from worker count: many more
    # tasks than workers => dynamic pull-based load balancing + desynced pipeline phases =>
    # the VL server receives a smoother, more continuous request stream (less sawtooth).
    if args.batch_size <= 0:
        # 使用 (a + b - 1) // b 这种技巧来实现向上取整
        args.batch_size = (total_files + num_workers - 1) // num_workers
    args.batch_size = max(1, min(args.batch_size, total_files))
    total_batches = (total_files + args.batch_size - 1) // args.batch_size

    # 启动并行环境
    mp_ctx = mp.get_context("spawn")
    # 设置背压保护：限制任务队列长度，防止内存因路径列表积压过多
    total_workers = len(args.gpu_ids) * args.process_per_gpu
    shared_task_q = mp_ctx.Queue(maxsize=num_workers * 2) 
    result_q = mp_ctx.Queue()
    
    workers = []
    print(f"[System] Initializing {total_workers} parallel pipelines on GPU(s) {args.gpu_ids}...")
    
    for i in range(total_workers):
        gid = args.gpu_ids[i % len(args.gpu_ids)]
        p = mp_ctx.Process(
            target=worker_loop,
            args=(i, gid, args.paddlex_config_path, args.device, shared_task_q, result_q)
        )
        p.start()
        workers.append(p)

    # 等待所有 Worker 初始化
    ready_count = 0
    while ready_count < total_workers:
        msg = result_q.get()
        if msg[0] == "READY": 
            ready_count += 1
        elif msg[0] == "ERR":
            print(f"[FATAL] Worker {msg[1]} failed: {msg[2]}")
            for p in workers: p.terminate()
            sys.exit(2)

    # 启动性能监控与生产者
    gpu_metrics_list = []
    stop_monitor = Event()
    monitor_thread = Thread(target=monitor_device, args=(args.gpu_ids, gpu_metrics_list, stop_monitor), daemon=True)
    monitor_thread.start()

    producer = Thread(target=task_producer, args=(all_input_paths, args.batch_size, shared_task_q), daemon=True)
    producer.start()

    # 结果收集与汇总
    task_info_list = []
    start_time = get_curr_time()
    
    print(f"[Processing] {total_files} files | {total_batches} batches | batch_size={args.batch_size}")
    

    with open("generated_markdown.md", "w", encoding="utf-8") as f:
        pbar = tqdm(total=total_batches, desc="Inference Progress")
        batches_done = 0
        while batches_done < total_batches:
            msg = result_q.get()
            if msg[0] == "DONE":
                info = msg[1]
                if info["successful"]:
                    f.write(info["markdown"])
                    f.write("\n\n")
                    task_info_list.append(info)
                else:
                    pbar.write(f"\n[Warning] Task {info['id']} failed: {info.get('error')} | traceback: {info.get('traceback', '无栈信息')}")
                
                batches_done += 1
                pbar.update(1)
        pbar.close()

    end_time = get_curr_time()

    # 清理现场
    stop_monitor.set()
    # 放入哨兵信号，通知 Worker 退出
    for _ in range(total_workers): shared_task_q.put(None)
    for w in workers: w.join(timeout=5)
    monitor_thread.join(timeout=2)

    # ======================
    # 5. Summary & 统计输出
    # ======================
    duration = end_time - start_time
    throughput_file = total_files / duration
    
    latencies = [info["end_time"] - info["start_time"] for info in task_info_list if info["successful"]]
    avg_latency_batch = sum(latencies) / len(latencies) if latencies else 0

    print("\n" + "="*50)
    print(f"{'BENCHMARK SUMMARY':^50}")
    print("="*50)
    print(f"Total Files: {total_files}")
    print(f"Throughput: {throughput_file:.4f} files/sec")
    print(f"Total Execution Time: {duration:.2f} sec")
    print(f"Avg Batch Latency: {avg_latency_batch:.4f} sec")

    successful_batches = len(task_info_list)
    if successful_batches > 0:
        processed_pages = sum(info.get("processed_pages", 0) for info in task_info_list)
        throughput_page = processed_pages / duration
        print(f"Total Processed Pages: {processed_pages}")
        print(f"Page Throughput: {throughput_page:.4f} pages/sec")
        
        generated_tokens = sum(info.get("generated_tokens", 0) for info in task_info_list)
        throughput_token = generated_tokens / duration
        print(f"Total Generated Tokens: {generated_tokens}")
        print(f"Token Throughput: {throughput_token:.1f} tokens/sec")
    
    if gpu_metrics_list:
        # 计算系统级的平均和峰值
        all_utils = [sum(g['util'] for g in sample['per_gpu'])/len(args.gpu_ids) for sample in gpu_metrics_list]
        all_mems = [sum(g['mem'] for g in sample['per_gpu']) for sample in gpu_metrics_list]
        
        print("-" * 50)
        print(f"System GPU Utilization (%): Peak={max(all_utils):.1f}, Avg={sum(all_utils)/len(all_utils):.1f}")
        print(f"System GPU Memory (MB): Peak={max(all_mems)/1024**2:.1f}, Avg={sum(all_mems)/len(gpu_metrics_list)/1024**2:.1f}")
    print("="*50)

    # 导出 JSON 报告
    final_report = {
        "config": vars(args),
        "metrics": {
            "total_files": total_files,
            "throughput_file": throughput_file,
            "avg_latency_batch": avg_latency_batch,
            "processed_pages": processed_pages if successful_batches > 0 else 0,
            "throughput_page": throughput_page if successful_batches > 0 else 0,
            "generated_tokens": generated_tokens if successful_batches > 0 else 0,
            "throughput_token": throughput_token if successful_batches > 0 else 0,
        },
        "gpu_history": gpu_metrics_list
    }
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)
    print(f"\n[Done] Results saved to {args.output_path}")

