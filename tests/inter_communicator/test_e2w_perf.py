"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
"""
import os
import sys
import time
import copy
import paddle
import pickle
import types
import numpy as np
from collections import defaultdict, deque
from multiprocessing import Event, Process
from fastdeploy.inter_communicator.engine_worker_queue import EngineWorkerQueue
from fastdeploy.engine.request import Request
from typing import Any

def save_large_object(obj: Any, save_dir: str, tensor_prefix="tensor"):
    """
    完全安全版本：保存大对象，支持循环引用、深度嵌套。
    - Paddle Tensor 自动保存到磁盘
    - 其他对象用 pickle 保存
    """
    os.makedirs(save_dir, exist_ok=True)
    seen = {}  # id(obj) -> 替换后的对象，避免循环引用
    tensor_counter = 0

    def _traverse(o):
        nonlocal tensor_counter
        obj_id = id(o)
        if obj_id in seen:
            return seen[obj_id]  # 避免循环引用

        if isinstance(o, paddle.Tensor):
            # 保存 Tensor
            tensor_file = os.path.join(save_dir, f"{tensor_prefix}_{tensor_counter}.pdparams")
            paddle.save(o, tensor_file)
            tensor_counter += 1
            placeholder = {"_tensor_file_": tensor_file}
            seen[obj_id] = placeholder
            return placeholder

        elif isinstance(o, dict):
            new_dict = {}
            seen[obj_id] = new_dict
            for k, v in o.items():
                new_dict[k] = _traverse(v)
            return new_dict

        elif isinstance(o, (list, tuple, set)):
            cls = type(o)
            new_collection = []
            seen[obj_id] = new_collection
            for v in o:
                new_collection.append(_traverse(v))
            # 转回原类型
            if cls is tuple:
                new_collection = tuple(new_collection)
            elif cls is set:
                new_collection = set(new_collection)
            seen[obj_id] = new_collection
            return new_collection

        elif hasattr(o, "__dict__") and not isinstance(o, type):
            # 类对象
            seen[obj_id] = o
            for attr_name, attr_value in o.__dict__.items():
                setattr(o, attr_name, _traverse(attr_value))
            return o

        else:
            seen[obj_id] = o
            return o

    # 遍历对象并替换 Tensor
    obj_copy = _traverse(obj)

    # 保存非 Tensor 部分
    metadata_file = os.path.join(save_dir, "metadata.pkl")
    with open(metadata_file, "wb") as f:
        pickle.dump(obj_copy, f)

    print(f"Saved metadata to {metadata_file} and {tensor_counter} Tensor files.")


def load_large_object(save_dir: str):
    """
    从 save_dir 加载对象，将 Tensor 占位符替换回 paddle.Tensor
    """
    metadata_file = os.path.join(save_dir, "metadata.pkl")
    with open(metadata_file, "rb") as f:
        obj = pickle.load(f)

    seen = {}

    def _traverse(o):
        obj_id = id(o)
        if obj_id in seen:
            return seen[obj_id]

        if isinstance(o, dict) and "_tensor_file_" in o:
            paddle.device.set_device("cpu")
            tensor = paddle.load(o["_tensor_file_"])
            seen[obj_id] = tensor
            return tensor
        elif isinstance(o, dict):
            new_dict = {}
            seen[obj_id] = new_dict
            for k, v in o.items():
                new_dict[k] = _traverse(v)
            return new_dict
        elif isinstance(o, (list, tuple, set)):
            cls = type(o)
            new_collection = []
            seen[obj_id] = new_collection
            for v in o:
                new_collection.append(_traverse(v))
            if cls is tuple:
                new_collection = tuple(new_collection)
            elif cls is set:
                new_collection = set(new_collection)
            seen[obj_id] = new_collection
            return new_collection
        elif hasattr(o, "__dict__") and not isinstance(o, type):
            seen[obj_id] = o
            for attr_name, attr_value in o.__dict__.items():
                setattr(o, attr_name, _traverse(attr_value))
            return o
        else:
            seen[obj_id] = o
            return o

    return _traverse(obj)


def get_obj_size(obj):
    """返回对象占用的字节数"""
    try:
        if isinstance(obj, paddle.Tensor):
            return obj.element_size() * obj.numel()
        else:
            return sys.getsizeof(obj)
    except Exception:
        return 0


def calc_obj_size(obj, seen=None):
    """递归计算对象及其所有属性的大小"""
    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    if isinstance(obj, (type, types.ModuleType, types.FunctionType)):
        return 0

    size = sys.getsizeof(obj)

    # 遍历容器或对象属性
    if isinstance(obj, dict):
        size += sum(calc_obj_size(k, seen) + calc_obj_size(v, seen) for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set, frozenset)):
        size += sum(calc_obj_size(i, seen) for i in obj)
    elif isinstance(obj, np.ndarray):
        size += obj.nbytes
    elif hasattr(obj, '__dict__'):
        size += calc_obj_size(vars(obj), seen)
    elif hasattr(obj, '__slots__'):  # 支持 __slots__ 的对象
        size += sum(calc_obj_size(getattr(obj, s), seen) for s in obj.__slots__ if hasattr(obj, s))

    return size


def print_object_size(obj, name="root", max_depth=10):
    """
    遍历对象结构，打印每个成员的内存占用（以字节为单位），并统计总大小。
    支持 dict / list / set / tuple / 自定义类对象 / Paddle Tensor。
    """
    seen = set()
    queue = deque([(obj, name, 0)])  # (对象, 路径, 深度)
    results = []
    stats = defaultdict(lambda: {"count": 0, "size": 0})

    while queue:
        current, path, depth = queue.popleft()
        obj_id = id(current)

        if obj_id in seen:
            results.append((path, "<循环引用>", "", ""))
            continue
        seen.add(obj_id)

        size_bytes = 0
        if isinstance(current, np.ndarray):
            size_bytes = calc_obj_size(current)
        else:
            size_bytes = get_obj_size(current)
        obj_type = type(current).__name__

        # 收集统计
        stats[obj_type]["count"] += 1
        stats[obj_type]["size"] += size_bytes

        results.append((path, f"{size_bytes:,} B", obj_type, f"id={obj_id:x}"))

        # 深度控制
        if depth >= max_depth:
            continue

        # 遍历下层
        if isinstance(current, dict):
            for k, v in current.items():
                queue.append((v, f"{path}.{k}", depth + 1))

        elif isinstance(current, (list, tuple, set)):
            for i, v in enumerate(current):
                queue.append((v, f"{path}[{i}]", depth + 1))

        elif hasattr(current, "__dict__") and not isinstance(current, (type, types.FunctionType)):
            for attr, val in vars(current).items():
                queue.append((val, f"{path}.{attr}", depth + 1))

    # 打印表头
    print(f"{'Path':<60} {'Size':>16}  {'Type':<20} {'Info'}")
    print("-" * 110)
    for path, size, t, info in results:
        print(f"{path:<60} {size:>16}  {t:<20} {info}")

    # 汇总统计
    total_size = sum(v["size"] for v in stats.values())
    print("\n" + "=" * 110)
    print(f"{'Type':<20} {'Count':>10} {'Total Size (Bytes)':>25}")
    print("-" * 110)
    for t, v in sorted(stats.items(), key=lambda x: -x[1]["size"]):
        print(f"{t:<20} {v['count']:>10} {v['size']:>25,}")
    print("-" * 110)
    print(f"{'TOTAL':<20} {sum(v['count'] for v in stats.values()):>10} {total_size:>25,}")
    print("=" * 110)

def mock_tasks(size_kb=None):
    if size_kb:
        return [os.urandom(size_kb * 1024)]
    return load_large_object("/workspace/task_tmp/")

Q_ADDRESS = "0.0.0.0:8002"

def producer_proc(mock_data, test_sizes, ready_event, done_event):
    # engine
    engine_queue = EngineWorkerQueue(
        address=Q_ADDRESS,
        is_server=False,
        num_client=1,
        client_id=0,
        local_data_parallel_size=1,
        local_data_parallel_id=0,
    )
    for size in test_sizes:
        # data = mock_tasks(size_kb)
        # mock_data.multimodal_inputs['images'] = None # paddle.to_tensor(mock_data.multimodal_inputs['images'])
        tasks = [copy.deepcopy(mock_data) for _ in range(size)]
        t1 = time.perf_counter()
        engine_queue.put_tasks((tasks, size))
        t2 = time.perf_counter()
        size_b = 0
        for t in tasks:
            size_b += calc_obj_size(t)
        print("\033[31m[Engine] "
              + f"Pushed:{size_b/1024:>12.2f} KB,  "
              + f"Perf:put_tasks={(t2 - t1) * 1e3:>10.4f} ms, "
              + f"BatchSize:{len(tasks):>5}\033[0m")
        # 通知消费者可以读了
        ready_event.set()
        # 等待消费者读完（done_event 由 consumer 清除）
        done_event.wait()
        done_event.clear()
    print("[Engine] Done.")


def consumer_proc(ready_event, done_event):
    work_queue = EngineWorkerQueue(
        address=Q_ADDRESS,
        is_server=False,
        num_client=1,
        client_id=0,
        local_data_parallel_size=1,
        local_data_parallel_id=0,
    )

    while True:
        # 等待生产者通知
        ready_event.wait()
        ready_event.clear()

        t1 = time.perf_counter()
        num_tasks = work_queue.num_tasks()
        t2 = time.perf_counter()
        tasks, _ = work_queue.get_tasks()
        t3 = time.perf_counter()
        if not tasks:
            continue
        size_b = 0
        task_size = 0
        bash_size = 0
        for batch_tasks in tasks:
            task_size += len(batch_tasks)
            for t in batch_tasks:
                size_b += calc_obj_size(t)
                bash_size += 1
        print("\033[32m[Worker]" 
              + f"Recved:{size_b/1024:>12.2f} KB,"  
              + f"Perf:num_tasks/get_tasks={(t2 - t1) * 1e3:>7.4f} / {(t3 - t2) * 1e3:>10.4f} ms, "
              + f"BatchSize:{bash_size:>3}\033[0m\n")
        # 通知生产者可以继续了
        done_event.set()
    print("[Worker] Done.")


def main():
    print(f"[Main] Starting test, pid:{os.getpid()}")
    engine_worker_queue_server = EngineWorkerQueue(
        address=Q_ADDRESS,
        is_server=True,
        num_client=1,
        local_data_parallel_size=1,
    )
    print(f"[Main] Started engine_worker_queue_server, pid:{engine_worker_queue_server.address}")
    test_sizes = [150, 150, 150, 150, 150, 150] * 3
    # test_sizes = [100]

    # 用 Event 同步两边
    ready_event = Event()
    done_event = Event()

    # 启动消费者进程
    consumer_p = Process(target=consumer_proc, args=(ready_event, done_event), name="Consumer")
    consumer_p.start()
    print(f"[Main] Started consumer process, pid:{consumer_p.pid}")

    # 等待消费者就绪
    time.sleep(1)

    # 启动生产者进程，测试不同大小任务
    mock_datas, _ = mock_tasks()
    mock_data: Request = mock_datas[0]
    producer_p = Process(target=producer_proc, args=(mock_data, test_sizes, ready_event, done_event), name="Producer")
    producer_p.start()
    print(f"[Main] Started producer process, pid:{producer_p.pid}")

    try:
        producer_p.join()
        consumer_p.join()
        print("[Main] All processes terminated.")
    except KeyboardInterrupt:
        print("[Main] Interrupted by user.")

    # 停止测试
    time.sleep(5000)
    print("[Main] Test finished. Terminating processes...")


if __name__ == "__main__":
    main()

    # mock_datas, _ = mock_tasks()
    # mock_data: Request = mock_datas[0]
    # tasks = [copy.deepcopy(mock_data) for _ in range(100)]

    # t0 = time.perf_counter()

    # paddle.to_tensor([1])

    # t1 = time.perf_counter()
    # print(f"Perf: to_tensor={(t1 - t0) * 1e3}ms")
    # images = []
    # for task in tasks:
    #     images.append(task.multimodal_inputs['images'])
    # tensor = paddle.to_tensor(images)
    # for task in tasks:
    #     task.multimodal_inputs['images'] = paddle.to_tensor(task.multimodal_inputs['images'])
    # t2 = time.perf_counter()
    # print(f"Perf: to_tensor={(t2 - t1) * 1e3}ms")

    # mock, bsz = mock_tasks()
    # task = mock[0]
    # t1 = time.perf_counter()
    # images = task.multimodal_inputs['images']
    # task.multimodal_inputs['images'] = paddle.to_tensor(images)
    # t2 = time.perf_counter()
    # images = task.multimodal_inputs['images'].numpy()
    # t3 = time.perf_counter()
    # print(f"Perf: to_tensor={(t2 - t1) * 1e3}ms, to_numpy={(t3 - t2) * 1e3}ms")
    # print_object_size(mock)