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

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import pytest

pytest.importorskip("paddle")

from fastdeploy import envs
from fastdeploy.engine.request import CompletionOutput, Request, RequestOutput
from fastdeploy.scheduler import global_scheduler
from fastdeploy.scheduler.data import ScheduledRequest, ScheduledResponse
from fastdeploy.scheduler.workers import Task


class _FakeRedis:
    """
    内存中的 Redis 替身，模拟调度器使用的 Redis API。
    用于单元测试，避免依赖真实的 Redis 服务。
    """

    def __init__(self) -> None:
        # 模拟 Redis 的 Key-Value 存储
        self.kv: Dict[str, str] = {}
        # 模拟 Redis 的 List (用于队列)
        self.lists: Dict[str, List[bytes]] = {}
        # 模拟 Redis 的 Sorted Set (用于负载均衡记录)
        self.sorted_sets: Dict[str, Dict[str, float]] = {}
        self.version = "fake-redis"
        # 用于模拟阻塞弹出的返回值存储
        self.blocking_returns: Dict[str, List[bytes]] = {}

    # ---------------------------- helpers used in the tests -----------------
    def queue_blocking_value(self, key: str, value: bytes) -> None:
        """测试辅助函数：预先放入将在 blpop 中返回的数据"""
        self.blocking_returns.setdefault(key, []).append(value)

    # -------------------------------- redis-like operations -----------------
    def set(self, key: str, value: str, ex: Optional[int] = None, nx: bool = False) -> bool:
        if nx and key in self.kv:
            return False
        self.kv[key] = value
        return True

    def delete(self, *keys: str) -> int:
        removed = 0
        for key in keys:
            removed += int(key in self.kv or key in self.lists or key in self.sorted_sets)
            self.kv.pop(key, None)
            self.lists.pop(key, None)
            self.sorted_sets.pop(key, None)
        return removed

    def exists(self, key: str) -> int:
        if key in self.kv or key in self.lists or key in self.sorted_sets:
            return 1
        return 0

    def rpush(self, key: str, *values: bytes, ttl: Optional[int] = None) -> None:
        bucket = self.lists.setdefault(key, [])
        bucket.extend(values)

    def lpush(self, key: str, *values: bytes) -> None:
        bucket = self.lists.setdefault(key, [])
        for value in values:
            bucket.insert(0, value)

    def lpop(self, key: str, count: Optional[int] = None, ttl: Optional[int] = None):
        bucket = self.lists.get(key)
        if not bucket:
            return None
        if count == 0:
            return []
        if count is None or count == 1:
            return [bucket.pop(0)]
        count = min(count, len(bucket))
        result = [bucket.pop(0) for _ in range(count)]
        return result if result else None

    def blpop(self, keys: Iterable[str], timeout: int) -> Optional[Tuple[bytes, bytes]]:
        # 模拟阻塞弹出：先检查普通队列
        for key in keys:
            bucket = self.lists.get(key)
            if bucket:
                return key.encode("utf-8"), bucket.pop(0)
        # 再检查测试预设的阻塞返回队列
        for key in keys:
            bucket = self.blocking_returns.get(key)
            if bucket:
                return key.encode("utf-8"), bucket.pop(0)
        return None

    def zincrby(
        self,
        key: str,
        amount: float,
        member: str,
        rem_amount: Optional[int] = None,
        ttl: Optional[int] = None,
    ) -> None:
        bucket = self.sorted_sets.setdefault(key, {})
        bucket[member] = bucket.get(member, 0) + amount

    def zrangebyscore(
        self,
        key: str,
        min_score: float,
        max_score: float,
        start: int = 0,
        num: Optional[int] = None,
    ) -> List[bytes]:
        """模拟按分数范围查询 Sorted Set，用于获取低负载节点"""
        bucket = self.sorted_sets.get(key, {})
        items = [item for item in bucket.items() if min_score <= item[1] <= max_score]
        # 按 (分数, 成员名) 排序，保证确定性
        items.sort(key=lambda it: (it[1], it[0]))
        members = [member.encode("utf-8") for member, _ in items]
        if num is None or num < 0:
            return members[start:]
        return members[start : start + num]

    def zrem(self, key: str, member: str) -> int:
        bucket = self.sorted_sets.get(key)
        if bucket is None:
            return 0
        return int(bucket.pop(member, None) is not None)


class _ImmediateWorkers:
    """一个同步执行回调的 Worker 池，用于简化测试流程。"""

    def __init__(self, name, work, max_task_batch_size, task_filters=None):
        self.work = work
        self.results: List[Task] = []

    def start(self, workers: int) -> None:  # pragma: no cover - unused in tests
        return None

    def add_tasks(self, tasks: List[Task], unique: bool = False) -> None:
        if unique:
            seen = set()
            unique_tasks: List[Task] = []
            for task in tasks:
                if task.id in seen:
                    continue
                seen.add(task.id)
                unique_tasks.append(task)
            tasks = unique_tasks
        # 同步执行任务并将结果保存
        results = self.work(tasks)
        if results:
            self.results.extend(results)

    def get_results(self, max_size: int, timeout: float) -> List[Task]:
        returned = self.results[:max_size]
        del self.results[:max_size]
        return returned


class _DormantThread:
    """线程桩（Stub），记录启动状态但不执行实际的目标函数。"""

    def __init__(self, target=None, args=None, kwargs=None, daemon=None):
        self.target = target
        self.args = args or ()
        self.kwargs = kwargs or {}
        self.daemon = daemon
        self.started = False

    def start(self) -> None:
        self.started = True

    def join(self, timeout: Optional[float] = None) -> None:  # pragma: no cover - unused
        return None


@dataclass
class _SamplingParamsStub:
    temperature: float = 0.0


def _make_request(request_id: str, token_count: int = 4) -> Request:
    """构造一个测试用的 Request 对象"""
    tokens = list(range(token_count))
    return Request(
        request_id=request_id,
        prompt="hello",
        prompt_token_ids=tokens,
        prompt_token_ids_len=len(tokens),
        messages=None,
        history=None,
        tools=None,
        system=None,
        eos_token_ids=[0],
        sampling_params=_SamplingParamsStub(),
    )


def _make_output(request_id: str, finished: bool = False) -> RequestOutput:
    """构造一个测试用的 RequestOutput 对象"""
    completion = CompletionOutput.from_dict({"index": 0, "send_idx": 0, "token_ids": [1]})
    return RequestOutput(request_id=request_id, outputs=completion, finished=finished)


@pytest.fixture
def scheduler_fixture(monkeypatch):
    """
    初始化 GlobalScheduler 并替换其依赖（Redis, Workers, Thread）为 Mock 对象。
    """
    fake_redis = _FakeRedis()

    # 使用 monkeypatch 替换全局依赖
    monkeypatch.setattr(global_scheduler, "ConnectionPool", lambda **_: object())
    monkeypatch.setattr(global_scheduler, "AdaptedRedis", lambda connection_pool: fake_redis)
    monkeypatch.setattr(global_scheduler, "Workers", _ImmediateWorkers)
    monkeypatch.setattr(global_scheduler.threading, "Thread", _DormantThread)
    monkeypatch.setattr(global_scheduler.utils, "get_hostname_ip", lambda: ("host", "scheduler"))

    scheduler = global_scheduler.GlobalScheduler(
        host="localhost",
        port=0,
        db=0,
        password=None,
        topic="topic",
        ttl=30,
        min_load_score=0,
        load_shards_num=2,
        enable_chunked_prefill=True,
        max_num_partial_prefills=1,
        max_long_partial_prefills=0,
        long_prefill_token_threshold=4,
    )
    return scheduler, fake_redis


def test_put_requests_handles_duplicates_and_load_accounting(scheduler_fixture):
    """测试 put_requests：验证重复请求处理及负载计数是否正确更新。"""
    scheduler, fake_redis = scheduler_fixture

    req = _make_request("req-1")
    duplicate = _make_request("req-1")

    # 尝试放入原始请求和重复请求
    results = scheduler.put_requests([req, duplicate])

    # 预期结果：第一个成功，第二个因 ID 重复失败
    assert results == [("req-1", None), ("req-1", "duplicate request_id")]
    
    # 验证 Redis 队列中只有一个请求
    queue = scheduler._request_queue_name()
    assert len(fake_redis.lists[queue]) == 1

    # 验证负载表 (Sorted Set) 计数增加
    load_table = fake_redis.sorted_sets[scheduler._load_table_name()]
    assert load_table[scheduler.name] == 1


def test_get_requests_can_steal_remote_request(monkeypatch, scheduler_fixture):
    """测试 get_requests：验证当本地空闲时，能从其他节点窃取任务（Work Stealing）。"""
    scheduler, fake_redis = scheduler_fixture
    monkeypatch.setattr(envs, "FD_ENABLE_MAX_PREFILL", 0)

    # Mock 随机函数以确保测试行为确定性（总是选中第一个）
    monkeypatch.setattr(global_scheduler.random, "sample", lambda seq, k: list(seq)[:k])
    monkeypatch.setattr(global_scheduler.random, "choice", lambda seq: list(seq)[0])

    # 构造远程节点的队列和请求
    peer_queue = scheduler._request_queue_name("peer")
    peer_request = ScheduledRequest(_make_request("stolen"), peer_queue, scheduler._response_queue_name("peer"))
    fake_redis.rpush(peer_queue, peer_request.serialize())

    # 设置负载表：本地负载为0，对端负载为2（触发窃取条件）
    fake_redis.sorted_sets[f"{scheduler.topic}.load.0"] = {scheduler.name: 0, "peer": 2}

    requests = scheduler.get_requests(
        available_blocks=10,
        block_size=1,
        reserved_output_blocks=0,
        max_num_batched_tokens=100,
        batch=2,
    )

    # 验证成功窃取到 "stolen" 请求
    assert [req.request_id for req in requests] == ["stolen"]
    # 验证请求被记录在 stolen_requests 中
    assert "stolen" in scheduler.stolen_requests
    # 验证对端负载计数减少
    assert fake_redis.sorted_sets[f"{scheduler.topic}.load.0"]["peer"] == 1


def test_get_requests_requeues_when_chunked_limits_hit(scheduler_fixture):
    """测试 get_requests：当触发分块预填充限制时，长任务应被重新放回队列。"""
    scheduler, fake_redis = scheduler_fixture
    monkeypatch.setattr(envs, "FD_ENABLE_MAX_PREFILL", 0)

    queue = scheduler._request_queue_name()
    short_request = ScheduledRequest(_make_request("short", token_count=2), queue, scheduler._response_queue_name())
    long_request = ScheduledRequest(_make_request("long", token_count=10), queue, scheduler._response_queue_name())
    fake_redis.rpush(queue, short_request.serialize(), long_request.serialize())

    # 长任务阈值为4 (fixture设置)，token=10 的任务会被跳过
    pulled = scheduler.get_requests(
        available_blocks=100,
        block_size=1,
        reserved_output_blocks=0,
        max_num_batched_tokens=100,
        batch=2,
    )

    # 只有短任务被取出
    assert [req.request_id for req in pulled] == ["short"]
    # 长任务应该还在队列中（被重新放入）
    assert len(fake_redis.lists[queue]) == 1
    assert fake_redis.lists[queue][0] == long_request.serialize()


def test_get_requests_returns_empty_when_resources_insufficient(scheduler_fixture):
    """测试 get_requests：当资源不足时（available_blocks=0），应返回空列表。"""
    scheduler, fake_redis = scheduler_fixture

    monkeypatch.setattr(envs, "FD_ENABLE_MAX_PREFILL", 0)

    result = scheduler.get_requests(
        available_blocks=0,
        block_size=1,
        reserved_output_blocks=1,
        max_num_batched_tokens=1,
        batch=1,
    )

    assert result == []
    # 确认没有与 Redis 进行不必要的交互
    assert fake_redis.lists == {}


def test_get_requests_blocking_pop_returns_when_idle(scheduler_fixture):
    """测试 get_requests：模拟空闲状态下的阻塞读取（Blocking Pop）。"""
    scheduler, fake_redis = scheduler_fixture
    monkeypatch.setattr(envs, "FD_ENABLE_MAX_PREFILL", 0)

    queue = scheduler._request_queue_name()
    request = ScheduledRequest(_make_request("blocked"), queue, scheduler._response_queue_name())
    # 放入 fake-redis 的阻塞返回区
    fake_redis.queue_blocking_value(queue, request.serialize())

    pulled = scheduler.get_requests(
        available_blocks=10,
        block_size=1,
        reserved_output_blocks=0,
        max_num_batched_tokens=10,
        batch=1,
    )

    assert [req.request_id for req in pulled] == ["blocked"]


def test_put_results_worker_routes_local_and_stolen_responses(scheduler_fixture):
    """测试结果处理 Worker：区分本地任务结果和窃取任务结果的路由逻辑。"""
    scheduler, fake_redis = scheduler_fixture

    # 预设状态：一个本地任务，一个窃取来的任务
    with scheduler.mutex:
        scheduler.local_responses = {"local": []}
        scheduler.stolen_requests = {
            "stolen": ScheduledRequest(
                _make_request("stolen"),
                scheduler._request_queue_name("peer"),
                scheduler._response_queue_name("peer"),
            )
        }

    local_task = Task("local", _make_output("local"))
    stolen_task = Task("stolen", _make_output("stolen", finished=True))

    scheduler._put_results_worker([local_task, stolen_task])

    # 本地任务结果存入 local_responses
    assert len(scheduler.local_responses["local"]) == 1
    # 窃取任务结果发送回对端队列
    peer_queue = scheduler._response_queue_name("peer")
    assert len(fake_redis.lists[peer_queue]) == 1
    # 窃取任务完成后，从 stolen_requests 中移除
    assert "stolen" not in scheduler.stolen_requests


def test_put_results_worker_keeps_unfinished_stolen_request(monkeypatch, scheduler_fixture):
    """测试结果处理 Worker：对于未完成的窃取任务，应保留在 stolen_requests 中以便后续处理。"""
    scheduler, fake_redis = scheduler_fixture

    with scheduler.mutex:
        scheduler.stolen_requests = {
            "stolen": ScheduledRequest(
                _make_request("stolen"),
                scheduler._request_queue_name("peer"),
                scheduler._response_queue_name("peer"),
            )
        }

    # 任务状态为未完成 finished=False
    unfinished = Task("stolen", _make_output("stolen", finished=False))
    scheduler._put_results_worker([unfinished])

    peer_queue = scheduler._response_queue_name("peer")
    assert len(fake_redis.lists[peer_queue]) == 1
    # 仍在追踪列表中
    assert "stolen" in scheduler.stolen_requests


def test_get_results_returns_batches_and_cleans_up(scheduler_fixture):
    """测试 get_results：批量获取结果并验证读取后是否被清理。"""
    scheduler, _ = scheduler_fixture

    responses = [ScheduledResponse(_make_output("req", finished=(i == 63))) for i in range(64)]
    with scheduler.mutex:
        scheduler.local_responses = {"req": responses}

    result = scheduler.get_results()

    assert "req" in result
    assert len(result["req"]) == 64
    # 读取后应从 local_responses 中移除
    assert "req" not in scheduler.local_responses


def test_reset_and_update_config_refreshes_tables(scheduler_fixture):
    """测试 reset 和 update_config：验证状态清理和配置热更新功能。"""
    scheduler, fake_redis = scheduler_fixture

    queue = scheduler._request_queue_name()
    resp_queue = scheduler._response_queue_name()
    fake_redis.lists[queue] = [b"item"]
    fake_redis.lists[resp_queue] = [b"resp"]
    fake_redis.sorted_sets.setdefault(scheduler._load_table_name(), {scheduler.name: 5})
    scheduler.local_responses = {"req": []}
    scheduler.stolen_requests = {"req": ScheduledRequest(_make_request("req"), queue, resp_queue)}

    # 执行重置
    scheduler.reset()

    # 验证 Redis 数据和本地状态已被清理
    assert queue not in fake_redis.lists
    assert resp_queue not in fake_redis.lists
    assert scheduler.name not in fake_redis.sorted_sets[scheduler._load_table_name()]
    assert scheduler.local_responses == {}
    assert scheduler.stolen_requests == {}

    # 测试配置更新（如分片数量变更）
    scheduler.update_config(load_shards_num=3, reallocate=True)
    assert scheduler.load_shards_num == 3
    assert scheduler.shard == scheduler._get_hash_slot(scheduler.name) % 3


def test_mark_helpers_and_block_calculation(scheduler_fixture):
    """测试辅助函数：Block 计算、请求标记逻辑。"""
    scheduler, _ = scheduler_fixture

    # 测试 Block 数量计算 (ceil division)
    assert global_scheduler.GlobalScheduler.calc_required_blocks(17, 4) == 5

    queue_name = scheduler._request_queue_name("peer")
    scheduler_name = scheduler._scheduler_name_from_request_queue(queue_name)
    assert scheduler_name == "peer"
    assert scheduler._load_table_name(slot=3) == f"{scheduler.topic}.load.{3 % scheduler.load_shards_num}"

    # 测试请求标记（用于区分窃取任务）
    scheduled = ScheduledRequest(_make_request("mark"), queue_name, scheduler._response_queue_name("peer"))
    global_scheduler.GlobalScheduler._mark_request(scheduled)
    assert scheduled.request_id.startswith("mark<")

    # 测试响应去标记
    response = ScheduledResponse(_make_output(scheduled.request_id))
    global_scheduler.GlobalScheduler._unmark_response(response, queue_name)
    assert response.request_id == "mark"
