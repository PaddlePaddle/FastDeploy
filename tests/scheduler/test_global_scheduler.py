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
    In-memory Redis stand-in that simulates the Redis API used by the scheduler.
    Used for unit tests to avoid depending on a real Redis service.
    """

    def __init__(self) -> None:
        # Simulated Redis key-value storage
        self.kv: Dict[str, str] = {}
        # Simulated Redis list (for queues)
        self.lists: Dict[str, List[bytes]] = {}
        # Simulated Redis sorted set (for load balancing records)
        self.sorted_sets: Dict[str, Dict[str, float]] = {}
        self.version = "fake-redis"
        # Storage for simulated blocking-pop return values
        self.blocking_returns: Dict[str, List[bytes]] = {}

    # ---------------------------- helpers used in the tests -----------------
    def queue_blocking_value(self, key: str, value: bytes) -> None:
        """Test helper: pre-enqueue a value that will be returned by blpop"""
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
        # Simulate blocking pop: check normal queue first
        for key in keys:
            bucket = self.lists.get(key)
            if bucket:
                return key.encode("utf-8"), bucket.pop(0)
        # Then check the pre-seeded blocking return queue for tests
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
        """Simulate querying a Sorted Set by score range, used to fetch low-load nodes"""
        bucket = self.sorted_sets.get(key, {})
        items = [item for item in bucket.items() if min_score <= item[1] <= max_score]
        # Sort by (score, member) to ensure determinism
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
    """A worker pool that executes callbacks synchronously to simplify the test flow."""

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
        # Execute tasks synchronously and store results
        results = self.work(tasks)
        if results:
            self.results.extend(results)

    def get_results(self, max_size: int, timeout: float) -> List[Task]:
        returned = self.results[:max_size]
        del self.results[:max_size]
        return returned


class _DormantThread:
    """Thread stub that records start state but does not execute the actual target function."""

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
    """Build a Request object for tests"""
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
    """Build a RequestOutput object for tests"""
    completion = CompletionOutput.from_dict({"index": 0, "send_idx": 0, "token_ids": [1]})
    return RequestOutput(request_id=request_id, outputs=completion, finished=finished)


@pytest.fixture
def scheduler_fixture(monkeypatch):
    """
    Initialize GlobalScheduler and replace its dependencies (Redis, Workers, Thread) with mock objects.
    """
    fake_redis = _FakeRedis()

    # Use monkeypatch to replace global dependencies
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
    """Test put_requests: verify duplicate request handling and that load counters are updated correctly."""
    scheduler, fake_redis = scheduler_fixture

    req = _make_request("req-1")
    duplicate = _make_request("req-1")

    # Try to enqueue the original request and a duplicate request
    results = scheduler.put_requests([req, duplicate])

    # Expected: first succeeds, second fails due to duplicate ID
    assert results == [("req-1", None), ("req-1", "duplicate request_id")]

    # Verify only one request exists in the Redis queue
    queue = scheduler._request_queue_name()
    assert len(fake_redis.lists[queue]) == 1

    # Verify the load table (Sorted Set) counter increases
    load_table = fake_redis.sorted_sets[scheduler._load_table_name()]
    assert load_table[scheduler.name] == 1


def test_get_requests_can_steal_remote_request(monkeypatch, scheduler_fixture):
    """Test get_requests: verify that when local is idle, it can steal tasks from other nodes (work stealing)."""
    scheduler, fake_redis = scheduler_fixture
    monkeypatch.setattr(envs, "FD_ENABLE_MAX_PREFILL", 0)

    # Mock random functions to make the test deterministic (always pick the first)
    monkeypatch.setattr(global_scheduler.random, "sample", lambda seq, k: list(seq)[:k])
    monkeypatch.setattr(global_scheduler.random, "choice", lambda seq: list(seq)[0])

    # Build the remote node's queue and request
    peer_queue = scheduler._request_queue_name("peer")
    peer_request = ScheduledRequest(_make_request("stolen"), peer_queue, scheduler._response_queue_name("peer"))
    fake_redis.rpush(peer_queue, peer_request.serialize())

    # Set load table: local load is 0, peer load is 2 (triggers stealing condition)
    fake_redis.sorted_sets[f"{scheduler.topic}.load.0"] = {scheduler.name: 0, "peer": 2}

    requests = scheduler.get_requests(
        available_blocks=10,
        block_size=1,
        reserved_output_blocks=0,
        max_num_batched_tokens=100,
        batch=2,
    )

    # Verify we successfully stole the "stolen" request
    assert [req.request_id for req in requests] == ["stolen"]
    # Verify the request is recorded in stolen_requests
    assert "stolen" in scheduler.stolen_requests
    # Verify the peer load counter decreases
    assert fake_redis.sorted_sets[f"{scheduler.topic}.load.0"]["peer"] == 1


def test_get_requests_requeues_when_chunked_limits_hit(monkeypatch, scheduler_fixture):
    """Test get_requests: when chunked prefill limits are hit, long requests should be re-queued."""
    scheduler, fake_redis = scheduler_fixture
    monkeypatch.setattr(envs, "FD_ENABLE_MAX_PREFILL", 0)

    queue = scheduler._request_queue_name()
    short_request = ScheduledRequest(_make_request("short", token_count=2), queue, scheduler._response_queue_name())
    long_request = ScheduledRequest(_make_request("long", token_count=10), queue, scheduler._response_queue_name())
    fake_redis.rpush(queue, short_request.serialize(), long_request.serialize())

    # Long-task threshold is 4 (set by fixture); the task with token=10 will be skipped
    pulled = scheduler.get_requests(
        available_blocks=100,
        block_size=1,
        reserved_output_blocks=0,
        max_num_batched_tokens=100,
        batch=2,
    )

    # Only the short task is pulled
    assert [req.request_id for req in pulled] == ["short"]
    # The long task should still be in the queue (re-queued)
    assert len(fake_redis.lists[queue]) == 1
    assert fake_redis.lists[queue][0] == long_request.serialize()


def test_get_requests_returns_empty_when_resources_insufficient(monkeypatch, scheduler_fixture):
    """Test get_requests: when resources are insufficient (available_blocks=0), it should return an empty list."""
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
    # Ensure there was no unnecessary interaction with Redis
    assert fake_redis.lists == {}


def test_get_requests_blocking_pop_returns_when_idle(monkeypatch, scheduler_fixture):
    """Test get_requests: simulate blocking read (blocking pop) when idle."""
    scheduler, fake_redis = scheduler_fixture
    monkeypatch.setattr(envs, "FD_ENABLE_MAX_PREFILL", 0)

    queue = scheduler._request_queue_name()
    request = ScheduledRequest(_make_request("blocked"), queue, scheduler._response_queue_name())
    # Put into fake-redis blocking return buffer
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
    """Test result-processing worker: route local results and stolen results correctly."""
    scheduler, fake_redis = scheduler_fixture

    # Preset state: one local task and one stolen task
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

    # Local task result is stored in local_responses
    assert len(scheduler.local_responses["local"]) == 1
    # Stolen task result is sent back to the peer queue
    peer_queue = scheduler._response_queue_name("peer")
    assert len(fake_redis.lists[peer_queue]) == 1
    # After the stolen task finishes, remove it from stolen_requests
    assert "stolen" not in scheduler.stolen_requests


def test_put_results_worker_keeps_unfinished_stolen_request(monkeypatch, scheduler_fixture):
    """Test result-processing worker: unfinished stolen tasks should remain in stolen_requests for later handling."""
    scheduler, fake_redis = scheduler_fixture

    with scheduler.mutex:
        scheduler.stolen_requests = {
            "stolen": ScheduledRequest(
                _make_request("stolen"),
                scheduler._request_queue_name("peer"),
                scheduler._response_queue_name("peer"),
            )
        }

    # Task is unfinished: finished=False
    unfinished = Task("stolen", _make_output("stolen", finished=False))
    scheduler._put_results_worker([unfinished])

    peer_queue = scheduler._response_queue_name("peer")
    assert len(fake_redis.lists[peer_queue]) == 1
    # Still in the tracking map
    assert "stolen" in scheduler.stolen_requests


def test_get_results_returns_batches_and_cleans_up(scheduler_fixture):
    """Test get_results: fetch results in batches and verify they are cleaned up after reading."""
    scheduler, _ = scheduler_fixture

    responses = [ScheduledResponse(_make_output("req", finished=(i == 63))) for i in range(64)]
    with scheduler.mutex:
        scheduler.local_responses = {"req": responses}

    result = scheduler.get_results()

    assert "req" in result
    assert len(result["req"]) == 64
    # After reading, it should be removed from local_responses
    assert "req" not in scheduler.local_responses


def test_reset_and_update_config_refreshes_tables(scheduler_fixture):
    """Test reset and update_config: verify state cleanup and hot config update."""
    scheduler, fake_redis = scheduler_fixture

    queue = scheduler._request_queue_name()
    resp_queue = scheduler._response_queue_name()
    fake_redis.lists[queue] = [b"item"]
    fake_redis.lists[resp_queue] = [b"resp"]
    fake_redis.sorted_sets.setdefault(scheduler._load_table_name(), {scheduler.name: 5})
    scheduler.local_responses = {"req": []}
    scheduler.stolen_requests = {"req": ScheduledRequest(_make_request("req"), queue, resp_queue)}

    # Perform reset
    scheduler.reset()

    # Verify Redis data and local state have been cleared
    assert queue not in fake_redis.lists
    assert resp_queue not in fake_redis.lists
    assert scheduler.name not in fake_redis.sorted_sets[scheduler._load_table_name()]
    assert scheduler.local_responses == {}
    assert scheduler.stolen_requests == {}

    # Test config update (e.g., shard count change)
    scheduler.update_config(load_shards_num=3, reallocate=True)
    assert scheduler.load_shards_num == 3
    assert scheduler.shard == scheduler._get_hash_slot(scheduler.name) % 3


def test_mark_helpers_and_block_calculation(scheduler_fixture):
    """Test helper functions: block calculation and request marking logic."""
    scheduler, _ = scheduler_fixture

    # Test block count calculation (ceil division)
    assert global_scheduler.GlobalScheduler.calc_required_blocks(17, 4) == 5

    queue_name = scheduler._request_queue_name("peer")
    scheduler_name = scheduler._scheduler_name_from_request_queue(queue_name)
    assert scheduler_name == "peer"
    assert scheduler._load_table_name(slot=3) == f"{scheduler.topic}.load.{3 % scheduler.load_shards_num}"

    # Test request marking (to distinguish stolen tasks)
    scheduled = ScheduledRequest(_make_request("mark"), queue_name, scheduler._response_queue_name("peer"))
    global_scheduler.GlobalScheduler._mark_request(scheduled)
    assert scheduled.request_id.startswith("mark<")

    # Test response unmarking
    response = ScheduledResponse(_make_output(scheduled.request_id))
    global_scheduler.GlobalScheduler._unmark_response(response, queue_name)
    assert response.request_id == "mark"
