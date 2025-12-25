# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import annotations

import argparse
import importlib
import pickle
import random
import sys
import time
import types
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _install_stub_modules() -> None:
    """Install lightweight stand-ins for the external dependencies."""

    if getattr(_install_stub_modules, "_installed", False):
        return

    # --------------------------------------------------------------- Redis stubs
    class _FakePipeline:
        def __init__(self, client: "_FakeRedis") -> None:
            self._client = client
            self._commands: list[tuple[str, tuple[Any, ...]]] = []

        def __enter__(self) -> "_FakePipeline":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
            return None

        def multi(self) -> "_FakePipeline":
            return self

        def lpush(self, key: str, *values: Any) -> "_FakePipeline":
            self._commands.append(("lpush", (key, values)))
            return self

        def expire(self, key: str, ttl: int) -> "_FakePipeline":
            self._commands.append(("expire", (key, ttl)))
            return self

        def execute(self) -> None:
            for name, params in self._commands:
                if name == "lpush":
                    key, values = params
                    self._client.lpush(key, *values)
                elif name == "expire":
                    key, ttl = params
                    self._client.expire(key, ttl)
            self._commands.clear()

    class _FakeRedis:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.storage: dict[str, list[Any]] = {}
            self.hashes: dict[str, dict[Any, Any]] = {}
            self.expirations: dict[str, int] = {}

        # ------------------------------- list operations used by the scheduler
        def lpush(self, key: str, *values: Any) -> None:
            items = list(values)
            if not items:
                return
            bucket = self.storage.setdefault(key, [])
            for value in items:
                bucket.insert(0, value)

        def rpop(self, key: str, count: Optional[int] = None) -> Optional[list[Any]]:
            bucket = self.storage.get(key)
            if not bucket:
                return None
            if count is None:
                return [bucket.pop()]
            count = min(count, len(bucket))
            values = [bucket.pop() for _ in range(count)]
            return values

        def brpop(self, keys: Iterable[str], timeout: int = 0):  # type: ignore[override]
            for key in keys:
                bucket = self.storage.get(key)
                if bucket:
                    return (key, bucket.pop())
            return None

        # ------------------------------------------ hash operations for cluster
        def hset(self, key: str, field: str, value: Any) -> None:
            self.hashes.setdefault(key, {})[field] = value

        def hgetall(self, key: str) -> dict[Any, Any]:
            return {k: v for k, v in self.hashes.get(key, {}).items()}

        def hdel(self, key: str, field: str) -> None:
            if key in self.hashes:
                self.hashes[key].pop(field, None)

        # -------------------------------------------------------------- misc ops
        def expire(self, key: str, ttl: int) -> None:
            self.expirations[key] = ttl

        def pipeline(self) -> _FakePipeline:
            return _FakePipeline(self)

        # Metadata required by InferScheduler.check_redis_version
        def info(self) -> dict[str, str]:
            return {"redis_version": "6.2.0"}

        # Health check used by InferScheduler.start
        def ping(self) -> bool:
            return True

    redis_mod = types.ModuleType("redis")
    redis_mod.Redis = _FakeRedis  # type: ignore[attr-defined]
    sys.modules.setdefault("redis", redis_mod)

    # ------------------------------------------- fastdeploy.engine.request stub
    request_mod = types.ModuleType("fastdeploy.engine.request")

    @dataclass
    class CompletionOutput:
        index: int
        send_idx: int
        token_ids: List[int]
        finished: bool = False

        def to_dict(self) -> Dict[str, Any]:
            return {
                "index": self.index,
                "send_idx": self.send_idx,
                "token_ids": list(self.token_ids),
                "finished": self.finished,
            }

        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> "CompletionOutput":
            return cls(
                index=data.get("index", 0),
                send_idx=data.get("send_idx", 0),
                token_ids=list(data.get("token_ids", [])),
                finished=data.get("finished", False),
            )

    @dataclass
    class RequestMetrics:
        arrival_time: float
        inference_start_time: Optional[float] = None

        def to_dict(self) -> Dict[str, Any]:
            return {
                "arrival_time": self.arrival_time,
                "inference_start_time": self.inference_start_time,
            }

        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> "RequestMetrics":
            return cls(
                arrival_time=data.get("arrival_time", time.time()),
                inference_start_time=data.get("inference_start_time"),
            )

    class Request:
        def __init__(
            self,
            request_id: str,
            prompt: Optional[str] = None,
            prompt_token_ids: Optional[List[int]] = None,
            prompt_token_ids_len: int = 0,
            arrival_time: Optional[float] = None,
            disaggregate_info: Optional[Dict[str, Any]] = None,
        ) -> None:
            self.request_id = request_id
            self.prompt = prompt or ""
            self.prompt_token_ids = prompt_token_ids or []
            self.prompt_token_ids_len = prompt_token_ids_len
            self.arrival_time = arrival_time if arrival_time is not None else time.time()
            self.metrics = RequestMetrics(arrival_time=self.arrival_time)
            self.disaggregate_info = disaggregate_info

        def to_dict(self) -> Dict[str, Any]:
            return {
                "request_id": self.request_id,
                "prompt": self.prompt,
                "prompt_token_ids": list(self.prompt_token_ids),
                "prompt_token_ids_len": self.prompt_token_ids_len,
                "arrival_time": self.arrival_time,
                "metrics": self.metrics.to_dict(),
                "disaggregate_info": self.disaggregate_info,
            }

        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> "Request":
            req = cls(
                request_id=data["request_id"],
                prompt=data.get("prompt"),
                prompt_token_ids=data.get("prompt_token_ids"),
                prompt_token_ids_len=data.get("prompt_token_ids_len", 0),
                arrival_time=data.get("arrival_time", time.time()),
                disaggregate_info=data.get("disaggregate_info"),
            )
            metrics_dict = data.get("metrics")
            if metrics_dict:
                req.metrics = RequestMetrics.from_dict(metrics_dict)
            else:
                req.refresh_metrics()
            return req

        def refresh_metrics(self) -> None:
            self.metrics = RequestMetrics.from_dict({"arrival_time": self.arrival_time})

    class RequestOutput:
        def __init__(
            self,
            request_id: str,
            prompt: str,
            prompt_token_ids: List[int],
            outputs: CompletionOutput,
            metrics: RequestMetrics,
            finished: bool = False,
            error_code: int = 200,
            error_msg: Optional[str] = None,
        ) -> None:
            self.request_id = request_id
            self.prompt = prompt
            self.prompt_token_ids = prompt_token_ids
            self.outputs = outputs
            self.metrics = metrics
            self.finished = finished
            self.error_code = error_code
            self.error_msg = error_msg

        def to_dict(self) -> Dict[str, Any]:
            return {
                "request_id": self.request_id,
                "prompt": self.prompt,
                "prompt_token_ids": list(self.prompt_token_ids),
                "outputs": self.outputs.to_dict(),
                "metrics": self.metrics.to_dict(),
                "finished": self.finished,
                "error_code": self.error_code,
                "error_msg": self.error_msg,
            }

        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> "RequestOutput":
            return cls(
                request_id=data["request_id"],
                prompt=data.get("prompt", ""),
                prompt_token_ids=list(data.get("prompt_token_ids", [])),
                outputs=CompletionOutput.from_dict(data.get("outputs", {})),
                metrics=RequestMetrics.from_dict(data.get("metrics", {})),
                finished=data.get("finished", False),
                error_code=data.get("error_code", 200),
                error_msg=data.get("error_msg"),
            )

    request_mod.CompletionOutput = CompletionOutput  # type: ignore[attr-defined]
    request_mod.RequestMetrics = RequestMetrics  # type: ignore[attr-defined]
    request_mod.Request = Request  # type: ignore[attr-defined]
    request_mod.RequestOutput = RequestOutput  # type: ignore[attr-defined]
    sys.modules["fastdeploy.engine.request"] = request_mod

    fd_pkg = types.ModuleType("fastdeploy")
    fd_pkg.__path__ = [str(PROJECT_ROOT / "fastdeploy")]
    sys.modules["fastdeploy"] = fd_pkg

    scheduler_pkg = types.ModuleType("fastdeploy.scheduler")
    scheduler_pkg.__path__ = [str(PROJECT_ROOT / "fastdeploy" / "scheduler")]
    sys.modules["fastdeploy.scheduler"] = scheduler_pkg

    logger_mod = types.ModuleType("fastdeploy.utils.scheduler_logger")

    def _log(*_args: Any, **_kwargs: Any) -> None:
        return None

    for level in ("info", "error", "debug", "warning"):
        setattr(logger_mod, level, _log)  # type: ignore[attr-defined]
    sys.modules["fastdeploy.utils.scheduler_logger"] = logger_mod

    utils_mod = types.ModuleType("fastdeploy.utils")
    utils_mod.scheduler_logger = logger_mod  # type: ignore[attr-defined]
    sys.modules["fastdeploy.utils"] = utils_mod

    _install_stub_modules._installed = True


def _import_splitwise_scheduler():
    """Import the scheduler module with the stub environment."""

    _install_stub_modules()
    return importlib.import_module("fastdeploy.scheduler.splitwise_scheduler")


class _PatchedThread:
    def __init__(self, *args: Any, target=None, **kwargs: Any) -> None:  # type: ignore[override]
        self._target = target
        self.started = False

    def start(self) -> None:
        self.started = True


class _Writer:
    def __init__(self) -> None:
        self.items: list[tuple[str, list[bytes]]] = []

    def put(self, key: str, items: list[bytes]) -> None:
        self.items.append((key, items))


class SplitWiseSchedulerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.module = _import_splitwise_scheduler()
        self._orig_thread = self.module.threading.Thread
        self.module.threading.Thread = _PatchedThread  # type: ignore[assignment]

    def tearDown(self) -> None:
        self.module.threading.Thread = self._orig_thread  # type: ignore[assignment]


class SplitWiseSchedulerConfigTest(SplitWiseSchedulerTestCase):
    def test_threshold_defaults_to_model_ratio(self) -> None:
        config = self.module.SplitWiseSchedulerConfig(
            enable_chunked_prefill=True,
            max_num_partial_prefills=5,
            max_long_partial_prefills=3,
            max_model_len=1000,
        )
        self.assertEqual(config.long_prefill_token_threshold, 40)
        self.assertEqual(config.expire_period, 3.0)

    def test_check_and_print_cover_logging(self) -> None:
        config = self.module.SplitWiseSchedulerConfig(
            enable_chunked_prefill=True,
            max_num_partial_prefills=1,
            max_long_partial_prefills=1,
            max_model_len=50,
        )
        config.check()
        config.print()


class NodeInfoTest(SplitWiseSchedulerTestCase):
    def test_serialization_and_expiration(self) -> None:
        node = self.module.NodeInfo(
            nodeid="node-1",
            role="prefill",
            host="localhost",
            disaggregated={"transfer_protocol": ["ipc", "rdma"]},
            load=2,
        )

        payload = node.serialize()
        loaded = self.module.NodeInfo.load_from("node-1", payload)
        self.assertFalse(loaded.expired(10))

        loaded.ts -= 20
        self.assertTrue(loaded.expired(1))

        loaded.add_req("req-1", 4)
        self.assertIn("req-1", loaded.reqs)

        loaded.update_req_timestamp(["req-1"])
        before = loaded.reqs["req-1"][1]
        loaded.reqs["req-1"][1] -= 1000
        loaded.expire_reqs(ttl=1)
        self.assertNotIn("req-1", loaded.reqs)

        loaded.add_req("req-2", 2)
        loaded.finish_req("req-2")
        self.assertNotIn("req-2", loaded.reqs)
        self.assertNotEqual(before, loaded.ts)

    def test_comparisons(self) -> None:
        low = self.module.NodeInfo("a", "prefill", "h", {"transfer_protocol": ["ipc"]}, load=1)
        high = self.module.NodeInfo("b", "prefill", "h", {"transfer_protocol": ["ipc"]}, load=5)
        self.assertTrue(low < high)
        self.assertIn("a(1)", repr(low))


class ResultReaderTest(SplitWiseSchedulerTestCase):
    def test_read_handles_group_tokens_with_buffer_and_outputs(self) -> None:
        client = sys.modules["redis"].Redis()
        reader = self.module.ResultReader(client, idx=0, batch=10, ttl=30, group="grp")

        req = self.module.Request("req-buffer", prompt_token_ids_len=2)
        reader.add_req(req)

        metrics = self.module.RequestMetrics(arrival_time=time.time())
        head = self.module.RequestOutput(
            request_id=req.request_id,
            prompt="",
            prompt_token_ids=[],
            outputs=self.module.CompletionOutput(index=0, send_idx=0, token_ids=[1]),
            metrics=metrics,
            finished=False,
        )
        buffered = self.module.RequestOutput(
            request_id=req.request_id,
            prompt="",
            prompt_token_ids=[],
            outputs=self.module.CompletionOutput(index=0, send_idx=3, token_ids=[5]),
            metrics=metrics,
            finished=False,
        )
        trailing = self.module.RequestOutput(
            request_id=req.request_id,
            prompt="",
            prompt_token_ids=[],
            outputs=self.module.CompletionOutput(index=0, send_idx=4, token_ids=[6]),
            metrics=metrics,
            finished=True,
        )

        with reader.lock:
            reader.out_buffer[req.request_id] = [buffered]

        reader.data.appendleft(head)
        reader.data.appendleft(trailing)
        outputs = reader.read()
        self.assertIn(req.request_id, outputs)

        # Triggers the path where group_tokens has no pre-existing output bucket
        # so the branch at lines 353-354 is exercised.
        another = self.module.RequestOutput(
            request_id="req-new",
            prompt="",
            prompt_token_ids=[],
            outputs=self.module.CompletionOutput(index=0, send_idx=1, token_ids=[9]),
            metrics=metrics,
            finished=True,
        )
        reader.data.appendleft(another)
        outputs = reader.read()
        self.assertEqual(outputs["req-new"][0].outputs.token_ids, [9])

    def test_read_groups_partial_outputs(self) -> None:
        client = sys.modules["redis"].Redis()
        reader = self.module.ResultReader(client, idx=0, batch=10, ttl=30, group="group-a")

        req = self.module.Request("req-A", prompt_token_ids_len=3)
        reader.add_req(req)

        metrics = self.module.RequestMetrics(arrival_time=time.time())
        first = self.module.RequestOutput(
            request_id="req-A",
            prompt="",
            prompt_token_ids=[],
            outputs=self.module.CompletionOutput(index=0, send_idx=0, token_ids=[1, 2]),
            metrics=metrics,
            finished=False,
        )
        follow = self.module.RequestOutput(
            request_id="req-A",
            prompt="",
            prompt_token_ids=[],
            outputs=self.module.CompletionOutput(index=0, send_idx=1, token_ids=[3]),
            metrics=metrics,
            finished=True,
        )

        reader.data.appendleft(follow)
        reader.data.appendleft(first)

        outputs = reader.read()
        self.assertIn("req-A", outputs)
        self.assertEqual(len(outputs["req-A"]), 2)

    def test_sync_results_converts_payloads(self) -> None:
        client = sys.modules["redis"].Redis()
        reader = self.module.ResultReader(client, idx=0, batch=10, ttl=30, group="")

        metrics = self.module.RequestMetrics(arrival_time=time.time())
        ro = self.module.RequestOutput(
            request_id="req-B",
            prompt="p",
            prompt_token_ids=[1],
            outputs=self.module.CompletionOutput(index=0, send_idx=0, token_ids=[4]),
            metrics=metrics,
            finished=True,
        )

        payload = self.module.orjson.dumps(ro.to_dict())
        client.storage.setdefault("req-key", []).append(payload)

        total = reader.sync_results(["req-key"])
        self.assertEqual(total, 1)
        self.assertTrue(reader.data)

    def test_read_uses_out_buffer(self) -> None:
        client = sys.modules["redis"].Redis()
        reader = self.module.ResultReader(client, idx=0, batch=10, ttl=30, group="grp")

        req = self.module.Request("req-out", prompt_token_ids_len=2)
        reader.add_req(req)

        metrics = self.module.RequestMetrics(arrival_time=time.time())
        head = self.module.RequestOutput(
            request_id="req-out",
            prompt="",
            prompt_token_ids=[],
            outputs=self.module.CompletionOutput(index=0, send_idx=0, token_ids=[1]),
            metrics=metrics,
            finished=False,
        )
        tail = self.module.RequestOutput(
            request_id="req-out",
            prompt="",
            prompt_token_ids=[],
            outputs=self.module.CompletionOutput(index=0, send_idx=2, token_ids=[2, 3]),
            metrics=metrics,
            finished=True,
        )

        with reader.lock:
            reader.out_buffer[req.request_id] = [tail]
        reader.data.appendleft(head)

        outputs = reader.read()
        self.assertEqual(len(outputs["req-out"]), 2)

    def test_sync_results_with_group_override(self) -> None:
        client = sys.modules["redis"].Redis()
        reader = self.module.ResultReader(client, idx=0, batch=10, ttl=30, group="grp")

        metrics = self.module.RequestMetrics(arrival_time=time.time())
        ro = self.module.RequestOutput(
            request_id="req-group",
            prompt="",
            prompt_token_ids=[],
            outputs=self.module.CompletionOutput(index=0, send_idx=0, token_ids=[7]),
            metrics=metrics,
            finished=True,
        )
        payload = self.module.orjson.dumps(ro.to_dict())
        client.storage.setdefault("grp", []).append(payload)

        total = reader.sync_results(["unused"])
        self.assertEqual(total, 1)
        self.assertEqual(reader.data[-1].request_id, "req-group")

    def test_run_emits_expired_placeholder(self) -> None:
        client = sys.modules["redis"].Redis()
        reader = self.module.ResultReader(client, idx=0, batch=10, ttl=1, group="")
        reader.reqs["old"] = {"arrival_time": time.time() - 5}
        reader.reqs["active"] = {"arrival_time": time.time()}

        call_count = {"rpop": 0}

        def _rpop(key: str, batch: int):
            call_count["rpop"] += 1
            if call_count["rpop"] > 1:
                raise SystemExit()
            return []

        reader.client.rpop = _rpop  # type: ignore[assignment]

        with self.assertRaises(SystemExit):
            reader.run()

        self.assertNotIn("old", reader.reqs)
        self.assertTrue(reader.data)
        self.assertGreaterEqual(call_count["rpop"], 1)

    def test_run_handles_empty_keys_and_exceptions(self) -> None:
        client = sys.modules["redis"].Redis()
        reader = self.module.ResultReader(client, idx=0, batch=5, ttl=10, group="")
        reader.reqs.clear()

        original_sleep = self.module.time.sleep
        try:
            self.module.time.sleep = lambda _t: (_ for _ in ()).throw(SystemExit())
            with self.assertRaises(SystemExit):
                reader.run()
        finally:
            self.module.time.sleep = original_sleep

        # Now cover the exception logging path inside run()
        reader.reqs["rid"] = {"arrival_time": time.time()}
        calls = {"count": 0}

        def _rpop(_key: str, _batch: int):
            calls["count"] += 1
            if calls["count"] == 1:
                raise ValueError("boom")
            raise SystemExit()

        reader.client.rpop = _rpop  # type: ignore[assignment]
        with self.assertRaises(SystemExit):
            reader.run()

        self.assertGreaterEqual(calls["count"], 2)


class APISchedulerTest(SplitWiseSchedulerTestCase):
    def _make_config(self) -> Any:
        return self.module.SplitWiseSchedulerConfig(
            enable_chunked_prefill=True,
            max_num_partial_prefills=5,
            max_long_partial_prefills=3,
            max_model_len=200,
        )

    def test_schedule_mixed_node_uses_single_queue(self) -> None:
        config = self._make_config()
        scheduler = self.module.APIScheduler(config)

        req = self.module.Request("req-1", prompt_token_ids_len=10)
        mixed = self.module.NodeInfo("mixed", "mixed", "host-a", {"transfer_protocol": ["ipc"]}, load=1)
        scheduler.select_pd = lambda *args, **kwargs: mixed  # type: ignore[assignment]

        scheduler.schedule(req, [mixed], [], [], group="g0")
        key = f"ReqQ_{mixed.nodeid}"
        self.assertIn(key, scheduler.client.storage)
        stored = scheduler.client.storage[key][0]
        decoded = pickle.loads(stored)
        self.assertEqual(decoded["group"], "g0")
        self.assertIsNone(decoded["disaggregate_info"])

    def test_schedule_disaggregated_nodes_fill_metadata(self) -> None:
        config = self._make_config()
        scheduler = self.module.APIScheduler(config)

        req = self.module.Request("req-meta", prompt_token_ids_len=10)
        disagg = {
            "host_ip": "1.1.1.1",
            "transfer_protocol": ["ipc"],
            "connector_port": 10,
            "device_ids": [0],
            "rdma_ports": [100],
            "tp_size": 2,
        }
        pre = self.module.NodeInfo("p", "prefill", "host-a", disagg, load=1)
        dec = self.module.NodeInfo(
            "d",
            "decode",
            "host-b",
            {
                "host_ip": "1.1.1.1",
                "transfer_protocol": ["ipc"],
                "connector_port": 11,
                "device_ids": [1],
                "rdma_ports": [101],
                "tp_size": 2,
            },
            load=2,
        )

        scheduler.schedule(req, [pre], [dec], [], group="g1")
        self.assertIsNotNone(req.disaggregate_info)
        self.assertEqual(req.disaggregate_info["transfer_protocol"], "ipc")
        self.assertIn(f"ReqQ_{pre.nodeid}", scheduler.client.storage)
        self.assertIn(f"ReqQ_{dec.nodeid}", scheduler.client.storage)

    def test_loop_schedule_consumes_queue_and_uses_reader(self) -> None:
        config = self._make_config()
        scheduler = self.module.APIScheduler(config)
        scheduler.reqs_queue.append(self.module.Request("req-loop", prompt_token_ids_len=1))
        scheduler.readers = [types.SimpleNamespace(add_req=lambda _req: None, group="grp")]
        scheduler.sync_cluster = lambda: ([types.SimpleNamespace(load=0, role="prefill", disaggregated={})], [types.SimpleNamespace(load=0, role="decode", disaggregated={})], [])  # type: ignore[assignment]
        scheduler.schedule = lambda *_args, **_kwargs: (_ for _ in ()).throw(SystemExit())  # type: ignore[assignment]

        with self.assertRaises(SystemExit):
            scheduler.loop_schedule()

    def test_sync_cluster_partitions_and_filters_nodes(self) -> None:
        config = self._make_config()
        scheduler = self.module.APIScheduler(config)
        now = time.time()
        cluster_key = scheduler.cluster_key

        expired_payload = self.module.orjson.dumps(
            {"ts": now - 10, "role": "prefill", "load": 1, "host": "h", "disaggregated": {}}
        )
        scheduler.client.hset(cluster_key, b"expired", expired_payload)

        valid_prefill = self.module.NodeInfo("p1", "prefill", "h1", {"transfer_protocol": ["ipc"]}, load=1)
        scheduler.client.hset(cluster_key, b"p1", valid_prefill.serialize())

        valid_decode = self.module.NodeInfo("d1", "decode", "h2", {"transfer_protocol": ["ipc"]}, load=2)
        scheduler.client.hset(cluster_key, b"d1", valid_decode.serialize())

        invalid_payload = self.module.orjson.dumps(
            {"ts": now, "role": "unknown", "load": 0, "host": "h3", "disaggregated": {}}
        )
        scheduler.client.hset(cluster_key, b"bad", invalid_payload)

        pnodes, dnodes, mnodes = scheduler.sync_cluster()
        self.assertEqual(len(pnodes), 1)
        self.assertEqual(len(dnodes), 1)
        self.assertEqual(len(mnodes), 0)

    def test_loop_clear_expired_nodes_removes_entries(self) -> None:
        config = self._make_config()
        scheduler = self.module.APIScheduler(config)
        cluster_key = scheduler.cluster_key
        stale_payload = self.module.orjson.dumps(
            {
                "ts": time.time() - (scheduler.clear_expired_nodes_period + 5),
                "role": "prefill",
                "load": 0,
                "host": "h",
                "disaggregated": {"transfer_protocol": ["ipc"]},
            }
        )
        scheduler.client.hset(cluster_key, b"old", stale_payload)

        original_sleep = self.module.time.sleep
        self.module.time.sleep = lambda _t: (_ for _ in ()).throw(SystemExit())
        with self.assertRaises(SystemExit):
            scheduler.loop_clear_expired_nodes()
        self.module.time.sleep = original_sleep
        self.assertNotIn(b"old", scheduler.client.hgetall(cluster_key))

    def test_select_pd_paths(self) -> None:
        config = self._make_config()
        scheduler = self.module.APIScheduler(config)
        req = self.module.Request("req-sel", prompt_token_ids_len=50)
        nodes = [
            self.module.NodeInfo(str(i), "prefill", "h", {"transfer_protocol": ["ipc"]}, load=i) for i in range(3)
        ]
        random.seed(0)
        chosen = scheduler.select_pd(req, nodes, "prefill")
        self.assertIn(chosen, nodes)

        decode_nodes = [
            self.module.NodeInfo(str(i), "decode", "h", {"transfer_protocol": ["ipc"]}, load=i) for i in range(2)
        ]
        chosen_decode = scheduler.select_pd(req, decode_nodes, "decode")
        self.assertIn(chosen_decode, decode_nodes)

    def test_schedule_disaggregated_updates_protocol(self) -> None:
        config = self._make_config()
        scheduler = self.module.APIScheduler(config)

        req = self.module.Request("req-2", prompt_token_ids_len=10)
        prefill = self.module.NodeInfo(
            "prefill",
            "prefill",
            "host-a",
            {
                "transfer_protocol": ["ipc"],
                "host_ip": "1.1.1.1",
                "connector_port": 10,
                "device_ids": [0],
                "rdma_ports": [1],
                "tp_size": 1,
            },
            load=1,
        )
        decode = self.module.NodeInfo(
            "decode",
            "decode",
            "host-b",
            {
                "transfer_protocol": ["ipc", "rdma"],
                "host_ip": "2.2.2.2",
                "connector_port": 11,
                "device_ids": [1],
                "rdma_ports": [2],
                "tp_size": 1,
            },
            load=1,
        )

        def _select(req_obj, nodes, role):
            return nodes[0]

        scheduler.select_pd = _select  # type: ignore[assignment]

        scheduler.schedule(req, [prefill], [decode], [], group="")
        self.assertIn("ReqQ_prefill", scheduler.client.storage)
        self.assertIn("ReqQ_decode", scheduler.client.storage)

        decoded = pickle.loads(scheduler.client.storage["ReqQ_prefill"][0])
        self.assertEqual(decoded["disaggregate_info"]["transfer_protocol"], "rdma")

    def test_sync_cluster_filters_expired_nodes(self) -> None:
        config = self._make_config()
        scheduler = self.module.APIScheduler(config)

        fresh = self.module.NodeInfo("n1", "prefill", "h", {"transfer_protocol": ["ipc"]}, load=1)
        scheduler.client.hset(scheduler.cluster_key, fresh.nodeid.encode(), fresh.serialize())

        stale_payload = self.module.orjson.dumps(
            {
                "ts": time.time() - (config.expire_period + 1),
                "role": "prefill",
                "load": 1,
                "host": "h",
                "disaggregated": {"transfer_protocol": ["ipc"]},
            }
        )
        scheduler.client.hset(scheduler.cluster_key, b"n2", stale_payload)

        pnodes, _, _ = scheduler.sync_cluster()
        self.assertEqual([node.nodeid for node in pnodes], ["n1"])

    def test_start_put_and_get_results(self) -> None:
        config = self._make_config()
        scheduler = self.module.APIScheduler(config)
        scheduler.start()

        reqs = [self.module.Request(f"req-{i}", prompt_token_ids_len=1) for i in range(2)]
        result = scheduler.put_requests(reqs)
        self.assertEqual(len(result), 2)

        fake_output = {"a": ["value"]}
        scheduler.readers = [types.SimpleNamespace(read=lambda: fake_output)]
        outputs = scheduler.get_results()
        self.assertEqual(outputs, fake_output)

    def test_select_pd_prefill_and_decode(self) -> None:
        config = self._make_config()
        scheduler = self.module.APIScheduler(config)

        req = self.module.Request("req-select", prompt_token_ids_len=50)
        prefill_nodes = [
            self.module.NodeInfo("a", "prefill", "h", {"transfer_protocol": ["ipc"]}, load=5),
            self.module.NodeInfo("b", "prefill", "h", {"transfer_protocol": ["ipc"]}, load=20),
        ]
        decode_nodes = [
            self.module.NodeInfo("c", "decode", "h", {"transfer_protocol": ["ipc"]}, load=1),
            self.module.NodeInfo("d", "decode", "h", {"transfer_protocol": ["ipc"]}, load=2),
        ]

        original_choice = self.module.random.choice
        self.module.random.choice = lambda seq: seq[-1]  # type: ignore[assignment]
        try:
            picked_prefill = scheduler.select_pd(req, prefill_nodes, "prefill")
            picked_decode = scheduler.select_pd(req, decode_nodes, "decode")
        finally:
            self.module.random.choice = original_choice

        self.assertEqual(picked_prefill.nodeid, "b")
        self.assertEqual(picked_decode.nodeid, "d")

        with self.assertRaises(Exception):
            scheduler.select_pd(req, prefill_nodes, "unknown")


class InferSchedulerTest(SplitWiseSchedulerTestCase):
    def _make_config(self, **overrides: Any) -> Any:
        base = dict(
            enable_chunked_prefill=True,
            max_num_partial_prefills=3,
            max_long_partial_prefills=1,
            max_model_len=200,
        )
        base.update(overrides)
        return self.module.SplitWiseSchedulerConfig(**base)

    def test_get_requests_limits_partial_prefills(self) -> None:
        config = self._make_config(long_prefill_token_threshold=5)
        infer = self.module.InferScheduler(config)
        infer.role = "prefill"
        infer.node = self.module.NodeInfo("n", "prefill", "h", {"transfer_protocol": ["ipc"]}, load=0)

        long = self.module.Request("req-long", prompt_token_ids_len=10)
        longer = self.module.Request("req-longer", prompt_token_ids_len=12)
        infer.reqs_queue.extend([longer, long])

        picked = infer.get_requests(
            available_blocks=100,
            block_size=4,
            reserved_output_blocks=1,
            max_num_batched_tokens=100,
            batch=5,
        )
        self.assertEqual([req.request_id for req in picked], ["req-longer"])
        self.assertEqual([req.request_id for req in infer.reqs_queue], ["req-long"])

    def test_get_requests_non_chunked_uses_token_cap(self) -> None:
        config = self._make_config(enable_chunked_prefill=False)
        infer = self.module.InferScheduler(config)
        infer.role = "prefill"
        infer.node = self.module.NodeInfo("n", "prefill", "h", {"transfer_protocol": ["ipc"]}, load=0)

        infer.reqs_queue.extend(
            [
                self.module.Request("req-1", prompt_token_ids_len=10),
                self.module.Request("req-2", prompt_token_ids_len=20),
            ]
        )

        picked = infer.get_requests(
            available_blocks=100,
            block_size=4,
            reserved_output_blocks=1,
            max_num_batched_tokens=15,
            batch=5,
        )
        self.assertEqual([req.request_id for req in picked], ["req-1"])
        self.assertEqual(len(infer.reqs_queue), 1)

    def test_put_results_groups_by_writer_index(self) -> None:
        config = self._make_config()
        infer = self.module.InferScheduler(config)
        infer.role = "prefill"
        infer.node = self.module.NodeInfo("n", "prefill", "h", {"transfer_protocol": ["ipc"]}, load=0)

        infer.writers = [_Writer(), _Writer()]
        infer.node.add_req("req#0#g", 1)

        metrics = self.module.RequestMetrics(arrival_time=time.time())
        result = self.module.RequestOutput(
            request_id="req#0#g",
            prompt="",
            prompt_token_ids=[],
            outputs=self.module.CompletionOutput(index=0, send_idx=0, token_ids=[1]),
            metrics=metrics,
            finished=True,
        )

        infer.put_results([result])
        self.assertEqual(len(infer.writers[0].items), 1)
        key, payloads = infer.writers[0].items[0]
        self.assertEqual(key, "g")
        decoded = self.module.orjson.loads(payloads[0])
        self.assertFalse(decoded["finished"])

    def test_put_results_handles_errors(self) -> None:
        config = self._make_config()
        infer = self.module.InferScheduler(config)
        infer.role = "decode"
        infer.node = self.module.NodeInfo("n", "decode", "h", {"transfer_protocol": ["ipc"]}, load=0)

        infer.writers = [_Writer()]
        infer.node.add_req("bad#0#", 1)

        metrics = self.module.RequestMetrics(arrival_time=time.time())
        result = self.module.RequestOutput(
            request_id="bad#0#",
            prompt="",
            prompt_token_ids=[],
            outputs=self.module.CompletionOutput(index=0, send_idx=1, token_ids=[1]),
            metrics=metrics,
            finished=True,
            error_code=500,
        )

        infer.put_results([result])
        self.assertFalse(infer.node.reqs)

    def test_start_initializes_writers(self) -> None:
        config = self._make_config()
        infer = self.module.InferScheduler(config)
        infer.start("prefill", "host", {"transfer_protocol": ["ipc"]})
        self.assertEqual(len(infer.writers), config.writer_parallel)

    def test_get_requests_skips_expired_entries(self) -> None:
        config = self._make_config()
        infer = self.module.InferScheduler(config)
        infer.role = "prefill"
        infer.node = self.module.NodeInfo("n", "prefill", "h", {"transfer_protocol": ["ipc"]}, load=0)

        expired = self.module.Request("expired", prompt_token_ids_len=1, arrival_time=time.time() - (infer.ttl + 1))
        infer.node.add_req("expired", 1)
        infer.reqs_queue.append(expired)

        picked = infer.get_requests(
            available_blocks=10,
            block_size=1,
            reserved_output_blocks=1,
            max_num_batched_tokens=10,
            batch=1,
        )

        self.assertEqual(picked, [])
        self.assertNotIn("expired", infer.node.reqs)

    def test_check_redis_version_requires_supported_version(self) -> None:
        config = self._make_config()
        infer = self.module.InferScheduler(config)
        infer.client.info = lambda: {"redis_version": "5.0.0"}  # type: ignore[assignment]

        with self.assertRaises(AssertionError):
            infer.check_redis_version()


class SplitWiseSchedulerFacadeTest(SplitWiseSchedulerTestCase):
    def test_facade_delegates_to_components(self) -> None:
        module = self.module

        class _FakeAPI:
            def __init__(self, _config: Any) -> None:
                self.started = False
                self.reqs: List[Any] = []

            def start(self) -> None:
                self.started = True

            def put_requests(self, reqs: List[Any]):
                self.reqs.extend(reqs)
                return [(req.request_id, None) for req in reqs]

            def get_results(self):
                return {"x": 1}

        class _FakeInfer:
            def __init__(self, _config: Any) -> None:
                self.started = False
                self.nodeid = None

            def start(self, role, host, disaggregated):
                self.started = True

            def get_requests(self, *args, **kwargs):
                return ["scheduled"]

            def put_results(self, results):
                return list(results)

        original_api = module.APIScheduler
        original_infer = module.InferScheduler
        module.APIScheduler = _FakeAPI  # type: ignore[assignment]
        module.InferScheduler = _FakeInfer  # type: ignore[assignment]

        try:
            config = module.SplitWiseSchedulerConfig(
                enable_chunked_prefill=True,
                max_num_partial_prefills=1,
                max_long_partial_prefills=1,
                max_model_len=10,
            )
            facade = module.SplitWiseScheduler(config)

            facade.start("prefill", "host", {"tp": "ipc"})
            self.assertTrue(facade.scheduler.started)
            self.assertTrue(facade.infer.started)

            reqs = [module.Request("req", prompt_token_ids_len=1)]
            result = facade.put_requests(reqs)
            self.assertEqual(result[0][0], "req")
            self.assertEqual(facade.get_results(), {"x": 1})

            scheduled = facade.get_requests(10, 1, 1, 10, batch=1)
            self.assertEqual(scheduled, ["scheduled"])

            outputs = facade.put_results([1, 2])
            self.assertEqual(outputs, [1, 2])
        finally:
            module.APIScheduler = original_api  # type: ignore[assignment]
            module.InferScheduler = original_infer  # type: ignore[assignment]

    def test_get_requests_with_insufficient_resources(self) -> None:
        module = self.module
        config = module.SplitWiseSchedulerConfig(
            enable_chunked_prefill=True,
            max_num_partial_prefills=1,
            max_long_partial_prefills=1,
            max_model_len=10,
        )
        facade = module.SplitWiseScheduler(config)
        facade.infer = types.SimpleNamespace(get_requests=lambda *args, **kwargs: ["should not reach"])
        facade.scheduler = types.SimpleNamespace()

        result = facade.get_requests(
            available_blocks=1, block_size=1, reserved_output_blocks=2, max_num_batched_tokens=10
        )
        self.assertEqual(result, [])

        result = facade.get_requests(
            available_blocks=10, block_size=1, reserved_output_blocks=2, max_num_batched_tokens=10, batch=0
        )
        self.assertEqual(result, [])

    def test_start_uses_real_components(self) -> None:
        module = self.module
        config = module.SplitWiseSchedulerConfig(
            enable_chunked_prefill=True,
            max_num_partial_prefills=1,
            max_long_partial_prefills=1,
            max_model_len=10,
        )
        facade = module.SplitWiseScheduler(config)

        infer_flags = {}
        scheduler_flags = {}

        facade.infer = types.SimpleNamespace(
            start=lambda role, host, disagg: infer_flags.setdefault("called", (role, host, disagg)),
        )
        facade.scheduler = types.SimpleNamespace(start=lambda: scheduler_flags.setdefault("called", True))

        facade.start("prefill", "host", {"mode": "ipc"})
        self.assertEqual(infer_flags["called"], ("prefill", "host", {"mode": "ipc"}))
        self.assertTrue(scheduler_flags["called"])
        facade.reset_nodeid("new-id")
        self.assertEqual(facade.scheduler.nodeid, "new-id")


class BackgroundWorkerTest(SplitWiseSchedulerTestCase):
    def test_result_writer_start_flags_thread(self) -> None:
        client = sys.modules["redis"].Redis()
        writer = self.module.ResultWriter(client, idx=0, batch=2, ttl=5)
        writer.start()
        self.assertTrue(writer.thread.started)

    def test_result_writer_run_single_iteration(self) -> None:
        client = sys.modules["redis"].Redis()
        writer = self.module.ResultWriter(client, idx=0, batch=5, ttl=10)
        with writer.cond:
            writer.data.appendleft(("key", b"payload"))

        class _Pipeline:
            def __init__(self, parent):
                self.parent = parent

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return None

            def multi(self):
                return self

            def lpush(self, key, *items):
                self.parent.lpush(key, *items)
                return self

            def expire(self, key, ttl):
                raise SystemExit()

            def execute(self):
                return None

        client.pipeline = lambda: _Pipeline(client)  # type: ignore[assignment]

        with self.assertRaises(SystemExit):
            writer.run()

    def test_result_writer_run_groups_batches(self) -> None:
        client = sys.modules["redis"].Redis()
        writer = self.module.ResultWriter(client, idx=0, batch=10, ttl=5)

        with writer.cond:
            writer.data.appendleft(("k1", b"a"))
            writer.data.appendleft(("k1", b"b"))
            writer.data.appendleft(("k2", b"c"))

        def _pipeline():
            class _P:
                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return None

                def multi(self):
                    return self

                def lpush(self, key, *items):
                    client.lpush(key, *items)
                    return self

                def expire(self, key, ttl):
                    raise SystemExit()

                def execute(self):
                    return None

            return _P()

        client.pipeline = _pipeline  # type: ignore[assignment]
        with self.assertRaises(SystemExit):
            writer.run()

    def test_infer_scheduler_routine_report(self) -> None:
        config = self.module.SplitWiseSchedulerConfig(
            enable_chunked_prefill=True,
            max_num_partial_prefills=1,
            max_long_partial_prefills=1,
            max_model_len=10,
        )
        infer = self.module.InferScheduler(config)
        infer.node = self.module.NodeInfo("nid", "prefill", "host", {"transfer_protocol": ["ipc"]}, load=0)

        def _fake_hset(*_args, **_kwargs):
            raise ValueError("fail")

        infer.client.hset = _fake_hset  # type: ignore[assignment]
        original_logger = self.module.logger.error
        self.module.logger.error = lambda *_args, **_kwargs: (_ for _ in ()).throw(SystemExit())

        try:
            with self.assertRaises(SystemExit):
                infer.routine_report()
        finally:
            self.module.logger.error = original_logger

    def test_infer_scheduler_loop_expire_reqs(self) -> None:
        config = self.module.SplitWiseSchedulerConfig(
            enable_chunked_prefill=True,
            max_num_partial_prefills=1,
            max_long_partial_prefills=1,
            max_model_len=10,
        )
        infer = self.module.InferScheduler(config)
        infer.node = self.module.NodeInfo("nid", "prefill", "host", {"transfer_protocol": ["ipc"]}, load=0)

        def _raise_exit(ttl):
            raise SystemExit()

        infer.node.expire_reqs = _raise_exit  # type: ignore[assignment]

        with self.assertRaises(SystemExit):
            infer.loop_expire_reqs()

    def test_infer_scheduler_loop_get_reqs(self) -> None:
        config = self.module.SplitWiseSchedulerConfig(
            enable_chunked_prefill=True,
            max_num_partial_prefills=1,
            max_long_partial_prefills=1,
            max_model_len=10,
        )
        infer = self.module.InferScheduler(config)
        infer.role = "prefill"
        infer.node = self.module.NodeInfo(infer.nodeid, "prefill", "host", {"transfer_protocol": ["ipc"]}, load=0)
        infer.writers = [types.SimpleNamespace(put=lambda key, items: None)]

        req = self.module.Request("rq", prompt_token_ids_len=3)
        payload = pickle.dumps(dict(req.to_dict(), group=""), protocol=5)
        key = f"ReqQ_{infer.nodeid}"
        infer.client.storage[key] = [payload]

        state = {"called": False}

        def _fake_rpop(k, batch):
            if not state["called"]:
                state["called"] = True
                return infer.client.storage[k][:]
            raise SystemExit()

        infer.client.rpop = _fake_rpop  # type: ignore[assignment]
        infer.client.brpop = lambda *_args, **_kwargs: None  # type: ignore[assignment]

        with self.assertRaises(SystemExit):
            infer.loop_get_reqs()

    def test_infer_scheduler_get_requests_limits(self) -> None:
        config = self.module.SplitWiseSchedulerConfig(
            enable_chunked_prefill=True,
            max_num_partial_prefills=1,
            max_long_partial_prefills=1,
            max_model_len=50,
        )
        infer = self.module.InferScheduler(config)
        infer.role = "prefill"
        infer.node = self.module.NodeInfo("nid", "prefill", "host", {"transfer_protocol": ["ipc"]}, load=0)

        heavy = self.module.Request("heavy", prompt_token_ids_len=10)
        infer.reqs_queue.append(heavy)
        picked = infer.get_requests(
            available_blocks=1,
            block_size=4,
            reserved_output_blocks=1,
            max_num_batched_tokens=100,
            batch=1,
        )
        self.assertEqual(picked, [])

        infer.reqs_queue.clear()
        infer.reqs_queue.append(
            self.module.Request("long", prompt_token_ids_len=config.long_prefill_token_threshold + 10)
        )
        infer.reqs_queue.append(self.module.Request("short", prompt_token_ids_len=2))
        selected = infer.get_requests(
            available_blocks=100,
            block_size=4,
            reserved_output_blocks=1,
            max_num_batched_tokens=100,
            batch=2,
        )
        self.assertEqual(len(selected), 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--print-coverage-command", action="store_true")
    known_args, remaining = parser.parse_known_args()

    if known_args.print_coverage_command:
        print("python -m coverage run -m unittest tests.scheduler.test_splitwise_scheduler")
        print("python -m coverage report -m --include='fastdeploy/scheduler/splitwise_scheduler.py'")

    unittest.main(argv=[sys.argv[0]] + remaining)
