"""
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
"""

import copy
import hashlib
import math
import random
import threading
import time
import traceback
from collections import deque
from typing import List

import orjson
import redis

from fastdeploy.engine.request import (
    CompletionOutput,
    Request,
    RequestMetrics,
    RequestOutput,
)
from fastdeploy.utils import scheduler_logger as logger


class SplitWiseSchedulerConfig:
    """SplitWise Scheduler Configuration"""

    def __init__(
        self,
        nodeid=None,
        host="127.0.0.1",  # redis host
        port=6379,  # redis port
        password=None,  # redis password
        topic="fd",  # redis topic
        ttl=900,
        release_load_expire_period=600,  # s
        sync_period=5,  # ms
        expire_period=3000,  # ms
        clear_expired_nodes_period=60,  # s
        reader_parallel=4,
        reader_batch_size=200,
        writer_parallel=4,
        writer_batch_size=200,
        **kwargs,
    ):

        if nodeid is None:
            import uuid

            nodeid = str(uuid.uuid4())
        self.nodeid = nodeid

        self.redis_host = host
        self.redis_port = port
        self.redis_password = password
        self.redis_topic = topic
        self.ttl = ttl
        self.release_load_expire_period = release_load_expire_period

        self.sync_period = sync_period
        self.expire_period = expire_period / 1000.0
        self.clear_expired_nodes_period = clear_expired_nodes_period
        self.reader_parallel = reader_parallel
        self.reader_batch_size = reader_batch_size
        self.writer_parallel = writer_parallel
        self.writer_batch_size = writer_batch_size

        self.max_model_len = kwargs.get("max_model_len")
        self.enable_chunked_prefill = kwargs.get("enable_chunked_prefill")
        self.max_num_partial_prefills = kwargs.get("max_num_partial_prefills")
        self.max_long_partial_prefills = kwargs.get("max_long_partial_prefills")
        self.long_prefill_token_threshold = kwargs.get("long_prefill_token_threshold")

        assert self.enable_chunked_prefill is not None, "enable_chunked_prefill must be set"
        assert self.max_num_partial_prefills is not None, "max_num_partial_prefills must be set"
        assert self.max_long_partial_prefills is not None, "max_long_partial_prefills must be set"
        if self.long_prefill_token_threshold is None or self.long_prefill_token_threshold == 0:
            assert self.max_model_len is not None, "max_model_len must be set"
            self.long_prefill_token_threshold = int(self.max_model_len * 0.04)

    def check(self):
        """check argument"""
        pass

    def print(self):
        """
        print config
        """
        logger.info("LocalScheduler Configuration Information :")
        for k, v in self.__dict__.items():
            logger.info("{:<20}:{:<6}{}".format(k, "", v))
        logger.info("=============================================================")


class SplitWiseScheduler:
    """
    SplitWise Scheduler
    """

    def __init__(self, config):
        self.scheduler = APIScheduler(config)
        self.infer = InferScheduler(config)

    def start(self, role, host, disaggregated):
        """
        Start APIScheduler and InferScheduler backup threads
        """
        logger.info(f"Scheduler Start With: role:{role}, host:{host}, disaggregated:{disaggregated}")
        self.infer.start(role, host, disaggregated)
        self.scheduler.start()

    def reset_nodeid(self, nodeid):
        """
        reset node id
        """
        self.scheduler.nodeid = nodeid
        self.infer.nodeid = nodeid

    def put_requests(self, reqs: List[Request]):
        """
        put requests to global splitwise scheduler
        """
        return self.scheduler.put_requests(reqs)

    def get_results(self, request_ids=[]):
        """
        get results from global splitwise scheduler
        """
        return self.scheduler.get_results()

    def get_requests(
        self,
        available_blocks,
        block_size,
        reserved_output_blocks,
        max_num_batched_tokens,
        batch=1,
    ):
        """
        get scheduled requests from global spltiwise scheduler
        """
        if available_blocks <= reserved_output_blocks or batch < 1:
            logger.info(
                f"Scheduler's resource are insufficient: available_blocks={available_blocks} "
                f"reserved_output_blocks={reserved_output_blocks} batch={batch} "
                f"max_num_batched_tokens={max_num_batched_tokens}"
            )
            return []
        return self.infer.get_requests(
            available_blocks,
            block_size,
            reserved_output_blocks,
            max_num_batched_tokens,
            batch,
        )

    def put_results(self, results: List[RequestOutput]):
        """
        put results to global splitwise scheduler
        """
        return self.infer.put_results(results)


class NodeInfo:
    """
    Infer Node Info: load, rdma/ipc info
    """

    @classmethod
    def load_from(self, nodeid, info):
        """
        load node info from seiralized string
        """
        health = orjson.loads(info)
        ts = health["ts"]
        role = health["role"]
        load = int(health["load"])
        host = health["host"]
        disaggregated = health["disaggregated"]
        return NodeInfo(nodeid, role, host, disaggregated, load, ts)

    def __init__(self, nodeid, role, host, disaggregated, load, ts=time.time()):
        self.nodeid = nodeid
        self.ts = ts
        self.host = host
        self.disaggregated = disaggregated
        self.role = role
        self.lock = threading.Lock()
        self.load = load
        self.reqs = dict()

    def __repr__(self):
        return f"{self.nodeid}({self.load})"

    def expired(self, expire_period):
        """
        APIScheduler used to check if the node is expired
        """
        now = time.time()
        return (now - self.ts) > expire_period

    def serialize(self):
        """
        InferScheduler used to sync load
        """
        self.ts = time.time()
        health = {
            "ts": self.ts,
            "role": self.role,
            "load": self.load,
            "host": self.host,
            "disaggregated": self.disaggregated,
        }
        return orjson.dumps(health)

    def __lt__(self, other):
        return self.load < other.load

    def expire_reqs(self, ttl):
        """
        InferScheduler used to clear expired reqs
        """
        cur_time = time.time()
        with self.lock:
            expire_reqs = set()
            for req_id, pairs in self.reqs.items():
                load, arrival_time = pairs
                if cur_time - arrival_time > ttl:
                    logger.error(f"InferScheduler Expire Reqs({req_id}), arrival({arrival_time}), ttl({ttl})")
                    expire_reqs.add((req_id, load))
            for req_id, load in expire_reqs:
                if req_id in self.reqs:
                    self.load -= load
                    del self.reqs[req_id]

    def add_req(self, req_id, load):
        """
        InferScheduler used to record scheduled reqs(waiting or running)
        """
        with self.lock:
            if req_id not in self.reqs:
                self.reqs[req_id] = [load, time.time()]
                self.load += load

    def update_req_timestamp(self, req_ids):
        """
        InferScheduler used to update reqs timestamp
        """
        cur_time = time.time()
        with self.lock:
            for req_id in req_ids:
                if req_id in self.reqs:
                    self.reqs[req_id][1] = cur_time

    def finish_req(self, req_id):
        """
        InferScheduler used to clear finished reqs
        """
        with self.lock:
            if req_id in self.reqs:
                load = self.reqs[req_id][0]
                self.load -= load
                del self.reqs[req_id]


class ResultReader:
    """
    ResultReader use an async thread to continue get infer result from redis
    """

    def __init__(self, client, idx, batch=200, ttl=900, group=""):
        self.idx = idx
        self.batch = batch
        self.client = client
        self.data = deque()
        self.ttl = ttl
        self.group = group

        self.reqs = dict()
        self.out_buffer = dict()
        self.lock = threading.Lock()

        self.thread = threading.Thread(target=self.run)
        self.thread.start()

    def add_req(self, req):
        """
        add a req to reader, reader will async fetch infer result from redis
        """
        with self.lock:
            self.reqs[req.request_id] = {"arrival_time": req.arrival_time}
            self.out_buffer[req.request_id] = []

    def read(self):
        """
        batch read infer results
        returns: dict(req_id, [ResultOutput])
        """
        items = []
        size = len(self.data)
        for i in range(size):
            items.append(self.data.pop())

        outputs = dict()
        group_tokens = dict()
        finish_reqs = set()
        for item in items:
            req_id = item.request_id

            is_error = item.error_code != 200

            if is_error or item.finished:
                finish_reqs.add(req_id)

            if is_error or item.outputs.send_idx == 0:
                outputs[req_id] = [item]
                continue

            if req_id not in group_tokens:
                group_tokens[req_id] = []
            group_tokens[req_id].append(item)

        with self.lock:
            for key in finish_reqs:
                if key in self.reqs:
                    del self.reqs[key]

            for req_id, items in outputs.items():
                if req_id in self.out_buffer:
                    items.extend(self.out_buffer[req_id])
                    del self.out_buffer[req_id]

            for req_id, items in group_tokens.items():
                if req_id in self.out_buffer:
                    self.out_buffer[req_id].extend(items)
                    continue

                if req_id not in outputs:
                    outputs[req_id] = []
                outputs[req_id].extend(items)

            return outputs

    def run(self):
        """
        continue fetch infer results from redis
        """
        while True:
            try:
                keys = []
                cur_time = time.time()
                with self.lock:
                    expired_reqs = set()
                    for req_id, req in self.reqs.items():
                        if cur_time - req.get("arrival_time", cur_time) > self.ttl:
                            result = RequestOutput(
                                request_id=req_id,
                                prompt="",
                                prompt_token_ids=[],
                                outputs=CompletionOutput(-1, -1, []),
                                metrics=RequestMetrics(arrival_time=req["arrival_time"]),
                                error_code=500,
                                error_msg=f"Req({req_id}) is expired({self.ttl})",
                            )
                            self.data.appendleft(result)

                            logger.error(f"Req({req_id}) is expired({self.ttl})")
                            expired_reqs.add(req_id)
                            continue
                        keys.append(req_id)
                    for req_id in expired_reqs:
                        del self.reqs[req_id]

                if len(keys) == 0:
                    time.sleep(0.01)
                    continue

                total = self.sync_results(keys)
                if total == 0:
                    time.sleep(0.01)
            except Exception as e:
                logger.error(f"ResultsReader{self.idx} sync results error: {e!s}, {str(traceback.format_exc())}")

    def sync_results(self, keys):
        """
        fetch infer results from redis for the give keys
        """
        total = 0
        if self.group != "":
            keys = [self.group]
        for key in keys:
            # logger.info(f"Sync Results from Redis {key}")
            results = self.client.rpop(key, self.batch)
            if results is None or len(results) == 0:
                continue
            # logger.info(f"Rpop {key} {self.idx}: {len(results)}")
            total += len(results)
            for result in results:
                try:
                    # logger.info(f"Scheduler Get Results: {result.request_id}")
                    data = orjson.loads(result)
                    result = RequestOutput.from_dict(data)
                    self.data.appendleft(result)
                except Exception as e:
                    logger.error(f"Parse Result Error:{e}, {str(traceback.format_exc())}, {result}")
        return total


class APIScheduler:
    """
    APIScheduler: put requests to global schedule, and get recording infer results
    """

    def __init__(self, config):
        self.nodeid = config.nodeid
        self.reader_parallel = config.reader_parallel
        self.reader_batch_size = config.reader_batch_size
        self.expire_period = config.expire_period
        self.clear_expired_nodes_period = config.clear_expired_nodes_period
        self.ttl = config.ttl
        self.topic = config.redis_topic
        self.cluster_key = f"{self.topic}.cluster"

        self.client = redis.Redis(
            host=config.redis_host,
            port=config.redis_port,
            password=config.redis_password,
        )

        self.req_cond = threading.Condition()
        self.reqs_queue = deque()
        self.readers = []

    def start(self):
        """
        start backup threads
        """
        for i in range(self.reader_parallel):
            group = f"{self.nodeid}-{i}"
            reader = ResultReader(self.client, i, self.reader_batch_size, self.ttl, group)
            self.readers.append(reader)

        self.clear_expired_nodes_thread = threading.Thread(target=self.loop_clear_expired_nodes)
        self.clear_expired_nodes_thread.start()

        self.schedule_thread = threading.Thread(target=self.loop_schedule)
        self.schedule_thread.start()

    def put_requests(self, reqs):
        """
        put requests to local req queue. reqs will be async scheduled
        """
        ret = []
        with self.req_cond:
            for req in reqs:
                self.reqs_queue.appendleft(req)
                ret.append((req.request_id, None))
            self.req_cond.notify_all()
        return ret

    def get_results(self):
        """
        get infer results from local queue. results is async fetched from redis
        """
        outputs = dict()
        for reader in self.readers:
            outs = reader.read()
            outputs.update(outs)
        return outputs

    def loop_schedule(self):
        """
        loop schedule req based on global load states.
        """
        reader_idx = 0
        while True:
            try:
                with self.req_cond:
                    if len(self.reqs_queue) == 0:
                        self.req_cond.wait()

                pnodes, dnodes, mnodes = self.sync_cluster()
                if len(mnodes) == 0 and (len(pnodes) == 0 or len(dnodes) == 0):
                    logger.error(
                        f"No Schedule Nodes: mixed:{len(mnodes)}, prefill:{len(pnodes)}, decode:{len(dnodes)}"
                    )
                    time.sleep(1)
                    continue

                req = self.reqs_queue.pop()

                reader = self.readers[reader_idx]
                reader.add_req(req)
                group = self.readers[reader_idx].group
                reader_idx = (reader_idx + 1) % len(self.readers)

                self.schedule(req, pnodes, dnodes, mnodes, group)
            except IndexError:
                continue
            except Exception as e:
                logger.error(f"APIScheduler Schedule req error: {e!s}, {str(traceback.format_exc())}")

    def schedule(self, req, pnodes, dnodes, mnodes, group=""):
        """
        schedule an req to according redis node queue
        """
        pnodes.extend(mnodes)
        pnodes.sort()
        pnode = self.select_pd(req, pnodes, "prefill")
        if pnode.role == "mixed":
            req.disaggregate_info = None
            req_dict = req.to_dict()
            req_dict["group"] = group
            req_str = orjson.dumps(req_dict)
            pkey = f"ReqQ_{pnode.nodeid}"
            # logger.info(f"Schedule Req {req_str} to Mixed")
            self.client.lpush(pkey, req_str)
        else:
            dnodes.sort()
            dnode = self.select_pd(req, dnodes, "decode")
            disaggregated = copy.deepcopy(dnode.disaggregated)
            transfer_protocol = disaggregated["transfer_protocol"]
            if len(transfer_protocol) > 1 and "ipc" in transfer_protocol and "rdma" in transfer_protocol:
                if pnode.host == dnode.host:
                    disaggregated["transfer_protocol"] = "ipc"
                else:
                    disaggregated["transfer_protocol"] = "rdma"
            else:
                disaggregated["transfer_protocol"] = transfer_protocol[0]
            req.disaggregate_info = disaggregated
            pkey, dkey = f"ReqQ_{pnode.nodeid}", f"ReqQ_{dnode.nodeid}"
            req_dict = req.to_dict()
            req_dict["group"] = group
            req_str = orjson.dumps(req_dict)
            # logger.info(f"Schedule Req {req_str}")
            self.client.lpush(dkey, req_str)
            self.client.lpush(pkey, req_str)

    def sync_cluster(self):
        """
        fetch cluster load states from redis
        """
        clusters = self.client.hgetall(self.cluster_key)
        pnodes, dnodes, mnodes = [], [], []
        for nodeid, info in clusters.items():
            node = NodeInfo.load_from(nodeid.decode(), info)
            if node.expired(self.expire_period):
                logger.error(f"node {nodeid} is expired: {info}")
                continue
            if node.role == "prefill":
                pnodes.append(node)
            elif node.role == "decode":
                dnodes.append(node)
            elif node.role == "mixed":
                mnodes.append(node)
            else:
                logger.error(f"Invalid Role: {node.role} {info}")
        return pnodes, dnodes, mnodes

    def loop_clear_expired_nodes(self):
        """
        loop clear expired node's dirty data in redis
        """
        while True:
            try:
                expire_nodes = set()
                clusters = self.client.hgetall(self.cluster_key)
                for nodeid, info in clusters.items():
                    node = NodeInfo.load_from(nodeid.decode(), info)
                    if node.expired(self.clear_expired_nodes_period):
                        expire_nodes.add(nodeid)
                for nodeid in expire_nodes:
                    # logger.info(f"clear expired nodes: {nodeid}")
                    self.client.hdel(self.cluster_key, nodeid)
                time.sleep(self.clear_expired_nodes_period)
            except Exception as e:
                logger.error(f"APIScheduler clear expired nodes error: {str(e)}, {str(traceback.format_exc())}")

    def select_pd(self, req, nodes, role):
        """
        select a prefill/decode/mixed node based on load states
        """

        def select(req, nodes, blur_step):
            min_load = nodes[0].load
            blur_max = min_load + blur_step
            blur_idx = 0
            for idx, node in enumerate(nodes):
                if node.load >= blur_max:
                    break
                blur_idx = idx
            node = random.choice(nodes[: blur_idx + 1])
            logger.info(f"Schedule Req {req.request_id}(len:{req.prompt_token_ids_len}) to {node}")
            return node

        if role == "prefill" or role == "mixed":
            size = req.prompt_token_ids_len
            rate = 2 if size < 1000 else 10
            pblur_step = max(100, min(500, int(size / rate)))
            pnode = select(req, nodes, pblur_step)
            return pnode
        elif role == "decode":
            dblur_step = min(len(nodes), 10)
            dnode = select(req, nodes, dblur_step)
            return dnode

        raise Exception(f"Invalid Role: {role}")


class ResultWriter:
    """
    ResultWriter use an async thread to continue writer infer results to redis
    """

    def __init__(self, client, idx, batch, ttl=900):
        self.idx = idx
        self.batch = batch
        self.client = client
        self.data = deque()
        self.cond = threading.Condition()
        self.thread = threading.Thread(target=self.run)
        self.ttl = ttl

    def start(self):
        """start backup thread"""
        self.thread.start()

    def put(self, key, items):
        """
        put infer results to writer
        """
        with self.cond:
            for item in items:
                self.data.appendleft((key, item))
            self.cond.notify_all()

    def run(self):
        """
        continue batch write infer results to redis
        """
        while True:
            try:
                with self.cond:
                    size = len(self.data)
                    if size == 0:
                        self.cond.wait()
                # qsize = size
                size = min(size, self.batch)
                # logger.info(f"Writer {self.idx} Queue Size: {qsize}, Cur Size: {size}")
                groups = dict()
                for i in range(size):
                    key, item = self.data.pop()
                    if key not in groups:
                        groups[key] = []
                    groups[key].append(item)
                for key, items in groups.items():
                    # s = time.time()
                    with self.client.pipeline() as pipe:
                        pipe.multi()
                        pipe.lpush(key, *items)
                        pipe.expire(key, math.ceil(self.ttl))
                        pipe.execute()
                    # self.client.lpush(key, *items)
                    # e = time.time()
                    # logger.info(f"Lpush {self.idx}: {key} used {e-s} {len(items)} items")
            except Exception as e:
                logger.error(f"ResultWriter write error: {e!s}, {str(traceback.format_exc())}")


class InferScheduler:
    """
    InferScheduler: get scheduled requests to local queue, write results to redis
    """

    def __init__(self, config):
        self.config = config
        self.nodeid = config.nodeid
        self.writer_parallel = config.writer_parallel
        self.writer_batch_size = config.writer_batch_size
        self.sync_period = config.sync_period
        self.topic = config.redis_topic
        self.cluster_key = f"{self.topic}.cluster"
        self.ttl = config.ttl
        self.release_load_expire_period = config.release_load_expire_period

        self.client = redis.Redis(
            host=config.redis_host,
            port=config.redis_port,
            password=config.redis_password,
        )

        self.reqs_queue = deque()
        self.writers = []

    def check_redis_version(self):
        # Get Redis version information
        redis_info = self.client.info()
        redis_version = redis_info.get("redis_version", "")
        version_parts = [int(x) for x in redis_version.split(".")]

        # Redis 6.2 and above versions support RPOP with count parameter
        assert (
            version_parts[0] >= 6
        ), f"Redis major version too low: {version_parts[0]}. Please upgrade to Redis 6.2+ to support batch RPOP operations."
        assert (
            version_parts[1] >= 2 if version_parts[0] == 6 else True
        ), f"Redis version {redis_version} too low. Please upgrade to Redis 6.2+ to support batch RPOP operations."

        logger.info(f"Redis version {redis_version} detected. Using native batch RPOP.")

    def start(self, role, host, disaggregated):
        """
        start backup threads
        """

        # Check Redis version first
        self.check_redis_version()

        for i in range(self.writer_parallel):
            writer = ResultWriter(self.client, i, self.writer_batch_size, self.ttl)
            writer.start()
            self.writers.append(writer)

        self.getreq_thread = threading.Thread(target=self.loop_get_reqs)
        self.getreq_thread.start()

        self.role = role
        self.host = host
        self.node = NodeInfo(self.nodeid, role, host, disaggregated, 0)

        self.report_thread = threading.Thread(target=self.routine_report)
        self.report_thread.start()

        self.expire_reqs_thread = threading.Thread(target=self.loop_expire_reqs)
        self.expire_reqs_thread.start()

    def routine_report(self):
        """
        routine report node info: load, health
        """
        while True:
            try:
                info = self.node.serialize()
                self.client.hset(self.cluster_key, self.nodeid, info)
                time.sleep(self.sync_period / 1000.0)
            except Exception as e:
                logger.error(f"InferScheduler routine report error: {e!s}, {str(traceback.format_exc())}")

    def loop_expire_reqs(self):
        """
        loop clear expired reqs
        """
        while True:
            try:
                self.node.expire_reqs(self.release_load_expire_period)
                time.sleep(60)
            except Exception as e:
                logger.error(f"InferScheduler expire reqs error: {e}, {str(traceback.format_exc())}")

    def loop_get_reqs(self):
        """
        loop get global scheduled reqs to local queue
        """

        def select_writer(req):
            req_id = req.request_id
            md5 = hashlib.md5()
            md5.update(req_id.encode())
            writer_idx = int(md5.hexdigest(), 16) % len(self.writers)
            return writer_idx

        batch = 50
        while True:
            try:
                key = f"ReqQ_{self.nodeid}"
                reqs = self.client.rpop(key, batch)
                if reqs is None:
                    ret = self.client.brpop([key], timeout=1)
                    if ret is None:
                        continue
                    reqs = [ret[1]]

                for req_str in reqs:
                    req = orjson.loads(req_str)
                    group = req.get("group", "")
                    req = Request.from_dict(req)
                    writer_idx = select_writer(req)
                    logger.info(f"Infer Scheduler Get Req: {req.request_id} writer idx {writer_idx}")
                    req.request_id = f"{req.request_id}#{writer_idx}#{group}"
                    if self.role == "prefill" or self.role == "mixed":
                        self.reqs_queue.append(req)
                        self.node.add_req(req.request_id, req.prompt_token_ids_len)
                    else:
                        self.node.add_req(req.request_id, 1)
            except Exception as e:
                logger.error(f"InferScheduler loop get reqs error: {e!s}, {str(traceback.format_exc())}")

    def get_requests(
        self,
        available_blocks,
        block_size,
        reserved_output_blocks,
        max_num_batched_tokens,
        batch,
    ):
        """
        get scheduled reqs from local reqs queue
        """
        if len(self.reqs_queue) == 0:
            return []

        reqs = []
        required_blocks = 0
        current_prefill_tokens = 0
        long_partial_requests, short_partial_requests = 0, 0
        cur_time = time.time()
        for i in range(batch):
            try:
                if len(self.reqs_queue) == 0:
                    break

                req = self.reqs_queue.popleft()
                if cur_time - req.arrival_time > self.ttl:
                    logger.error(f"req({req.request_id}) is expired({self.ttl}) when InferScheduler Get Requests")
                    self.node.finish_req(req.request_id)
                    continue
                current_prefill_tokens += req.prompt_token_ids_len
                required_input_blocks = (req.prompt_token_ids_len + block_size - 1) // block_size
                required_blocks += required_input_blocks + reserved_output_blocks
                if required_blocks > available_blocks:
                    self.reqs_queue.appendleft(req)
                    return reqs

                if self.config.enable_chunked_prefill:
                    if req.prompt_token_ids_len > self.config.long_prefill_token_threshold:
                        # long partial requests
                        long_partial_requests += 1
                        if long_partial_requests > self.config.max_long_partial_prefills:
                            self.reqs_queue.appendleft(req)
                            break
                    else:
                        short_partial_requests += 1

                    if short_partial_requests + long_partial_requests > self.config.max_num_partial_prefills:
                        self.reqs_queue.appendleft(req)
                        break
                else:
                    if current_prefill_tokens > max_num_batched_tokens:
                        self.reqs_queue.appendleft(req)
                        break
                # logger.info(f"Get Requests from Scheduler: {req.request_id}")
                reqs.append(req)
            except Exception as e:
                logger.error(f"InferScheduler get requests error: {e}, {str(traceback.format_exc())}")
                return reqs
        return reqs

    def put_results(self, results):
        """
        put infer results to according writer's local queue
        """
        groups = dict()
        req_ids = set()
        for result in results:
            if result.error_code != 200 or result.finished:
                self.node.finish_req(result.request_id)
                logger.info(f"{result.request_id} finished, node load is {self.node.load}")

            req_ids.add(result.request_id)

            req_id, idx, group = result.request_id.split("#")
            result.request_id = req_id

            key = (req_id if group == "" else group, int(idx))
            if key not in groups:
                groups[key] = list()

            if self.role == "prefill" and result.outputs.send_idx == 0:
                result.finished = False

            result_str = orjson.dumps(result.to_dict())
            # if self.role == "prefill" or result.error_code != 200 or result.finished:
            #    logger.info(f"Infer Put Finish Result: {result_str}")
            groups[key].append(result_str)

        self.node.update_req_timestamp(req_ids)

        for key, outputs in groups.items():
            req_id, idx = key
            self.writers[idx].put(req_id, outputs)
