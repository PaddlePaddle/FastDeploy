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

ResourceManagerV2 — Lock-free scheduling via single-writer message queue
=========================================================================

Design
------
V2 inherits V1 and reuses **all** logic from V1's methods.  The only
difference is how external threads invoke state-mutating methods:

- V1: external threads acquire ``self.lock`` and run the method body.
- V2: external threads enqueue a message; ``schedule()`` drains the queue
  and calls ``super().method()`` directly.

Since ``self.lock`` is replaced with a ``_NoOpLock``, every
``with self.lock:`` in V1 becomes a transparent pass-through.  V2 contains
**zero duplicated logic** — all message processors delegate to the parent
class via ``super()``.
"""

import threading
import time
import traceback
from collections import deque
from collections.abc import Iterable
from typing import Union

from fastdeploy.engine.request import Request, RequestOutput
from fastdeploy.engine.sched.request_manager import RequestManager
from fastdeploy.engine.sched.resource_manager_v1 import ResourceManagerV1
from fastdeploy.utils import llm_logger

# ---------------------------------------------------------------------------
#  No-op lock — makes every ``with self.lock:`` a transparent pass-through
# ---------------------------------------------------------------------------


class _NoOpLock:
    """Context manager that does nothing. Replaces V1's global lock."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def acquire(self, *args, **kwargs):
        pass

    def release(self):
        pass


# ---------------------------------------------------------------------------
#  Message types
# ---------------------------------------------------------------------------


class _Msg:
    ADD_REQUEST = "add_request"
    FINISH_REQUESTS = "finish_requests"
    RESCHEDULE_PREEMPT = "reschedule_preempt"
    RECYCLE_ABORT = "recycle_abort"
    ADD_ABORT_IDS = "add_abort_ids"
    CACHE_OUTPUT_TOKENS = "cache_output_tokens"
    PRE_RECYCLE_RESOURCE = "pre_recycle_resource"
    ADD_REQUEST_IN_P = "add_request_in_p"
    PREALLOCATE_IN_P = "preallocate_in_p"
    PREALLOCATE_IN_D = "preallocate_in_d"
    HAS_RESOURCE_FOR_PREFILLED = "has_resource_for_prefilled"
    ADD_PREFILLED_REQUEST = "add_prefilled_request"


class ResourceManagerV2(ResourceManagerV1):
    """
    Lock-free scheduler that reuses V1's logic verbatim via ``super()``.

    External threads do not acquire any lock.  Instead they enqueue messages
    into ``_msg_queue``.  ``schedule()`` drains the queue, then delegates to
    ``super().schedule()`` which runs with the no-op lock — effectively
    lock-free.
    """

    def __init__(self, config):
        super().__init__(
            config.scheduler_config.max_num_seqs,
            config,
            config.parallel_config.tensor_parallel_size,
            config.scheduler_config.splitwise_role,
            config.parallel_config.local_data_parallel_id,
        )
        # ---- Replace the global lock with a no-op ----
        self.lock = _NoOpLock()

        # ---- Unified slot management interface (wraps V1's arrays) ----
        self._req_mgr = RequestManager(self.stop_flags, self.tasks_list)

        # ---- Lock-free message queue ----
        self._msg_queue: deque = deque()
        self._msg_lock = threading.Lock()  # only protects _msg_queue

        # ---- Futures for synchronous query methods (P/D disaggregation) ----
        self._result_futures: dict[str, threading.Event] = {}
        self._result_values: dict[str, object] = {}

    # ==================================================================
    #  Message queue helpers
    # ==================================================================

    def _enqueue_msg(self, msg_type: str, *args) -> None:
        with self._msg_lock:
            self._msg_queue.append((msg_type, *args))

    def _drain_messages(self) -> int:
        """Drain and process all pending messages. Called inside schedule()."""
        with self._msg_lock:
            msgs = list(self._msg_queue)
            self._msg_queue.clear()

        count = 0
        for msg in msgs:
            msg_type = msg[0]
            args = msg[1:]
            try:
                handler = self._MSG_HANDLERS.get(msg_type)
                if handler is not None:
                    handler(self, args)
                else:
                    llm_logger.warning(f"Unknown message type: {msg_type}")
            except Exception as e:
                llm_logger.error(f"Error processing message {msg_type}: {e}, {traceback.format_exc()}")
            count += 1
        return count

    # ==================================================================
    #  schedule() — drain messages, then delegate to V1 (lock-free)
    # ==================================================================

    def schedule(self):
        """
        Drain pending messages, then delegate to V1's schedule().
        V1's ``with self.lock:`` becomes a no-op because ``self.lock``
        is a ``_NoOpLock``.
        """
        self._drain_messages()
        return super().schedule()

    # ==================================================================
    #  External-thread API — override V1 to use message queue
    # ==================================================================

    def add_request(self, request: Request) -> None:
        """Enqueue an add-request message. Non-blocking."""
        # apply_async_preprocess is called here (before enqueuing) so that
        # the async download starts as early as possible.  The processor
        # only does waiting.append + requests[...] — it must NOT call
        # super().add_request() because that would call apply_async_preprocess
        # a second time, submitting a duplicate download task.
        self.apply_async_preprocess(request)
        self._enqueue_msg(_Msg.ADD_REQUEST, request)

    def finish_requests(self, request_ids: Union[str, Iterable[str]]) -> None:
        """
        Immediate slot release + deferred block recycle.

        Slot release (stop_flags[idx] = True) is the most latency-sensitive
        operation — it directly controls available_batch() which gates how
        many new requests schedule() can accept.  A single list element
        assignment is GIL-atomic, so we do it right here without any lock.

        Block recycle + cache write + state cleanup are deferred to the
        message queue so they run in schedule()'s drain phase.  This avoids
        concurrent modification of self.running (which schedule iterates)
        and keeps I/O off the caller's thread.
        """
        if isinstance(request_ids, str):
            request_ids = (request_ids,)
        else:
            request_ids = set(request_ids)

        # Immediate: release slots via RequestManager
        for req_id in request_ids:
            request = self.requests.get(req_id)
            if request is not None:
                self._req_mgr.release_slot(request.idx)

        # Deferred: block recycle + cache write + state cleanup
        self._enqueue_msg(_Msg.FINISH_REQUESTS, request_ids)

    def reschedule_preempt_task(self, request_id, process_func=None):
        """Enqueue a reschedule message. Non-blocking."""
        self._enqueue_msg(_Msg.RESCHEDULE_PREEMPT, request_id, process_func)

    def recycle_abort_task(self, request_id):
        """Enqueue a recycle-abort message. Non-blocking."""
        self._enqueue_msg(_Msg.RECYCLE_ABORT, request_id)

    def add_abort_req_ids(self, req_ids):
        """Enqueue an add-abort-ids message. Non-blocking."""
        self._enqueue_msg(_Msg.ADD_ABORT_IDS, req_ids)

    def cache_output_tokens(self, request):
        """Enqueue a cache-output-tokens message. Non-blocking."""
        self._enqueue_msg(_Msg.CACHE_OUTPUT_TOKENS, request)

    def pre_recycle_resource(self, request_id: str):
        """Enqueue a pre-recycle-resource message. Non-blocking."""
        self._enqueue_msg(_Msg.PRE_RECYCLE_RESOURCE, request_id)

    def add_request_in_p(self, requests: list[Request]):
        """Enqueue an add-request-in-p message. Non-blocking."""
        self._enqueue_msg(_Msg.ADD_REQUEST_IN_P, requests)

    # ------------------------------------------------------------------
    #  Synchronous query methods (P/D disaggregation)
    # ------------------------------------------------------------------

    def preallocate_resource_in_p(self, request: Request):
        return self._sync_query(_Msg.PREALLOCATE_IN_P, request)

    def preallocate_resource_in_d(self, request: Request):
        return self._sync_query(_Msg.PREALLOCATE_IN_D, request)

    def has_resource_for_prefilled_req(self, request_id: str):
        return self._sync_query(_Msg.HAS_RESOURCE_FOR_PREFILLED, request_id)

    def add_prefilled_request(self, request_output: RequestOutput):
        return self._sync_query(_Msg.ADD_PREFILLED_REQUEST, request_output)

    def _sync_query(self, msg_type: str, payload):
        """Submit a synchronous query and wait for schedule() to process it."""
        query_id = f"{msg_type}_{id(payload)}_{time.time()}"
        event = threading.Event()
        self._result_futures[query_id] = event
        self._enqueue_msg(msg_type, payload, query_id)
        event.wait()  # block until schedule() processes and signals
        result = self._result_values.pop(query_id, None)
        self._result_futures.pop(query_id, None)
        return result

    def _set_query_result(self, query_id: str, result) -> None:
        self._result_values[query_id] = result
        event = self._result_futures.get(query_id)
        if event is not None:
            event.set()

    # ==================================================================
    #  Message processors — all delegate to super(), zero code duplication
    # ==================================================================

    def _process_add_request(self, args) -> None:
        """
        Only handler that does NOT call super().add_request(), because
        apply_async_preprocess was already called before enqueuing.
        """
        request = args[0]
        self.waiting.append(request)
        self.requests[request.request_id] = request

    def _process_finish_requests(self, args) -> None:
        super().finish_requests(args[0])

    def _process_reschedule_preempt(self, args) -> None:
        request_id = args[0]
        process_func = args[1] if len(args) > 1 else None
        super().reschedule_preempt_task(request_id, process_func)

    def _process_recycle_abort(self, args) -> None:
        super().recycle_abort_task(args[0])

    def _process_add_abort_ids(self, args) -> None:
        super().add_abort_req_ids(args[0])

    def _process_cache_output_tokens(self, args) -> None:
        super().cache_output_tokens(args[0])

    def _process_pre_recycle_resource(self, args) -> None:
        super().pre_recycle_resource(args[0])

    def _process_add_request_in_p(self, args) -> None:
        super().add_request_in_p(args[0])

    # ------------------------------------------------------------------
    #  Synchronous query processors — call super() + set Future result
    # ------------------------------------------------------------------

    def _process_preallocate_in_p(self, args) -> None:
        request, query_id = args[0], args[1]
        result = super().preallocate_resource_in_p(request)
        self._set_query_result(query_id, result)

    def _process_preallocate_in_d(self, args) -> None:
        request, query_id = args[0], args[1]
        result = super().preallocate_resource_in_d(request)
        self._set_query_result(query_id, result)

    def _process_has_resource_for_prefilled(self, args) -> None:
        request_id, query_id = args[0], args[1]
        result = super().has_resource_for_prefilled_req(request_id)
        self._set_query_result(query_id, result)

    def _process_add_prefilled_request(self, args) -> None:
        request_output, query_id = args[0], args[1]
        super().add_prefilled_request(request_output)
        self._set_query_result(query_id, True)

    # ------------------------------------------------------------------
    #  Handler dispatch table
    # ------------------------------------------------------------------

    _MSG_HANDLERS = {
        _Msg.ADD_REQUEST: _process_add_request,
        _Msg.FINISH_REQUESTS: _process_finish_requests,
        _Msg.RESCHEDULE_PREEMPT: _process_reschedule_preempt,
        _Msg.RECYCLE_ABORT: _process_recycle_abort,
        _Msg.ADD_ABORT_IDS: _process_add_abort_ids,
        _Msg.CACHE_OUTPUT_TOKENS: _process_cache_output_tokens,
        _Msg.PRE_RECYCLE_RESOURCE: _process_pre_recycle_resource,
        _Msg.ADD_REQUEST_IN_P: _process_add_request_in_p,
        _Msg.PREALLOCATE_IN_P: _process_preallocate_in_p,
        _Msg.PREALLOCATE_IN_D: _process_preallocate_in_d,
        _Msg.HAS_RESOURCE_FOR_PREFILLED: _process_has_resource_for_prefilled,
        _Msg.ADD_PREFILLED_REQUEST: _process_add_prefilled_request,
    }
