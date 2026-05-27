"""
# Copyright (c) 2026  PaddlePaddle Authors. All Rights Reserved.
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

import unittest
from unittest.mock import MagicMock, patch


def _make_cfg(
    router_url="http://router:8080",
    role="prefill",
    rdma_eager=True,
    transfer_protocol=None,
    rdma_ports=None,
):
    """Create a mock FDConfig for RegisterManager."""
    cfg = MagicMock()
    cfg.router_config.router = router_url
    cfg.router_config.api_server_host = "127.0.0.1"
    cfg.router_config.api_server_port = 8088
    cfg.scheduler_config.splitwise_role = role
    cfg.model_config.version = "v1.0"
    if transfer_protocol is None:
        transfer_protocol = ["rdma"]
    if rdma_ports is None:
        rdma_ports = [18515]
    cfg.register_info = {
        "host_ip": "10.0.0.1",
        "port": 8088,
        "role": role,
        "transfer_protocol": transfer_protocol,
        "rdma_ports": rdma_ports,
    }
    cfg.cache_config.local_rdma_comm_ports = rdma_ports
    return cfg


def _make_manager(cfg=None, **kwargs):
    """Create a RegisterManager with mocked dependencies."""
    from fastdeploy.engine.register_manager import RegisterManager

    if cfg is None:
        cfg = _make_cfg(**kwargs)
    queue = MagicMock()
    get_is_paused = MagicMock(return_value=False)
    return RegisterManager(cfg, queue, get_is_paused)


class TestRegisterManagerInit(unittest.TestCase):
    """Test RegisterManager.__init__."""

    def test_init_stores_attributes(self):
        """Init stores cfg, queue, get_is_paused and sets defaults."""
        from fastdeploy.engine.register_manager import RegisterManager

        cfg = _make_cfg()
        queue = MagicMock()
        get_paused = MagicMock(return_value=True)

        mgr = RegisterManager(cfg, queue, get_paused)

        self.assertIs(mgr.cfg, cfg)
        self.assertIs(mgr.engine_worker_queue, queue)
        self.assertIs(mgr.get_is_paused, get_paused)
        self.assertFalse(mgr._is_registered)
        self.assertEqual(mgr.connected_decodes, [])
        self.assertEqual(mgr.connect_status, {})
        self.assertEqual(mgr._timeout, 5)
        self.assertEqual(mgr._sleep_seconds, 5)


class TestGetConnectedDecodes(unittest.TestCase):
    """Test get_connected_decodes (lines 86-87)."""

    def test_returns_copy_of_connected_decodes(self):
        """get_connected_decodes() returns a copy, not a reference."""
        mgr = _make_manager()
        mgr.connected_decodes = [{"host_ip": "10.0.0.1", "port": 8080}]

        result = mgr.get_connected_decodes()

        self.assertEqual(result, [{"host_ip": "10.0.0.1", "port": 8080}])
        # Mutating result should not affect internal state
        result.append({"host_ip": "10.0.0.2", "port": 8081})
        self.assertEqual(len(mgr.connected_decodes), 1)

    def test_returns_empty_list_initially(self):
        """get_connected_decodes() returns empty list when nothing connected."""
        mgr = _make_manager()
        self.assertEqual(mgr.get_connected_decodes(), [])


class TestIsRegistered(unittest.TestCase):
    """Test is_registered (line 91)."""

    def test_is_registered_false_initially(self):
        """is_registered() returns False before registration."""
        mgr = _make_manager()
        self.assertFalse(mgr.is_registered())

    def test_is_registered_true_after_set(self):
        """is_registered() returns True when _is_registered is set."""
        mgr = _make_manager()
        mgr._is_registered = True
        self.assertTrue(mgr.is_registered())


class TestShouldEnableEagerConnect(unittest.TestCase):
    """Test _should_enable_eager_connect (lines 206, 210-215)."""

    def test_enabled_when_all_conditions_met(self):
        """Returns True when router, role, env, protocol, ports all valid."""
        mgr = _make_manager(
            router_url="http://router:8080",
            role="prefill",
            transfer_protocol=["rdma"],
            rdma_ports=[18515],
        )
        with patch("fastdeploy.engine.register_manager.envs") as mock_envs:
            mock_envs.FD_ENABLE_PD_RDMA_EAGER_CONNECT = True
            result = mgr._should_enable_eager_connect()
        self.assertTrue(result)

    def test_disabled_when_no_router(self):
        """Returns False when router is None."""
        mgr = _make_manager(router_url=None)
        self.assertFalse(mgr._should_enable_eager_connect())

    def test_disabled_when_not_prefill(self):
        """Returns False when role is not 'prefill'."""
        mgr = _make_manager(role="decode")
        with patch("fastdeploy.engine.register_manager.envs") as mock_envs:
            mock_envs.FD_ENABLE_PD_RDMA_EAGER_CONNECT = True
            result = mgr._should_enable_eager_connect()
        self.assertFalse(result)

    def test_disabled_when_env_not_set(self):
        """Returns False when FD_ENABLE_PD_RDMA_EAGER_CONNECT is False."""
        mgr = _make_manager(role="prefill")
        with patch("fastdeploy.engine.register_manager.envs") as mock_envs:
            mock_envs.FD_ENABLE_PD_RDMA_EAGER_CONNECT = False
            result = mgr._should_enable_eager_connect()
        self.assertFalse(result)

    def test_disabled_when_no_rdma_protocol(self):
        """Returns False when 'rdma' not in transfer_protocol."""
        mgr = _make_manager(transfer_protocol=["ipc"], rdma_ports=[18515])
        with patch("fastdeploy.engine.register_manager.envs") as mock_envs:
            mock_envs.FD_ENABLE_PD_RDMA_EAGER_CONNECT = True
            result = mgr._should_enable_eager_connect()
        self.assertFalse(result)

    def test_disabled_when_no_rdma_ports(self):
        """Returns False when rdma_ports is empty."""
        mgr = _make_manager(transfer_protocol=["rdma"], rdma_ports=[])
        with patch("fastdeploy.engine.register_manager.envs") as mock_envs:
            mock_envs.FD_ENABLE_PD_RDMA_EAGER_CONNECT = True
            result = mgr._should_enable_eager_connect()
        self.assertFalse(result)


class TestGetInstanceKey(unittest.TestCase):
    """Test _get_instance_key (line 329)."""

    def test_generates_key(self):
        """_get_instance_key returns 'host_ip:port'."""
        mgr = _make_manager()
        instance = {"host_ip": "192.168.1.100", "port": 9090}
        self.assertEqual(mgr._get_instance_key(instance), "192.168.1.100:9090")

    def test_handles_missing_fields(self):
        """_get_instance_key handles missing keys gracefully."""
        mgr = _make_manager()
        self.assertEqual(mgr._get_instance_key({}), "None:None")


class TestSupportsRdma(unittest.TestCase):
    """Test _supports_rdma (lines 333-334)."""

    def test_supports_rdma_true(self):
        """Returns True when rdma in transfer_protocol and rdma_ports set."""
        mgr = _make_manager()
        instance = {"transfer_protocol": ["rdma", "ipc"], "rdma_ports": [18515]}
        self.assertTrue(mgr._supports_rdma(instance))

    def test_not_supports_no_rdma_protocol(self):
        """Returns False when rdma not in transfer_protocol."""
        mgr = _make_manager()
        instance = {"transfer_protocol": ["ipc"], "rdma_ports": [18515]}
        self.assertFalse(mgr._supports_rdma(instance))

    def test_not_supports_no_rdma_ports(self):
        """Returns False when rdma_ports is empty/None."""
        mgr = _make_manager()
        instance = {"transfer_protocol": ["rdma"], "rdma_ports": []}
        self.assertFalse(mgr._supports_rdma(instance))

    def test_not_supports_missing_fields(self):
        """Returns False when fields are missing."""
        mgr = _make_manager()
        self.assertFalse(mgr._supports_rdma({}))


class TestCheckInstanceHealth(unittest.TestCase):
    """Test _check_instance_health (lines 338-346)."""

    @patch("fastdeploy.engine.register_manager.requests.get")
    def test_healthy_instance(self, mock_get):
        """Returns True when health endpoint returns 200."""
        mgr = _make_manager()
        mock_get.return_value = MagicMock(status_code=200)

        instance = {"host_ip": "10.0.0.1", "port": 8080}
        result = mgr._check_instance_health(instance)

        self.assertTrue(result)
        mock_get.assert_called_once_with("http://10.0.0.1:8080/health", timeout=5)

    @patch("fastdeploy.engine.register_manager.requests.get")
    def test_unhealthy_instance(self, mock_get):
        """Returns False when health endpoint returns non-200."""
        mgr = _make_manager()
        mock_get.return_value = MagicMock(status_code=503)

        instance = {"host_ip": "10.0.0.1", "port": 8080}
        result = mgr._check_instance_health(instance)

        self.assertFalse(result)

    @patch("fastdeploy.engine.register_manager.requests.get", side_effect=Exception("timeout"))
    def test_exception_returns_false(self, mock_get):
        """Returns False on request exception."""
        mgr = _make_manager()
        instance = {"host_ip": "10.0.0.1", "port": 8080}
        result = mgr._check_instance_health(instance)
        self.assertFalse(result)


class TestTryRdmaConnect(unittest.TestCase):
    """Test _try_rdma_connect (lines 365-386)."""

    def test_connect_success(self):
        """Returns True when connect_status gets successful result."""
        mgr = _make_manager()
        instance = {"host_ip": "10.0.0.2", "rdma_ports": [18515], "port": 8080}

        # Simulate the response loop setting connect_status
        def _put_task(task):
            task_id = task["task_id"]
            with mgr._lock:
                mgr.connect_status[task_id] = True

        mgr.engine_worker_queue.put_connect_rdma_task.side_effect = _put_task

        result = mgr._try_rdma_connect(instance)
        self.assertTrue(result)
        # connect_status should be cleaned up
        self.assertEqual(mgr.connect_status, {})

    def test_connect_failure(self):
        """Returns False when connect_status gets failure result."""
        mgr = _make_manager()
        instance = {"host_ip": "10.0.0.2", "rdma_ports": [18515], "port": 8080}

        def _put_task(task):
            task_id = task["task_id"]
            with mgr._lock:
                mgr.connect_status[task_id] = False

        mgr.engine_worker_queue.put_connect_rdma_task.side_effect = _put_task

        result = mgr._try_rdma_connect(instance)
        self.assertFalse(result)

    def test_connect_timeout(self):
        """Returns False on timeout (no response arrives)."""
        mgr = _make_manager()
        mgr._timeout = 0.1  # Short timeout for test
        instance = {"host_ip": "10.0.0.2", "rdma_ports": [18515], "port": 8080}

        result = mgr._try_rdma_connect(instance)
        self.assertFalse(result)
        # connect_status should be cleaned up after timeout
        self.assertEqual(mgr.connect_status, {})

    def test_connect_exception_returns_false(self):
        """Returns False when exception occurs."""
        mgr = _make_manager()
        mgr.engine_worker_queue.put_connect_rdma_task.side_effect = RuntimeError("queue error")

        instance = {"host_ip": "10.0.0.2", "rdma_ports": [18515], "port": 8080}
        result = mgr._try_rdma_connect(instance)
        self.assertFalse(result)


class TestCheckRdmaConnection(unittest.TestCase):
    """Test _check_rdma_connection (line 396)."""

    def test_delegates_to_try_rdma_connect(self):
        """_check_rdma_connection calls _try_rdma_connect."""
        mgr = _make_manager()
        instance = {"host_ip": "10.0.0.2", "rdma_ports": [18515], "port": 8080}

        with patch.object(mgr, "_try_rdma_connect", return_value=True) as mock_try:
            result = mgr._check_rdma_connection(instance)

        self.assertTrue(result)
        mock_try.assert_called_once_with(instance)


class TestFetchDecodeInstancesInternal(unittest.TestCase):
    """Test _fetch_decode_instances_internal (lines 301-325)."""

    @patch("fastdeploy.engine.register_manager.requests.get")
    def test_fetch_success(self, mock_get):
        """Returns instances on successful response."""
        mgr = _make_manager()
        instances = [{"host_ip": "10.0.0.2", "port": 8080}]
        mock_get.return_value = MagicMock(ok=True, json=MagicMock(return_value=instances))

        result = mgr._fetch_decode_instances_internal()

        self.assertEqual(result, instances)
        mock_get.assert_called_once_with(
            "http://router:8080/decode_instances",
            params={"version": "v1.0"},
            timeout=5,
        )

    @patch("fastdeploy.engine.register_manager.requests.get")
    def test_fetch_non_ok_returns_empty(self, mock_get):
        """Returns empty list on non-OK response."""
        mgr = _make_manager()
        mock_get.return_value = MagicMock(ok=False, status_code=500)

        result = mgr._fetch_decode_instances_internal()
        self.assertEqual(result, [])

    @patch("fastdeploy.engine.register_manager.requests.get", side_effect=Exception("network error"))
    def test_fetch_exception_returns_empty(self, mock_get):
        """Returns empty list on exception."""
        mgr = _make_manager()
        result = mgr._fetch_decode_instances_internal()
        self.assertEqual(result, [])

    def test_fetch_no_router_returns_empty(self):
        """Returns empty list when router is None."""
        mgr = _make_manager(router_url=None)
        result = mgr._fetch_decode_instances_internal()
        self.assertEqual(result, [])


class TestEagerConnectIteration(unittest.TestCase):
    """Test _eager_connect_iteration (lines 227-289)."""

    def test_skips_when_not_registered(self):
        """Returns early when not registered."""
        mgr = _make_manager()
        mgr._is_registered = False

        with patch.object(mgr, "_fetch_decode_instances_internal") as mock_fetch:
            mgr._eager_connect_iteration()
            mock_fetch.assert_not_called()

    def test_skips_when_no_instances(self):
        """Returns early when no decode instances fetched."""
        mgr = _make_manager()
        mgr._is_registered = True

        with patch.object(mgr, "_fetch_decode_instances_internal", return_value=[]):
            mgr._eager_connect_iteration()

    def test_connects_new_healthy_rdma_instance(self):
        """Connects to new healthy instance with RDMA support."""
        mgr = _make_manager()
        mgr._is_registered = True
        instance = {"host_ip": "10.0.0.2", "port": 8080, "transfer_protocol": ["rdma"], "rdma_ports": [18515]}

        with (
            patch.object(mgr, "_fetch_decode_instances_internal", return_value=[instance]),
            patch.object(mgr, "_check_instance_health", return_value=True),
            patch.object(mgr, "_supports_rdma", return_value=True),
            patch.object(mgr, "_try_rdma_connect", return_value=True),
        ):
            mgr._eager_connect_iteration()

        self.assertIn(instance, mgr.connected_decodes)

    def test_skips_already_connected_instance(self):
        """Skips instance that's already in connected_decodes."""
        mgr = _make_manager()
        mgr._is_registered = True
        instance = {"host_ip": "10.0.0.2", "port": 8080, "transfer_protocol": ["rdma"], "rdma_ports": [18515]}
        mgr.connected_decodes = [instance]

        with (
            patch.object(mgr, "_fetch_decode_instances_internal", return_value=[instance]),
            patch.object(mgr, "_check_instance_health", return_value=True),
            patch.object(mgr, "_try_rdma_connect") as mock_connect,
            patch.object(mgr, "_check_rdma_connection", return_value=True),
        ):
            mgr._eager_connect_iteration()

        # _try_rdma_connect should NOT be called for new instances (already connected)
        # but _check_instance_health IS called for existing instance verification
        mock_connect.assert_not_called()

    def test_removes_unhealthy_existing_instance(self):
        """Removes existing instance that becomes unhealthy."""
        mgr = _make_manager()
        mgr._is_registered = True
        instance = {"host_ip": "10.0.0.2", "port": 8080, "transfer_protocol": ["rdma"], "rdma_ports": [18515]}
        mgr.connected_decodes = [instance]

        with (
            patch.object(mgr, "_fetch_decode_instances_internal", return_value=[instance]),
            patch.object(mgr, "_check_instance_health", return_value=False),
        ):
            mgr._eager_connect_iteration()

        self.assertNotIn(instance, mgr.connected_decodes)

    def test_removes_instance_with_lost_rdma(self):
        """Removes existing instance whose RDMA connection is lost."""
        mgr = _make_manager()
        mgr._is_registered = True
        instance = {"host_ip": "10.0.0.2", "port": 8080, "transfer_protocol": ["rdma"], "rdma_ports": [18515]}
        mgr.connected_decodes = [instance]

        # First call for existing instance health check (True), second for new instance check
        health_calls = [True]  # existing instance is healthy

        with (
            patch.object(mgr, "_fetch_decode_instances_internal", return_value=[instance]),
            patch.object(mgr, "_check_instance_health", side_effect=health_calls),
            patch.object(mgr, "_check_rdma_connection", return_value=False),
        ):
            mgr._eager_connect_iteration()

        self.assertNotIn(instance, mgr.connected_decodes)

    def test_skips_instance_without_rdma(self):
        """Skips new instance that doesn't support RDMA."""
        mgr = _make_manager()
        mgr._is_registered = True
        instance = {"host_ip": "10.0.0.2", "port": 8080, "transfer_protocol": ["ipc"], "rdma_ports": []}

        with (
            patch.object(mgr, "_fetch_decode_instances_internal", return_value=[instance]),
            patch.object(mgr, "_check_instance_health", return_value=True),
            patch.object(mgr, "_supports_rdma", return_value=False),
            patch.object(mgr, "_try_rdma_connect") as mock_connect,
        ):
            mgr._eager_connect_iteration()

        mock_connect.assert_not_called()
        self.assertEqual(mgr.connected_decodes, [])

    def test_handles_exception_in_instance_processing(self):
        """Handles exception when processing a single instance."""
        mgr = _make_manager()
        mgr._is_registered = True
        instance = {"host_ip": "10.0.0.2", "port": 8080}

        with (
            patch.object(mgr, "_fetch_decode_instances_internal", return_value=[instance]),
            patch.object(mgr, "_check_instance_health", side_effect=RuntimeError("unexpected")),
        ):
            # Should not raise
            mgr._eager_connect_iteration()


class TestRegisterToRouter(unittest.TestCase):
    """Test _register_to_router (lines 117-132)."""

    def test_skips_when_no_router(self):
        """Does nothing when router is None."""
        mgr = _make_manager(router_url=None)
        # Should not start any thread - just return
        with patch("threading.Thread") as mock_thread:
            mgr._register_to_router()
            mock_thread.assert_not_called()

    @patch("fastdeploy.engine.register_manager.check_service_health", return_value=True)
    @patch("fastdeploy.engine.register_manager.requests.post")
    def test_register_thread_starts(self, mock_post, mock_health):
        """Starts a daemon thread for registration."""
        mgr = _make_manager()
        with patch("threading.Thread") as mock_thread:
            mock_thread_instance = MagicMock()
            mock_thread.return_value = mock_thread_instance
            mgr._register_to_router()
            mock_thread.assert_called_once()
            self.assertTrue(mock_thread.call_args[1].get("daemon", False))
            mock_thread_instance.start.assert_called_once()


class TestStartEagerConnectLoop(unittest.TestCase):
    """Test _start_eager_connect_loop (lines 162-190)."""

    def test_skips_when_not_enabled(self):
        """Does not start threads when eager connect not enabled."""
        mgr = _make_manager()
        with (
            patch.object(mgr, "_should_enable_eager_connect", return_value=False),
            patch("threading.Thread") as mock_thread,
        ):
            mgr._start_eager_connect_loop()
            mock_thread.assert_not_called()

    def test_starts_two_threads_when_enabled(self):
        """Starts eager connect loop + response loop threads."""
        mgr = _make_manager()
        with (
            patch.object(mgr, "_should_enable_eager_connect", return_value=True),
            patch("threading.Thread") as mock_thread,
        ):
            mock_thread.return_value = MagicMock()
            mgr._start_eager_connect_loop()
            # Should create 2 threads
            self.assertEqual(mock_thread.call_count, 2)


class TestStart(unittest.TestCase):
    """Test start() method (lines 78-79)."""

    def test_start_calls_register_and_eager_connect(self):
        """start() calls _register_to_router and _start_eager_connect_loop."""
        mgr = _make_manager()
        with (
            patch.object(mgr, "_register_to_router") as mock_reg,
            patch.object(mgr, "_start_eager_connect_loop") as mock_eager,
        ):
            mgr.start()
            mock_reg.assert_called_once()
            mock_eager.assert_called_once()


class TestRegisterLoopBody(unittest.TestCase):
    """Test the inner _register() loop body (lines 106-139)."""

    @patch("fastdeploy.engine.register_manager.time.sleep", side_effect=StopIteration)
    @patch("fastdeploy.engine.register_manager.check_service_health", return_value=True)
    @patch("fastdeploy.engine.register_manager.requests.post")
    def test_register_success_sets_is_registered(self, mock_post, mock_health, mock_sleep):
        """Lines 117-130: successful registration sets _is_registered=True."""
        mgr = _make_manager()
        mock_post.return_value = MagicMock(ok=True)

        # Capture the target function from Thread
        captured_target = None

        def capture_thread(*args, **kwargs):
            nonlocal captured_target
            captured_target = kwargs.get("target")
            return MagicMock()

        with patch("threading.Thread", side_effect=capture_thread):
            mgr._register_to_router()

        # Run one iteration of the register loop (StopIteration breaks the while True)
        self.assertIsNotNone(captured_target)
        with self.assertRaises(StopIteration):
            captured_target()

        self.assertTrue(mgr._is_registered)
        mock_post.assert_called_once()
        # Verify register_info was updated
        self.assertIn("is_paused", mgr.cfg.register_info)
        self.assertIn("version", mgr.cfg.register_info)

    @patch("fastdeploy.engine.register_manager.time.sleep", side_effect=StopIteration)
    @patch("fastdeploy.engine.register_manager.check_service_health", return_value=True)
    @patch("fastdeploy.engine.register_manager.requests.post")
    def test_register_failure_does_not_set_registered(self, mock_post, mock_health, mock_sleep):
        """Lines 131-135: failed registration logs error, doesn't set registered."""
        mgr = _make_manager()
        mock_post.return_value = MagicMock(ok=False, status_code=500, text="error")

        captured_target = None

        def capture_thread(*args, **kwargs):
            nonlocal captured_target
            captured_target = kwargs.get("target")
            return MagicMock()

        with patch("threading.Thread", side_effect=capture_thread):
            mgr._register_to_router()

        with self.assertRaises(StopIteration):
            captured_target()

        self.assertFalse(mgr._is_registered)

    @patch("fastdeploy.engine.register_manager.time.sleep", side_effect=[None, StopIteration])
    @patch("fastdeploy.engine.register_manager.check_service_health", return_value=False)
    def test_register_waits_for_health(self, mock_health, mock_sleep):
        """Lines 111-114: waits when API server is not healthy."""
        mgr = _make_manager()

        captured_target = None

        def capture_thread(*args, **kwargs):
            nonlocal captured_target
            captured_target = kwargs.get("target")
            return MagicMock()

        with patch("threading.Thread", side_effect=capture_thread):
            mgr._register_to_router()

        with self.assertRaises(StopIteration):
            captured_target()

        # Should not have registered since health check failed
        self.assertFalse(mgr._is_registered)

    @patch("fastdeploy.engine.register_manager.time.sleep", side_effect=StopIteration)
    @patch("fastdeploy.engine.register_manager.check_service_health", return_value=True)
    @patch("fastdeploy.engine.register_manager.requests.post", side_effect=Exception("connection refused"))
    def test_register_exception_handled(self, mock_post, mock_health, mock_sleep):
        """Lines 136-137: exception in registration is handled."""
        mgr = _make_manager()

        captured_target = None

        def capture_thread(*args, **kwargs):
            nonlocal captured_target
            captured_target = kwargs.get("target")
            return MagicMock()

        with patch("threading.Thread", side_effect=capture_thread):
            mgr._register_to_router()

        with self.assertRaises(StopIteration):
            captured_target()

        self.assertFalse(mgr._is_registered)


class TestEagerConnectLoopBody(unittest.TestCase):
    """Test the inner eager connect loop bodies (lines 162-186)."""

    @patch("fastdeploy.engine.register_manager.time.sleep", side_effect=StopIteration)
    def test_eager_connect_loop_calls_iteration(self, mock_sleep):
        """Lines 163-168: loop calls _eager_connect_iteration."""
        mgr = _make_manager()
        captured_targets = []

        def capture_thread(*args, **kwargs):
            captured_targets.append(kwargs.get("target"))
            return MagicMock()

        with (
            patch.object(mgr, "_should_enable_eager_connect", return_value=True),
            patch("threading.Thread", side_effect=capture_thread),
        ):
            mgr._start_eager_connect_loop()

        # First thread is the eager connect loop
        self.assertEqual(len(captured_targets), 2)
        with patch.object(mgr, "_eager_connect_iteration") as mock_iter:
            with self.assertRaises(StopIteration):
                captured_targets[0]()
            mock_iter.assert_called_once()

    @patch("fastdeploy.engine.register_manager.time.sleep", side_effect=StopIteration)
    def test_eager_connect_loop_handles_exception(self, mock_sleep):
        """Lines 166-167: exception in iteration is caught."""
        mgr = _make_manager()
        captured_targets = []

        def capture_thread(*args, **kwargs):
            captured_targets.append(kwargs.get("target"))
            return MagicMock()

        with (
            patch.object(mgr, "_should_enable_eager_connect", return_value=True),
            patch("threading.Thread", side_effect=capture_thread),
            patch.object(mgr, "_eager_connect_iteration", side_effect=RuntimeError("test")),
        ):
            mgr._start_eager_connect_loop()

        # Should not raise despite iteration error
        with self.assertRaises(StopIteration):
            captured_targets[0]()

    @patch("fastdeploy.engine.register_manager.time.sleep", side_effect=StopIteration)
    def test_response_loop_processes_response(self, mock_sleep):
        """Lines 175-186: response loop processes task responses."""
        mgr = _make_manager()
        mgr.engine_worker_queue.get_connect_rdma_task_response.return_value = {
            "task_id": "test-task-123",
            "success": True,
        }
        captured_targets = []

        def capture_thread(*args, **kwargs):
            captured_targets.append(kwargs.get("target"))
            return MagicMock()

        with (
            patch.object(mgr, "_should_enable_eager_connect", return_value=True),
            patch("threading.Thread", side_effect=capture_thread),
        ):
            mgr._start_eager_connect_loop()

        # Second thread is the response loop
        with self.assertRaises(StopIteration):
            captured_targets[1]()

        self.assertEqual(mgr.connect_status["test-task-123"], True)

    @patch("fastdeploy.engine.register_manager.time.sleep", side_effect=StopIteration)
    def test_response_loop_handles_exception(self, mock_sleep):
        """Lines 184-185: exception in response loop is caught."""
        mgr = _make_manager()
        mgr.engine_worker_queue.get_connect_rdma_task_response.side_effect = RuntimeError("queue error")
        captured_targets = []

        def capture_thread(*args, **kwargs):
            captured_targets.append(kwargs.get("target"))
            return MagicMock()

        with (
            patch.object(mgr, "_should_enable_eager_connect", return_value=True),
            patch("threading.Thread", side_effect=capture_thread),
        ):
            mgr._start_eager_connect_loop()

        # Should not raise
        with self.assertRaises(StopIteration):
            captured_targets[1]()


if __name__ == "__main__":
    unittest.main()
