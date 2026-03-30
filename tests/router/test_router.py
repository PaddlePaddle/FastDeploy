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
Tests for Router class.

Why mock:
  - register_instance calls check_service_health_async which does real HTTP.
    We mock it at the network boundary to test Router's registration and selection logic.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastdeploy.router.router import Router, RouterArgs


def _make_args(**kwargs):
    defaults = {"host": "0.0.0.0", "port": 9000, "splitwise": False, "request_timeout_secs": 30}
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _make_instance_dict(host_ip="10.0.0.1", port=8080, role="mixed", **kwargs):
    d = {
        "host_ip": host_ip,
        "port": port,
        "role": role,
    }
    d.update(kwargs)
    return d


class TestRouterArgs(unittest.TestCase):
    def test_defaults(self):
        args = RouterArgs()
        self.assertEqual(args.host, "0.0.0.0")
        self.assertEqual(args.port, 9000)
        self.assertFalse(args.splitwise)
        self.assertEqual(args.request_timeout_secs, 1800)


class TestRouterInit(unittest.TestCase):
    def test_init(self):
        args = _make_args()
        router = Router(args)
        self.assertEqual(router.host, "0.0.0.0")
        self.assertEqual(router.port, 9000)
        self.assertFalse(router.splitwise)
        self.assertEqual(router.mixed_servers, [])
        self.assertEqual(router.prefill_servers, [])
        self.assertEqual(router.decode_servers, [])


class TestRouterRegistration(unittest.IsolatedAsyncioTestCase):
    @patch("fastdeploy.router.router.check_service_health_async", new_callable=AsyncMock, return_value=True)
    async def test_register_mixed_instance(self, mock_health):
        router = Router(_make_args(splitwise=False))
        inst_dict = _make_instance_dict(role="mixed")
        await router.register_instance(inst_dict)
        self.assertEqual(len(router.mixed_servers), 1)

    @patch("fastdeploy.router.router.check_service_health_async", new_callable=AsyncMock, return_value=True)
    async def test_register_splitwise_instances(self, mock_health):
        router = Router(_make_args(splitwise=True))

        await router.register_instance(_make_instance_dict(host_ip="10.0.0.1", role="prefill"))
        await router.register_instance(_make_instance_dict(host_ip="10.0.0.2", role="decode"))

        self.assertEqual(len(router.prefill_servers), 1)
        self.assertEqual(len(router.decode_servers), 1)

    @patch("fastdeploy.router.router.check_service_health_async", new_callable=AsyncMock, return_value=True)
    async def test_register_invalid_role_raises(self, mock_health):
        """Splitwise mode should reject mixed instances."""
        router = Router(_make_args(splitwise=True))
        with self.assertRaises(ValueError):
            await router.register_instance(_make_instance_dict(role="mixed"))

    @patch("fastdeploy.router.router.check_service_health_async", new_callable=AsyncMock, return_value=False)
    async def test_register_unhealthy_instance_raises(self, mock_health):
        router = Router(_make_args(splitwise=False))
        with self.assertRaises(RuntimeError):
            await router.register_instance(_make_instance_dict(role="mixed"))


class TestRouterSelection(unittest.IsolatedAsyncioTestCase):
    async def test_select_mixed_no_servers_raises(self):
        router = Router(_make_args(splitwise=False))
        with self.assertRaises(RuntimeError):
            await router.select_mixed()

    async def test_select_pd_no_prefill_raises(self):
        router = Router(_make_args(splitwise=True))
        with self.assertRaises(RuntimeError):
            await router.select_pd()

    async def test_select_pd_no_decode_raises(self):
        """Test select_pd raises when no decode servers available (line 152)."""
        router = Router(_make_args(splitwise=True))
        # Manually add a prefill server without going through health check
        router.prefill_servers.append(_make_instance_dict(role="prefill"))
        with self.assertRaises(RuntimeError) as ctx:
            await router.select_pd()
        self.assertIn("No decode servers available", str(ctx.exception))

    @patch("fastdeploy.router.router.check_service_health_async", new_callable=AsyncMock, return_value=True)
    async def test_select_mixed_returns_instance(self, mock_health):
        router = Router(_make_args(splitwise=False))
        await router.register_instance(_make_instance_dict(role="mixed"))
        inst = await router.select_mixed()
        self.assertIsNotNone(inst)

    @patch("fastdeploy.router.router.check_service_health_async", new_callable=AsyncMock, return_value=True)
    async def test_select_pd_returns_pair(self, mock_health):
        router = Router(_make_args(splitwise=True))
        await router.register_instance(_make_instance_dict(host_ip="10.0.0.1", role="prefill"))
        await router.register_instance(_make_instance_dict(host_ip="10.0.0.2", role="decode"))
        prefill, decode = await router.select_pd()
        self.assertIsNotNone(prefill)
        self.assertIsNotNone(decode)


class TestRouterRegisteredNumber(unittest.IsolatedAsyncioTestCase):
    @patch("fastdeploy.router.router.check_service_health_async", new_callable=AsyncMock, return_value=True)
    async def test_registered_number(self, mock_health):
        router = Router(_make_args(splitwise=False))
        await router.register_instance(_make_instance_dict(role="mixed"))
        result = await router.registered_number()
        self.assertEqual(result["mixed"], 1)
        self.assertEqual(result["prefill"], 0)
        self.assertEqual(result["decode"], 0)


if __name__ == "__main__":
    unittest.main()
