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

import unittest
from unittest.mock import patch

from fastdeploy.cache_manager.v1.storage.attnstore.connector import (
    AttnStoreConnector,
    AttnStoreScheduler,
)


class TestAttnStoreSchedulerInit(unittest.TestCase):
    """Test AttnStoreScheduler initialization."""

    def test_default_config(self):
        """Init with no config uses empty dict."""
        scheduler = AttnStoreScheduler()
        self.assertEqual(scheduler.config, {})
        self.assertFalse(scheduler.is_connected())

    def test_custom_config(self):
        """Init with custom config stores it."""
        cfg = {"store_path": "/tmp/attn", "cache_size": 1024}
        scheduler = AttnStoreScheduler(config=cfg)
        self.assertEqual(scheduler.config, cfg)


class TestAttnStoreSchedulerConnect(unittest.TestCase):
    """Test AttnStoreScheduler connect/disconnect."""

    def setUp(self):
        self.scheduler = AttnStoreScheduler()

    def test_connect_returns_true(self):
        """connect() returns True and sets connected state."""
        result = self.scheduler.connect()
        self.assertTrue(result)
        self.assertTrue(self.scheduler.is_connected())

    def test_connect_exception_returns_false(self):
        """connect() returns False when exception occurs in try block."""
        scheduler = AttnStoreScheduler()

        # Make __setattr__ raise to trigger the except branch
        def raising_setattr(obj, name, value):
            if name == "_connected" and value is True:
                raise RuntimeError("simulated")
            object.__setattr__(obj, name, value)

        with patch.object(AttnStoreScheduler, "__setattr__", raising_setattr):
            result = scheduler.connect()
        self.assertFalse(result)
        self.assertFalse(scheduler.is_connected())

    def test_disconnect(self):
        """disconnect() sets connected to False."""
        self.scheduler.connect()
        self.scheduler.disconnect()
        self.assertFalse(self.scheduler.is_connected())


class TestAttnStoreSchedulerOperations(unittest.TestCase):
    """Test AttnStoreScheduler query operations."""

    def setUp(self):
        self.scheduler = AttnStoreScheduler()

    def test_exists_when_disconnected(self):
        """exists() returns False when not connected."""
        self.assertFalse(self.scheduler.exists("key1"))

    def test_exists_when_connected(self):
        """exists() returns False (placeholder) when connected."""
        self.scheduler.connect()
        self.assertFalse(self.scheduler.exists("key1"))

    def test_query_when_disconnected(self):
        """query() returns all False when not connected."""
        keys = ["a", "b", "c"]
        result = self.scheduler.query(keys)
        self.assertEqual(result, {"a": False, "b": False, "c": False})

    def test_query_when_connected(self):
        """query() returns all False (placeholder) when connected."""
        self.scheduler.connect()
        keys = ["x", "y"]
        result = self.scheduler.query(keys)
        self.assertEqual(result, {"x": False, "y": False})

    def test_get_metadata_when_disconnected(self):
        """get_metadata() returns None when not connected."""
        self.assertIsNone(self.scheduler.get_metadata("key1"))

    def test_get_metadata_when_connected(self):
        """get_metadata() returns None (placeholder) when connected."""
        self.scheduler.connect()
        self.assertIsNone(self.scheduler.get_metadata("key1"))

    def test_list_keys_when_disconnected(self):
        """list_keys() returns empty list when not connected."""
        self.assertEqual(self.scheduler.list_keys(), [])

    def test_list_keys_when_connected(self):
        """list_keys() returns empty list (placeholder) when connected."""
        self.scheduler.connect()
        self.assertEqual(self.scheduler.list_keys("prefix"), [])

    def test_get_stats(self):
        """get_stats() returns connection status and config."""
        stats = self.scheduler.get_stats()
        self.assertFalse(stats["connected"])
        self.assertEqual(stats["config"], {})


class TestAttnStoreConnectorInit(unittest.TestCase):
    """Test AttnStoreConnector initialization."""

    def test_default_config(self):
        """Init with no config uses empty dict."""
        connector = AttnStoreConnector()
        self.assertEqual(connector.config, {})
        self.assertFalse(connector.is_connected())

    def test_custom_config(self):
        """Init with custom config stores it."""
        cfg = {"store_path": "/tmp/attn", "transfer_threads": 4}
        connector = AttnStoreConnector(config=cfg)
        self.assertEqual(connector.config, cfg)


class TestAttnStoreConnectorConnect(unittest.TestCase):
    """Test AttnStoreConnector connect/disconnect."""

    def setUp(self):
        self.connector = AttnStoreConnector()

    def test_connect_returns_true(self):
        """connect() returns True and sets connected state."""
        result = self.connector.connect()
        self.assertTrue(result)
        self.assertTrue(self.connector.is_connected())

    def test_connect_exception_returns_false(self):
        """connect() returns False when exception occurs in try block."""
        connector = AttnStoreConnector()

        def raising_setattr(obj, name, value):
            if name == "_connected" and value is True:
                raise RuntimeError("simulated")
            object.__setattr__(obj, name, value)

        with patch.object(AttnStoreConnector, "__setattr__", raising_setattr):
            result = connector.connect()
        self.assertFalse(result)
        self.assertFalse(connector.is_connected())

    def test_disconnect(self):
        """disconnect() sets connected to False."""
        self.connector.connect()
        self.connector.disconnect()
        self.assertFalse(self.connector.is_connected())


class TestAttnStoreConnectorOperations(unittest.TestCase):
    """Test AttnStoreConnector data transfer operations."""

    def setUp(self):
        self.connector = AttnStoreConnector()

    def test_get_when_disconnected(self):
        """get() returns False when not connected."""
        self.assertFalse(self.connector.get("key1", bytearray(10)))

    def test_get_when_connected(self):
        """get() returns False (placeholder) when connected."""
        self.connector.connect()
        self.assertFalse(self.connector.get("key1", bytearray(10)))

    def test_set_when_disconnected(self):
        """set() returns False when not connected."""
        self.assertFalse(self.connector.set("key1", b"data", 4))

    def test_set_when_connected(self):
        """set() returns False (placeholder) when connected."""
        self.connector.connect()
        self.assertFalse(self.connector.set("key1", b"data", 4))

    def test_delete_when_disconnected(self):
        """delete() returns False when not connected."""
        self.assertFalse(self.connector.delete("key1"))

    def test_delete_when_connected(self):
        """delete() returns False (placeholder) when connected."""
        self.connector.connect()
        self.assertFalse(self.connector.delete("key1"))

    def test_clear_when_disconnected(self):
        """clear() returns 0 when not connected."""
        self.assertEqual(self.connector.clear(), 0)

    def test_clear_when_connected(self):
        """clear() returns 0 (placeholder) when connected."""
        self.connector.connect()
        self.assertEqual(self.connector.clear("prefix"), 0)

    def test_get_stats(self):
        """get_stats() returns connection status and config."""
        stats = self.connector.get_stats()
        self.assertFalse(stats["connected"])
        self.assertEqual(stats["config"], {})


if __name__ == "__main__":
    unittest.main()
