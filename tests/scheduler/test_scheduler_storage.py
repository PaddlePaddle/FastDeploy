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

from fastdeploy.scheduler.storage import AdaptedRedis


class TestAdaptedRedisInit(unittest.TestCase):
    """Test AdaptedRedis.__init__."""

    @patch.object(AdaptedRedis, "_register_script")
    @patch.object(AdaptedRedis, "_parse_version")
    @patch("redis.Redis.__init__", return_value=None)
    def test_init(self, mock_redis_init, mock_parse, mock_register):
        """__init__ calls super().__init__, _parse_version, and _register_script."""
        client = AdaptedRedis(host="localhost", port=6379)

        mock_redis_init.assert_called_once_with(host="localhost", port=6379)
        mock_parse.assert_called_once()
        mock_register.assert_called_once()
        self.assertFalse(client._old_version)


class TestParseVersion(unittest.TestCase):
    """Test AdaptedRedis._parse_version."""

    def _make_client(self):
        """Create an AdaptedRedis instance without calling __init__."""
        client = AdaptedRedis.__new__(AdaptedRedis)
        client._old_version = False
        return client

    @patch("redis.Redis.info")
    def test_new_version(self, mock_info):
        """_parse_version sets _old_version=False for version > 6.2.28."""
        client = self._make_client()
        mock_info.return_value = {"redis_version": "7.0.5"}

        client._parse_version()

        self.assertFalse(client._old_version)
        self.assertEqual(client.version, "7.0.5")

    @patch("redis.Redis.info")
    def test_old_version(self, mock_info):
        """_parse_version sets _old_version=True for version <= 6.2.28."""
        client = self._make_client()
        mock_info.return_value = {"redis_version": "6.2.28"}

        client._parse_version()

        self.assertTrue(client._old_version)
        self.assertEqual(client.version, "6.2.28")

    @patch("redis.Redis.info")
    def test_older_version(self, mock_info):
        """_parse_version sets _old_version=True for version < 6.2.28."""
        client = self._make_client()
        mock_info.return_value = {"redis_version": "5.0.7"}

        client._parse_version()

        self.assertTrue(client._old_version)
        self.assertEqual(client.version, "5.0.7")

    @patch("redis.Redis.info")
    def test_invalid_version_string(self, mock_info):
        """_parse_version defaults to '0.0.0' for unparseable version."""
        client = self._make_client()
        mock_info.return_value = {"redis_version": "invalid-version"}

        client._parse_version()

        self.assertTrue(client._old_version)
        self.assertEqual(client.version, "0.0.0")

    @patch("redis.Redis.info")
    def test_version_with_suffix(self, mock_info):
        """_parse_version extracts numeric prefix from version with suffix."""
        client = self._make_client()
        mock_info.return_value = {"redis_version": "7.2.1-rc1-extra"}

        client._parse_version()

        self.assertFalse(client._old_version)
        self.assertEqual(client.version, "7.2.1")


class TestRegisterScript(unittest.TestCase):
    """Test AdaptedRedis._register_script."""

    def _make_client(self):
        client = AdaptedRedis.__new__(AdaptedRedis)
        client._old_version = False
        return client

    @patch("redis.Redis.register_script")
    def test_old_version_registers_lpop(self, mock_register):
        """_register_script registers lpop script for old versions."""
        client = self._make_client()
        client._old_version = True
        mock_register.return_value = MagicMock()

        client._register_script()

        # Should register both LUA_LPOP and LUA_ZINCRBY
        self.assertEqual(mock_register.call_count, 2)

    @patch("redis.Redis.register_script")
    def test_new_version_no_lpop(self, mock_register):
        """_register_script only registers zincrby for new versions."""
        client = self._make_client()
        client._old_version = False
        mock_register.return_value = MagicMock()

        client._register_script()

        # Should register only LUA_ZINCRBY
        self.assertEqual(mock_register.call_count, 1)


class TestRpush(unittest.TestCase):
    """Test AdaptedRedis.rpush."""

    def _make_client(self):
        client = AdaptedRedis.__new__(AdaptedRedis)
        client._old_version = False
        return client

    @patch("redis.Redis.rpush", return_value=3)
    def test_rpush_no_ttl(self, mock_rpush):
        """rpush without ttl calls super().rpush directly."""
        client = self._make_client()

        result = client.rpush("mylist", "a", "b", "c")

        mock_rpush.assert_called_once_with("mylist", "a", "b", "c")
        self.assertEqual(result, 3)

    @patch("redis.Redis.pipeline")
    def test_rpush_with_ttl(self, mock_pipeline):
        """rpush with ttl uses pipeline with expire."""
        client = self._make_client()

        mock_pipe = MagicMock()
        mock_pipe.__enter__ = MagicMock(return_value=mock_pipe)
        mock_pipe.__exit__ = MagicMock(return_value=False)
        mock_pipe.execute.return_value = [5, True]
        mock_pipeline.return_value = mock_pipe

        result = client.rpush("mylist", "a", "b", ttl=60)

        mock_pipe.multi.assert_called_once()
        mock_pipe.rpush.assert_called_once_with("mylist", "a", "b")
        mock_pipe.expire.assert_called_once_with("mylist", 60)
        self.assertEqual(result, 5)


class TestZincrby(unittest.TestCase):
    """Test AdaptedRedis.zincrby."""

    def _make_client(self):
        client = AdaptedRedis.__new__(AdaptedRedis)
        client._old_version = False
        client._zincrby = MagicMock(return_value=5.0)
        return client

    @patch("redis.Redis.zincrby", return_value=3.0)
    def test_zincrby_no_ttl_no_rem(self, mock_zincrby):
        """zincrby without ttl or rem_amount calls super().zincrby."""
        client = self._make_client()

        result = client.zincrby("myset", 1.5, "member")

        mock_zincrby.assert_called_once_with("myset", "1.5", "member")
        self.assertEqual(result, 3.0)

    def test_zincrby_no_ttl_with_rem(self):
        """zincrby without ttl but with rem_amount uses lua script."""
        client = self._make_client()

        result = client.zincrby("myset", 2.0, "member", rem_amount=10.0)

        client._zincrby.assert_called_once_with(keys=["myset"], args=["2.0", "member", "10.0"])
        self.assertEqual(result, 5.0)

    @patch("redis.Redis.pipeline")
    def test_zincrby_with_ttl_no_rem(self, mock_pipeline):
        """zincrby with ttl and no rem_amount uses pipeline with pipe.zincrby."""
        client = self._make_client()

        mock_pipe = MagicMock()
        mock_pipe.__enter__ = MagicMock(return_value=mock_pipe)
        mock_pipe.__exit__ = MagicMock(return_value=False)
        mock_pipe.execute.return_value = [7.0, True]
        mock_pipeline.return_value = mock_pipe

        result = client.zincrby("myset", 1.0, "member", ttl=120)

        mock_pipe.multi.assert_called_once()
        mock_pipe.zincrby.assert_called_once_with("myset", "1.0", "member")
        mock_pipe.expire.assert_called_once_with("myset", 120)
        self.assertEqual(result, 7.0)

    @patch("redis.Redis.pipeline")
    def test_zincrby_with_ttl_and_rem(self, mock_pipeline):
        """zincrby with ttl and rem_amount uses pipeline with lua script."""
        client = self._make_client()

        mock_pipe = MagicMock()
        mock_pipe.__enter__ = MagicMock(return_value=mock_pipe)
        mock_pipe.__exit__ = MagicMock(return_value=False)
        mock_pipe.execute.return_value = [8.0, True]
        mock_pipeline.return_value = mock_pipe

        result = client.zincrby("myset", 3.0, "member", rem_amount=5.0, ttl=60)

        mock_pipe.multi.assert_called_once()
        client._zincrby.assert_called_once_with(keys=["myset"], args=["3.0", "member", "5.0"], client=mock_pipe)
        mock_pipe.expire.assert_called_once_with("myset", 60)
        self.assertEqual(result, 8.0)


class TestLpop(unittest.TestCase):
    """Test AdaptedRedis.lpop."""

    def _make_client(self, old_version=False):
        client = AdaptedRedis.__new__(AdaptedRedis)
        client._old_version = old_version
        client._lpop = MagicMock(return_value=["a", "b"])
        return client

    @patch("redis.Redis.lpop", return_value="value")
    def test_lpop_no_ttl_new_version(self, mock_lpop):
        """lpop without ttl on new version calls super().lpop."""
        client = self._make_client(old_version=False)

        result = client.lpop("mylist", 3)

        mock_lpop.assert_called_once_with("mylist", 3)
        self.assertEqual(result, "value")

    def test_lpop_no_ttl_old_version_with_count(self):
        """lpop without ttl on old version with count uses lua script."""
        client = self._make_client(old_version=True)

        result = client.lpop("mylist", count=5)

        client._lpop.assert_called_once_with(keys=["mylist"], args=[5])
        self.assertEqual(result, ["a", "b"])

    @patch("redis.Redis.lpop", return_value="single")
    def test_lpop_no_ttl_old_version_no_count(self, mock_lpop):
        """lpop without ttl on old version without count calls super().lpop."""
        client = self._make_client(old_version=True)

        result = client.lpop("mylist", count=None)

        mock_lpop.assert_called_once_with("mylist", None)
        self.assertEqual(result, "single")

    @patch("redis.Redis.pipeline")
    def test_lpop_with_ttl_new_version(self, mock_pipeline):
        """lpop with ttl on new version uses pipeline with pipe.lpop."""
        client = self._make_client(old_version=False)

        mock_pipe = MagicMock()
        mock_pipe.__enter__ = MagicMock(return_value=mock_pipe)
        mock_pipe.__exit__ = MagicMock(return_value=False)
        mock_pipe.execute.return_value = [["x", "y"], True]
        mock_pipeline.return_value = mock_pipe

        result = client.lpop("mylist", count=2, ttl=30)

        mock_pipe.multi.assert_called_once()
        mock_pipe.lpop.assert_called_once_with("mylist", 2)
        mock_pipe.expire.assert_called_once_with("mylist", 30)
        self.assertEqual(result, ["x", "y"])

    @patch("redis.Redis.pipeline")
    def test_lpop_with_ttl_old_version_with_count(self, mock_pipeline):
        """lpop with ttl on old version with count uses lua script in pipeline."""
        client = self._make_client(old_version=True)

        mock_pipe = MagicMock()
        mock_pipe.__enter__ = MagicMock(return_value=mock_pipe)
        mock_pipe.__exit__ = MagicMock(return_value=False)
        mock_pipe.execute.return_value = [["a"], True]
        mock_pipeline.return_value = mock_pipe

        result = client.lpop("mylist", count=3, ttl=45)

        mock_pipe.multi.assert_called_once()
        client._lpop.assert_called_once_with(keys=["mylist"], args=[3], client=mock_pipe)
        mock_pipe.expire.assert_called_once_with("mylist", 45)
        self.assertEqual(result, ["a"])


class TestBlpop(unittest.TestCase):
    """Test AdaptedRedis.blpop."""

    def _make_client(self, old_version=False):
        client = AdaptedRedis.__new__(AdaptedRedis)
        client._old_version = old_version
        return client

    @patch("redis.Redis.blpop", return_value=("key", "val"))
    def test_blpop_new_version_normal_timeout(self, mock_blpop):
        """blpop on new version with normal timeout passes through."""
        client = self._make_client(old_version=False)

        result = client.blpop(["key1"], timeout=5)

        mock_blpop.assert_called_once_with(keys=["key1"], timeout=5)
        self.assertEqual(result, ("key", "val"))

    @patch("redis.Redis.blpop", return_value=("key", "val"))
    def test_blpop_new_version_small_timeout(self, mock_blpop):
        """blpop on new version clamps timeout >= 0.01."""
        client = self._make_client(old_version=False)

        client.blpop(["key1"], timeout=0.001)

        mock_blpop.assert_called_once_with(keys=["key1"], timeout=0.01)

    @patch("redis.Redis.blpop", return_value=("key", "val"))
    def test_blpop_new_version_zero_timeout(self, mock_blpop):
        """blpop on new version with zero timeout passes through unchanged."""
        client = self._make_client(old_version=False)

        client.blpop(["key1"], timeout=0)

        mock_blpop.assert_called_once_with(keys=["key1"], timeout=0)

    @patch("redis.Redis.blpop", return_value=("key", "val"))
    def test_blpop_old_version_normal_timeout(self, mock_blpop):
        """blpop on old version converts timeout to int."""
        client = self._make_client(old_version=True)

        client.blpop(["key1"], timeout=5)

        mock_blpop.assert_called_once_with(keys=["key1"], timeout=5)

    @patch("redis.Redis.blpop", return_value=("key", "val"))
    def test_blpop_old_version_small_timeout(self, mock_blpop):
        """blpop on old version clamps small timeout to 1."""
        client = self._make_client(old_version=True)

        client.blpop(["key1"], timeout=0.5)

        mock_blpop.assert_called_once_with(keys=["key1"], timeout=1)

    @patch("redis.Redis.blpop", return_value=("key", "val"))
    def test_blpop_old_version_zero_timeout(self, mock_blpop):
        """blpop on old version with zero timeout passes through."""
        client = self._make_client(old_version=True)

        client.blpop(["key1"], timeout=0)

        mock_blpop.assert_called_once_with(keys=["key1"], timeout=0)


if __name__ == "__main__":
    unittest.main()
