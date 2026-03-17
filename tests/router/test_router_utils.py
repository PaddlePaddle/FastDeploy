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
Tests for router utils - InstanceInfo class.
"""

import unittest

from fastdeploy.router.utils import InstanceInfo, InstanceRole


class TestInstanceInfoFromDict(unittest.TestCase):
    """Tests for InstanceInfo.from_dict method."""

    def test_from_dict_success(self):
        """Test creating InstanceInfo from dict with all required fields."""
        info_dict = {
            "role": "mixed",
            "host_ip": "10.0.0.1",
            "port": 8080,
        }
        info = InstanceInfo.from_dict(info_dict)
        self.assertEqual(info.role, InstanceRole.MIXED)
        self.assertEqual(info.host_ip, "10.0.0.1")
        self.assertEqual(info.port, "8080")

    def test_from_dict_missing_required_field_raises_keyerror(self):
        """Test from_dict raises KeyError when required field is missing (line 60)."""
        # Missing 'host_ip' which is a required field
        info_dict = {
            "role": "mixed",
            "port": 8080,
        }
        with self.assertRaises(KeyError) as ctx:
            InstanceInfo.from_dict(info_dict)
        self.assertIn("Missing required field", str(ctx.exception))
        self.assertIn("host_ip", str(ctx.exception))

    def test_from_dict_missing_role_raises_keyerror(self):
        """Test from_dict raises KeyError when role is missing."""
        info_dict = {
            "host_ip": "10.0.0.1",
            "port": 8080,
        }
        with self.assertRaises(KeyError) as ctx:
            InstanceInfo.from_dict(info_dict)
        self.assertIn("Missing required field", str(ctx.exception))
        self.assertIn("role", str(ctx.exception))

    def test_from_dict_with_optional_fields(self):
        """Test from_dict with optional fields uses defaults."""
        info_dict = {
            "role": InstanceRole.PREFILL,
            "host_ip": "10.0.0.2",
            "port": 9090,
            "metrics_port": 9091,
            "transfer_protocol": ["ipc"],
        }
        info = InstanceInfo.from_dict(info_dict)
        self.assertEqual(info.role, InstanceRole.PREFILL)
        self.assertEqual(info.metrics_port, "9091")
        self.assertEqual(info.transfer_protocol, ["ipc"])
        # Check defaults
        self.assertEqual(info.connector_port, "0")
        self.assertEqual(info.tp_size, 1)


class TestInstanceInfoPostInit(unittest.TestCase):
    """Tests for InstanceInfo.__post_init__ method."""

    def test_role_string_conversion(self):
        """Test role string is converted to InstanceRole enum."""
        info = InstanceInfo(role="decode", host_ip="10.0.0.1", port=8080)
        self.assertEqual(info.role, InstanceRole.DECODE)

    def test_invalid_role_string_raises_valueerror(self):
        """Test invalid role string raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            InstanceInfo(role="invalid_role", host_ip="10.0.0.1", port=8080)
        self.assertIn("Invalid role string", str(ctx.exception))

    def test_invalid_role_type_raises_typeerror(self):
        """Test invalid role type raises TypeError."""
        with self.assertRaises(TypeError) as ctx:
            InstanceInfo(role=123, host_ip="10.0.0.1", port=8080)
        self.assertIn("role must be InstanceRole or str", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
