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

from unittest.mock import patch

import pytest

from fastdeploy.utils import find_free_ports


class TestFindFreePorts:
    """Unit tests for find_free_ports function."""

    def test_find_single_free_port_success(self):
        """Test finding a single free port successfully."""
        with patch("fastdeploy.utils.is_port_available", return_value=True):
            ports = find_free_ports(port_range=(20000, 20100), num_ports=1)
            assert len(ports) == 1
            assert 20000 <= ports[0] <= 20100

    def test_find_multiple_free_ports_success(self):
        """Test finding multiple free ports successfully."""
        with patch("fastdeploy.utils.is_port_available", return_value=True):
            ports = find_free_ports(port_range=(20000, 20100), num_ports=5)
            assert len(ports) == 5
            for port in ports:
                assert 20000 <= port <= 20100

    def test_find_ports_with_custom_host(self):
        """Test finding ports with a custom host."""
        with patch("fastdeploy.utils.is_port_available", return_value=True) as mock_avail:
            ports = find_free_ports(port_range=(30000, 30010), num_ports=2, host="127.0.0.1")
            assert len(ports) == 2
            # Verify is_port_available was called with the custom host
            for call in mock_avail.call_args_list:
                assert call[0][0] == "127.0.0.1"

    def test_find_all_ports_in_range(self):
        """Test finding all ports in a small range."""
        with patch("fastdeploy.utils.is_port_available", return_value=True):
            ports = find_free_ports(port_range=(40000, 40002), num_ports=3)
            assert len(ports) == 3
            # All ports should be from the range
            expected_ports = {40000, 40001, 40002}
            assert set(ports) == expected_ports

    def test_invalid_port_range_start_negative(self):
        """Test ValueError when port range start is negative."""
        with pytest.raises(ValueError, match="Invalid port range"):
            find_free_ports(port_range=(-1, 1000))

    def test_invalid_port_range_end_exceeds_max(self):
        """Test ValueError when port range end exceeds 65535."""
        with pytest.raises(ValueError, match="Invalid port range"):
            find_free_ports(port_range=(1000, 65536))

    def test_invalid_port_range_start_greater_than_end(self):
        """Test ValueError when port range start is greater than end."""
        with pytest.raises(ValueError, match="Invalid port range"):
            find_free_ports(port_range=(10000, 9000))

    def test_invalid_port_range_boundary_values(self):
        """Test port range boundary at exactly 0 and 65535."""
        # Valid: start = 0
        with patch("fastdeploy.utils.is_port_available", return_value=True):
            ports = find_free_ports(port_range=(0, 100), num_ports=1)
            assert len(ports) == 1

        # Valid: end = 65535
        with patch("fastdeploy.utils.is_port_available", return_value=True):
            ports = find_free_ports(port_range=(65530, 65535), num_ports=1)
            assert len(ports) == 1

    def test_num_ports_zero_raises_error(self):
        """Test ValueError when num_ports is zero."""
        with pytest.raises(ValueError, match="num_ports must be a positive integer"):
            find_free_ports(port_range=(20000, 30000), num_ports=0)

    def test_num_ports_negative_raises_error(self):
        """Test ValueError when num_ports is negative."""
        with pytest.raises(ValueError, match="num_ports must be a positive integer"):
            find_free_ports(port_range=(20000, 30000), num_ports=-1)

    def test_num_ports_larger_than_range_size(self):
        """Test ValueError when num_ports exceeds the range size."""
        # Range has only 5 ports (100-104), but requesting 6
        with pytest.raises(ValueError, match="num_ports is larger than range size"):
            find_free_ports(port_range=(100, 104), num_ports=6)

    def test_not_enough_free_ports_raises_runtime_error(self):
        """Test RuntimeError when not enough free ports are available."""
        # Mock to return False for all ports
        with patch("fastdeploy.utils.is_port_available", return_value=False):
            with pytest.raises(RuntimeError, match="Only found 0 free ports"):
                find_free_ports(port_range=(20000, 20010), num_ports=3)

    def test_partial_free_ports_raises_runtime_error(self):
        """Test RuntimeError when only some ports are free."""
        call_count = [0]

        def mock_availability(host, port):
            # Only first 2 ports are available
            call_count[0] += 1
            return call_count[0] <= 2

        with patch("fastdeploy.utils.is_port_available", side_effect=mock_availability):
            with pytest.raises(RuntimeError, match="Only found 2 free ports"):
                find_free_ports(port_range=(20000, 20005), num_ports=5)

    def test_random_start_offset(self):
        """Test that port scanning starts from a random offset."""
        # Track the order of ports checked
        checked_ports = []

        def mock_availability(host, port):
            checked_ports.append(port)
            return True

        with patch("fastdeploy.utils.is_port_available", side_effect=mock_availability):
            with patch("fastdeploy.utils.random.randint", return_value=0):
                ports = find_free_ports(port_range=(100, 105), num_ports=3)

        # With offset 0, ports should be checked in order
        assert checked_ports[:3] == [100, 101, 102]
        assert ports == [100, 101, 102]

    def test_random_start_offset_non_zero(self):
        """Test port scanning with non-zero random offset."""
        checked_ports = []

        def mock_availability(host, port):
            checked_ports.append(port)
            return True

        with patch("fastdeploy.utils.is_port_available", side_effect=mock_availability):
            # With offset 2, scanning starts from port 102
            with patch("fastdeploy.utils.random.randint", return_value=2):
                ports = find_free_ports(port_range=(100, 105), num_ports=3)

        # With offset 2, ports are rotated: [102, 103, 104, 105, 100, 101]
        assert checked_ports[:3] == [102, 103, 104]
        assert ports == [102, 103, 104]

    def test_single_port_range(self):
        """Test finding port from a single-port range."""
        with patch("fastdeploy.utils.is_port_available", return_value=True):
            ports = find_free_ports(port_range=(12345, 12345), num_ports=1)
            assert ports == [12345]

    def test_single_port_range_not_available(self):
        """Test RuntimeError when the single port in range is not available."""
        with patch("fastdeploy.utils.is_port_available", return_value=False):
            with pytest.raises(RuntimeError, match="Only found 0 free ports"):
                find_free_ports(port_range=(12345, 12345), num_ports=1)

    def test_default_parameters(self):
        """Test function with default parameters."""
        with patch("fastdeploy.utils.is_port_available", return_value=True):
            ports = find_free_ports()
            assert len(ports) == 1
            assert 8000 <= ports[0] <= 65535

    def test_stops_early_when_enough_ports_found(self):
        """Test that scanning stops as soon as enough ports are found."""
        checked_ports = []

        def mock_availability(host, port):
            checked_ports.append(port)
            return True

        with patch("fastdeploy.utils.is_port_available", side_effect=mock_availability):
            with patch("fastdeploy.utils.random.randint", return_value=0):
                # Range has 100 ports but we only need 2
                ports = find_free_ports(port_range=(20000, 20099), num_ports=2)

        # Should only check 2 ports, not all 100
        assert len(checked_ports) == 2
        assert len(ports) == 2

    def test_skips_unavailable_ports(self):
        """Test that unavailable ports are skipped."""
        checked_ports = []

        def mock_availability(host, port):
            checked_ports.append(port)
            # Only odd ports are available
            return port % 2 == 1

        with patch("fastdeploy.utils.is_port_available", side_effect=mock_availability):
            with patch("fastdeploy.utils.random.randint", return_value=0):
                ports = find_free_ports(port_range=(100, 110), num_ports=3)

        # Should find 3 odd ports: 101, 103, 105
        assert len(ports) == 3
        assert all(p % 2 == 1 for p in ports)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
