# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import unittest

from fastdeploy.engine.sampling_params import SamplingParams

MAX_SEED = 922337203685477580


class TestSamplingParamsDeterminism(unittest.TestCase):
    """Test SamplingParams deterministic seed behavior"""

    _ENV_KEYS = ("FD_DETERMINISTIC_MODE",)

    def setUp(self):
        """Save and clear deterministic env vars"""
        self._saved_env = {k: os.environ.pop(k, None) for k in self._ENV_KEYS}

    def tearDown(self):
        """Restore original env vars"""
        for key, value in self._saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_fixed_seed_in_deterministic_mode(self):
        """seed=None should always resolve to 42 when FD_DETERMINISTIC_MODE=1"""
        os.environ["FD_DETERMINISTIC_MODE"] = "1"

        for _ in range(5):
            params = SamplingParams(seed=None)
            self.assertEqual(params.seed, 42)

    def test_random_seed_in_non_deterministic_mode(self):
        """seed=None should produce varying seeds when FD_DETERMINISTIC_MODE=0"""
        os.environ["FD_DETERMINISTIC_MODE"] = "0"

        seeds = {SamplingParams(seed=None).seed for _ in range(10)}
        self.assertGreaterEqual(len(seeds), 2)

    def test_explicit_seed_respected_in_both_modes(self):
        """Explicit seed values should be kept regardless of deterministic mode"""
        test_seeds = [0, 1, 100, MAX_SEED]
        for mode in ("0", "1"):
            os.environ["FD_DETERMINISTIC_MODE"] = mode
            for seed in test_seeds:
                params = SamplingParams(seed=seed)
                self.assertEqual(params.seed, seed)

    def test_seed_out_of_range_rejected(self):
        """Seeds outside [0, MAX_SEED] should raise ValueError"""
        with self.assertRaises(ValueError):
            SamplingParams(seed=-1)

        with self.assertRaises(ValueError):
            SamplingParams(seed=MAX_SEED + 1)

    def test_env_switch_changes_behavior(self):
        """Switching FD_DETERMINISTIC_MODE at runtime should affect subsequent SamplingParams"""
        os.environ["FD_DETERMINISTIC_MODE"] = "1"
        params_det = SamplingParams(seed=None)
        self.assertEqual(params_det.seed, 42)

        os.environ["FD_DETERMINISTIC_MODE"] = "0"
        seeds = {SamplingParams(seed=None).seed for _ in range(10)}
        # At least some seeds should differ from the fixed value
        self.assertGreaterEqual(len(seeds), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
