"""
Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import unittest

from fastdeploy.entrypoints.cli.benchmark.throughput import (
    BenchmarkThroughputSubcommand,
)


class TestBenchmarkThroughputSubcommand(unittest.TestCase):
    """
    测试 BenchmarkThroughputSubcommand 类。
    """

    def test_add_cli_args(self):
        parser = argparse.ArgumentParser()
        BenchmarkThroughputSubcommand.add_cli_args(parser)

        args = parser.parse_args(
            [
                "--backend",
                "fastdeploy",
                "--dataset-name",
                "random",
                "--input-len",
                "100",
                "--output-len",
                "50",
                "--num-prompts",
                "10",
            ]
        )
        self.assertEqual(args.backend, "fastdeploy")
        self.assertEqual(args.dataset_name, "random")
        self.assertEqual(args.input_len, 100)
        self.assertEqual(args.output_len, 50)
        self.assertEqual(args.num_prompts, 10)


# 如果你在命令行运行这个文件，下面的代码会执行测试
if __name__ == "__main__":
    unittest.main()
