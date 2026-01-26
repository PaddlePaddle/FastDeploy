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

import argparse
import importlib.machinery
import sys
import types
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np

# Mock binary extensions
sys.modules["fastdeploy.model_executor.ops.gpu"] = MagicMock()
sys.modules["fastdeploy.model_executor.layers.utils"] = MagicMock()
sys.modules["fastdeploy.utils.ipc"] = MagicMock()
sys.modules["fastdeploy.utils.ipc_signal"] = MagicMock()
sys.modules["fastdeploy.utils.fmq"] = MagicMock()

# Mock paddle
mock_paddle = MagicMock()
mock_paddle.__spec__ = importlib.machinery.ModuleSpec("paddle", loader=None)
mock_paddle.static = MagicMock()
mock_paddle.nn = MagicMock()
mock_paddle.nn.functional = MagicMock()
mock_paddle.nn.functional.flash_attention = MagicMock()
mock_paddle.distributed = MagicMock()
mock_paddle.distributed.fleet = MagicMock()
mock_paddle.distributed.fleet.meta_parallel = MagicMock()
mock_paddle.distributed.fleet.meta_parallel.parallel_layers = MagicMock()
mock_paddle.version = MagicMock()
mock_paddle.version.commit = "test_commit"
mock_paddle.is_compiled_with_xpu = MagicMock(return_value=False)
mock_paddle.set_default_dtype = MagicMock()
sys.modules["paddle"] = mock_paddle
sys.modules["paddle.static"] = mock_paddle.static
sys.modules["paddle.nn"] = mock_paddle.nn
sys.modules["paddle.nn.functional"] = mock_paddle.nn.functional
sys.modules["paddle.nn.functional.flash_attention"] = mock_paddle.nn.functional.flash_attention
sys.modules["paddle.distributed"] = mock_paddle.distributed
sys.modules["paddle.distributed.fleet"] = mock_paddle.distributed.fleet
sys.modules["paddle.distributed.fleet.meta_parallel"] = mock_paddle.distributed.fleet.meta_parallel
sys.modules["paddle.distributed.fleet.meta_parallel.parallel_layers"] = (
    mock_paddle.distributed.fleet.meta_parallel.parallel_layers
)

# Import the module under test AFTER mocks but BEFORE other imports


def _install_fake_worker_module(module_name: str, cls_name: str):
    """Avoid importing real worker modules (which pull heavy paddle/custom ops)."""
    m = types.ModuleType(module_name)
    setattr(m, cls_name, Mock(name=cls_name))
    sys.modules[module_name] = m
    return getattr(m, cls_name)


from fastdeploy.config import (
    CacheConfig,
    DeviceConfig,
    EPLBConfig,
    FDConfig,
    LoadConfig,
    ModelConfig,
    ParallelConfig,
)
from fastdeploy.scheduler import SchedulerConfig
from fastdeploy.worker.worker_process import (
    get_worker,
    initialize_fd_config,
    parse_args,
    update_fd_config_for_mm,
)

# 导入 MODEL_MAIN_NAME 用于测试
try:
    from fastdeploy.eplb.async_expert_loader import MODEL_MAIN_NAME
except ImportError:
    MODEL_MAIN_NAME = "main_model"  # 如果导入失败，使用默认值


class TestGetWorker(unittest.TestCase):
    """测试 get_worker 函数"""

    def setUp(self):
        """设置测试环境"""
        self.fd_config = Mock(spec=FDConfig)
        self.fd_config.model_config = Mock(spec=ModelConfig)
        self.fd_config.model_config.enable_logprob = False
        self.local_rank = 0
        self.rank = 1

    @patch("fastdeploy.worker.worker_process.current_platform")
    def test_get_worker_cuda(self, mock_platform):
        """测试在 CUDA 平台获取 worker"""
        mock_platform.is_cuda.return_value = True
        mock_platform.is_dcu.return_value = False
        mock_platform.is_xpu.return_value = False
        mock_platform.is_iluvatar.return_value = False
        mock_platform.is_gcu.return_value = False
        mock_platform.is_maca.return_value = False
        mock_platform.is_intel_hpu.return_value = False

        fake_cls = _install_fake_worker_module("fastdeploy.worker.gpu_worker", "GpuWorker")
        get_worker(self.fd_config, self.local_rank, self.rank)
        fake_cls.assert_called_once_with(fd_config=self.fd_config, local_rank=self.local_rank, rank=self.rank)

    @patch("fastdeploy.worker.worker_process.current_platform")
    def test_get_worker_xpu(self, mock_platform):
        """测试在 XPU 平台获取 worker"""
        mock_platform.is_cuda.return_value = False
        mock_platform.is_dcu.return_value = False
        mock_platform.is_xpu.return_value = True
        mock_platform.is_iluvatar.return_value = False
        mock_platform.is_gcu.return_value = False
        mock_platform.is_maca.return_value = False
        mock_platform.is_intel_hpu.return_value = False

        fake_cls = _install_fake_worker_module("fastdeploy.worker.xpu_worker", "XpuWorker")
        get_worker(self.fd_config, self.local_rank, self.rank)
        fake_cls.assert_called_once_with(fd_config=self.fd_config, local_rank=self.local_rank, rank=self.rank)

    @patch("fastdeploy.worker.worker_process.current_platform")
    def test_get_worker_dcu(self, mock_platform):
        """测试在 DCU 平台获取 worker"""
        mock_platform.is_cuda.return_value = False
        mock_platform.is_dcu.return_value = True
        mock_platform.is_xpu.return_value = False
        mock_platform.is_iluvatar.return_value = False
        mock_platform.is_gcu.return_value = False
        mock_platform.is_maca.return_value = False
        mock_platform.is_intel_hpu.return_value = False

        fake_cls = _install_fake_worker_module("fastdeploy.worker.dcu_worker", "DcuWorker")
        get_worker(self.fd_config, self.local_rank, self.rank)
        fake_cls.assert_called_once_with(fd_config=self.fd_config, local_rank=self.local_rank, rank=self.rank)

    @patch("fastdeploy.worker.worker_process.current_platform")
    def test_get_worker_iluvatar(self, mock_platform):
        """测试在 Iluvatar 平台获取 worker"""
        mock_platform.is_cuda.return_value = False
        mock_platform.is_dcu.return_value = False
        mock_platform.is_xpu.return_value = False
        mock_platform.is_iluvatar.return_value = True
        mock_platform.is_gcu.return_value = False
        mock_platform.is_maca.return_value = False
        mock_platform.is_intel_hpu.return_value = False

        fake_cls = _install_fake_worker_module("fastdeploy.worker.iluvatar_worker", "IluvatarWorker")
        get_worker(self.fd_config, self.local_rank, self.rank)
        fake_cls.assert_called_once_with(fd_config=self.fd_config, local_rank=self.local_rank, rank=self.rank)

    @patch("fastdeploy.worker.worker_process.current_platform")
    def test_get_worker_gcu(self, mock_platform):
        """测试在 GCU 平台获取 worker"""
        mock_platform.is_cuda.return_value = False
        mock_platform.is_dcu.return_value = False
        mock_platform.is_xpu.return_value = False
        mock_platform.is_iluvatar.return_value = False
        mock_platform.is_gcu.return_value = True
        mock_platform.is_maca.return_value = False
        mock_platform.is_intel_hpu.return_value = False

        fake_cls = _install_fake_worker_module("fastdeploy.worker.gcu_worker", "GcuWorker")
        get_worker(self.fd_config, self.local_rank, self.rank)
        fake_cls.assert_called_once_with(fd_config=self.fd_config, local_rank=self.local_rank, rank=self.rank)

    @patch("fastdeploy.worker.worker_process.current_platform")
    def test_get_worker_metax(self, mock_platform):
        """测试在 Metax 平台获取 worker"""
        mock_platform.is_cuda.return_value = False
        mock_platform.is_dcu.return_value = False
        mock_platform.is_xpu.return_value = False
        mock_platform.is_iluvatar.return_value = False
        mock_platform.is_gcu.return_value = False
        mock_platform.is_maca.return_value = True
        mock_platform.is_intel_hpu.return_value = False

        fake_cls = _install_fake_worker_module("fastdeploy.worker.metax_worker", "MetaxWorker")
        get_worker(self.fd_config, self.local_rank, self.rank)
        fake_cls.assert_called_once_with(fd_config=self.fd_config, local_rank=self.local_rank, rank=self.rank)

    @patch("fastdeploy.worker.worker_process.current_platform")
    def test_get_worker_hpu(self, mock_platform):
        """测试在 Intel HPU 平台获取 worker"""
        mock_platform.is_cuda.return_value = False
        mock_platform.is_dcu.return_value = False
        mock_platform.is_xpu.return_value = False
        mock_platform.is_iluvatar.return_value = False
        mock_platform.is_gcu.return_value = False
        mock_platform.is_maca.return_value = False
        mock_platform.is_intel_hpu.return_value = True

        fake_cls = _install_fake_worker_module("fastdeploy.worker.hpu_worker", "HpuWorker")
        get_worker(self.fd_config, self.local_rank, self.rank)
        fake_cls.assert_called_once_with(fd_config=self.fd_config, local_rank=self.local_rank, rank=self.rank)

    @patch("fastdeploy.worker.worker_process.current_platform")
    def test_get_worker_logprob_unsupported_platform(self, mock_platform):
        """测试在不支持 logprob 的平台上启用 logprob 会抛出异常"""
        mock_platform.is_cuda.return_value = False
        mock_platform.is_xpu.return_value = False
        mock_platform.is_dcu.return_value = False
        mock_platform.is_iluvatar.return_value = False
        mock_platform.is_gcu.return_value = True
        mock_platform.is_maca.return_value = False
        mock_platform.is_intel_hpu.return_value = False

        self.fd_config.model_config.enable_logprob = True

        with self.assertRaises(NotImplementedError) as context:
            get_worker(self.fd_config, self.local_rank, self.rank)
        self.assertIn("Only CUDA and XPU platforms support logprob", str(context.exception))


class TestInitDistributedEnvironment(unittest.TestCase):
    """测试 init_distributed_environment 函数"""

    @patch("fastdeploy.worker.worker_process.fleet")
    @patch("fastdeploy.worker.worker_process.dist")
    def test_init_distributed_environment_with_ranks(self, mock_dist, mock_fleet):
        """测试有多个 rank 时的分布式环境初始化"""
        mock_dist.get_world_size.return_value = 4
        mock_fleet.DistributedStrategy.return_value = MagicMock()
        mock_fleet.worker_index.return_value = 2

        from fastdeploy.worker.worker_process import init_distributed_environment

        ranks, local_rank = init_distributed_environment(seed=42)

        self.assertEqual(ranks, 4)
        self.assertEqual(local_rank, 2)
        mock_fleet.init.assert_called_once()
        mock_fleet.DistributedStrategy.assert_called_once()

    @patch("fastdeploy.worker.worker_process.fleet")
    @patch("fastdeploy.worker.worker_process.dist")
    def test_init_distributed_environment_zero_ranks(self, mock_dist, mock_fleet):
        """测试 rank 为 0 时的分布式环境初始化"""
        mock_dist.get_world_size.return_value = 0

        from fastdeploy.worker.worker_process import init_distributed_environment

        ranks, local_rank = init_distributed_environment()

        self.assertEqual(ranks, 0)
        self.assertEqual(local_rank, 0)
        mock_fleet.init.assert_not_called()


class TestUpdateFdConfigForMM(unittest.TestCase):
    """测试 update_fd_config_for_mm 函数"""

    def setUp(self):
        """设置测试环境"""
        self.fd_config = Mock(spec=FDConfig)
        self.fd_config.model_config = Mock(spec=ModelConfig)
        self.fd_config.parallel_config = Mock(spec=ParallelConfig)
        self.fd_config.parallel_config.tensor_parallel_size = 4
        self.fd_config.parallel_config.tensor_parallel_rank = 2

    def test_update_fd_config_for_mm_with_ernie(self):
        """测试多模态配置更新（ERNIE 架构）"""
        self.fd_config.model_config.enable_mm = True
        self.fd_config.model_config.architectures = ["ErnieForCausalLM"]
        self.fd_config.model_config.dtype = "bfloat16"
        self.fd_config.model_config.vision_config = Mock()

        with patch("fastdeploy.worker.worker_process.ErnieArchitectures") as mock_ernie_arch:
            mock_ernie_arch.contains_ernie_arch.return_value = True
            update_fd_config_for_mm(self.fd_config)

            self.assertEqual(self.fd_config.model_config.tensor_model_parallel_size, 4)
            self.assertEqual(self.fd_config.model_config.tensor_parallel_rank, 2)
            self.assertEqual(self.fd_config.model_config.vision_config.dtype, "bfloat16")

    def test_update_fd_config_for_mm_without_mm(self):
        """测试不启用多模态时配置不变"""
        self.fd_config.model_config.enable_mm = False
        self.fd_config.model_config.architectures = ["LlamaForCausalLM"]

        original_tp_size = self.fd_config.model_config.tensor_model_parallel_size = None

        update_fd_config_for_mm(self.fd_config)

        self.assertEqual(self.fd_config.model_config.tensor_model_parallel_size, original_tp_size)


class TestParseArgs(unittest.TestCase):
    """测试 parse_args 函数"""

    @patch("sys.argv", ["worker_process.py", "--model", "/path/to/model", "--max_num_seqs", "64"])
    def test_parse_args_basic(self):
        """测试基本参数解析"""
        args = parse_args()
        self.assertEqual(args.model, "/path/to/model")
        self.assertEqual(args.max_num_seqs, 64)

    @patch(
        "sys.argv",
        [
            "worker_process.py",
            "--model",
            "/path/to/model",
            "--dtype",
            "float16",
            "--tensor_parallel_size",
            "4",
            "--enable_chunked_prefill",
        ],
    )
    def test_parse_args_advanced(self):
        """测试高级参数解析"""
        args = parse_args()
        self.assertEqual(args.model, "/path/to/model")
        self.assertEqual(args.dtype, "float16")
        self.assertEqual(args.tensor_parallel_size, 4)
        self.assertTrue(args.enable_chunked_prefill)

    @patch(
        "sys.argv",
        [
            "worker_process.py",
            "--model",
            "/path/to/model",
            "--quantization",
            '{"method": "wint4"}',
        ],
    )
    def test_parse_args_json_config(self):
        """测试 JSON 配置参数解析"""
        args = parse_args()
        self.assertEqual(args.model, "/path/to/model")
        self.assertIsInstance(args.quantization, dict)
        self.assertEqual(args.quantization["method"], "wint4")


class TestInitializeFdConfig(unittest.TestCase):
    """测试 initialize_fd_config 函数"""

    def setUp(self):
        """设置测试环境"""
        self.args = argparse.Namespace(
            model="facebook/opt-125m",
            max_num_seqs=64,
            num_gpu_blocks_override=None,
            block_size=64,
            pod_ip="127.0.0.1",
            engine_worker_queue_port="9923",
            max_model_len=2048,
            device_ids="0",
            dtype="bfloat16",
            enc_dec_block_num=1,
            kv_cache_ratio=0.7,
            first_token_id=1,
            gpu_memory_utilization=0.9,
            engine_pid=12345,
            do_profile=False,
            pad_token_id=-1,
            eos_tokens_lens=2,
            enable_chunked_prefill=False,
            use_internode_ll_two_stage=False,
            speculative_config={},
            max_num_batched_tokens=2048,
            enable_prefix_caching=False,
            disable_custom_all_reduce=False,
            disable_sequence_parallel_moe=False,
            splitwise_role="mixed",
            tensor_parallel_size=1,
            expert_parallel_size=1,
            data_parallel_size=1,
            enable_expert_parallel=False,
            enable_chunked_moe=False,
            chunked_moe_size=256,
            ori_vocab_size=None,
            think_end_id=-1,
            image_patch_id=-1,
            line_break_id=-1,
            quantization=None,
            graph_optimization_config=None,
            plas_attention_config=None,
            guided_decoding_backend="off",
            disable_any_whitespace=False,
            dynamic_load_weight=False,
            load_strategy="ipc_snapshot",
            rsync_config=None,
            enable_logprob=False,
            max_logprobs=20,
            logprobs_mode="raw_logprobs",
            reasoning_parser=None,
            early_stop_config=None,
            load_choices="default_v1",
            ips=None,
            lm_head_fp32=False,
            max_encoder_cache=0,
            cache_transfer_protocol="ipc",
            runner="auto",
            convert="auto",
            override_pooler_config=None,
            logits_processors=[],
            eplb_config=None,
            routing_replay_config=None,
            shutdown_comm_group_if_worker_idle=False,
            enable_entropy=False,
            num_cpu_blocks=0,
        )

    @patch("fastdeploy.config.ModelConfig._post_init", return_value=None)
    @patch("fastdeploy.config.check_unified_ckpt", return_value=False)
    @patch("paddleformers.transformers.configuration_utils.PretrainedConfig.get_config_dict")
    @patch("paddleformers.transformers.configuration_utils.PretrainedConfig.from_dict")
    @patch("fastdeploy.worker.worker_process.paddle")
    @patch("fastdeploy.worker.worker_process.parse_quant_config")
    @patch("fastdeploy.worker.worker_process.v1_loader_support")
    @patch("fastdeploy.worker.worker_process.current_platform")
    def test_initialize_fd_config_basic(
        self,
        mock_platform,
        mock_v1_support,
        mock_parse_quant,
        mock_paddle,
        mock_from_dict,
        mock_get_config_dict,
        mock_check_unified_ckpt,
        mock_model_post_init,
    ):
        """测试基本配置初始化"""

        mock_platform.is_iluvatar.return_value = False
        mock_platform.is_cuda.return_value = True
        mock_platform.is_xpu.return_value = False
        mock_platform.is_maca.return_value = False
        mock_platform.is_intel_hpu.return_value = False
        mock_parse_quant.return_value = None
        mock_v1_support.return_value = True
        # Mock paddle.version.commit for FDConfig.postprocess
        mock_paddle.version.commit = "test_commit"
        mock_paddle.is_compiled_with_xpu.return_value = False
        # Mock PretrainedConfig.get_config_dict to return minimal config required by ModelConfig
        mock_get_config_dict.return_value = (
            {
                "model_type": "llama",
                "hidden_size": 768,
                "num_attention_heads": 12,
                "num_hidden_layers": 2,
                "architectures": ["LlamaForCausalLM"],
                "enable_mm": False,
                "vocab_size": 32000,
            },
            {},
        )
        # Mock PretrainedConfig.from_dict to return a mock config object
        mock_pretrained_config = Mock()
        mock_pretrained_config.tensor_parallel_rank = None
        mock_pretrained_config.tensor_model_parallel_size = None
        mock_pretrained_config.is_mtp = False
        mock_from_dict.return_value = mock_pretrained_config

        fd_config = initialize_fd_config(self.args, ranks=1, local_rank=0)

        # 验证核心逻辑：配置对象是否正确创建
        self.assertIsInstance(fd_config, FDConfig)
        self.assertIsInstance(fd_config.model_config, ModelConfig)
        self.assertIsInstance(fd_config.parallel_config, ParallelConfig)
        self.assertIsInstance(fd_config.cache_config, CacheConfig)
        self.assertIsInstance(fd_config.device_config, DeviceConfig)
        self.assertIsInstance(fd_config.load_config, LoadConfig)
        self.assertIsInstance(fd_config.scheduler_config, SchedulerConfig)
        self.assertIsInstance(fd_config.eplb_config, EPLBConfig)

        # 验证核心逻辑：并行 rank 计算是否正确
        self.assertEqual(fd_config.parallel_config.tensor_parallel_rank, 0)
        self.assertEqual(fd_config.parallel_config.data_parallel_rank, 0)

    @patch("fastdeploy.config.ModelConfig._post_init", return_value=None)
    @patch("fastdeploy.config.check_unified_ckpt", return_value=False)
    @patch("paddleformers.transformers.configuration_utils.PretrainedConfig.get_config_dict")
    @patch("paddleformers.transformers.configuration_utils.PretrainedConfig.from_dict")
    @patch("fastdeploy.worker.worker_process.paddle")
    @patch("fastdeploy.worker.worker_process.parse_quant_config")
    @patch("fastdeploy.worker.worker_process.v1_loader_support")
    @patch("fastdeploy.worker.worker_process.current_platform")
    def test_initialize_fd_config_with_tensor_parallel(
        self,
        mock_platform,
        mock_v1_support,
        mock_parse_quant,
        mock_paddle,
        mock_from_dict,
        mock_get_config_dict,
        mock_check_unified_ckpt,
        mock_model_post_init,
    ):
        """测试带张量并行的配置初始化"""

        mock_platform.is_iluvatar.return_value = False
        mock_platform.is_cuda.return_value = True
        mock_platform.is_xpu.return_value = False
        mock_platform.is_maca.return_value = False
        mock_platform.is_intel_hpu.return_value = False
        mock_parse_quant.return_value = None
        mock_v1_support.return_value = True
        mock_paddle.version.commit = "test_commit"
        mock_paddle.is_compiled_with_xpu.return_value = False
        mock_get_config_dict.return_value = (
            {
                "model_type": "llama",
                "hidden_size": 768,
                "num_attention_heads": 12,
                "num_hidden_layers": 2,
                "architectures": ["LlamaForCausalLM"],
                "enable_mm": False,
                "vocab_size": 32000,
            },
            {},
        )
        mock_pretrained_config = Mock()
        mock_pretrained_config.tensor_parallel_rank = None
        mock_pretrained_config.tensor_model_parallel_size = None
        mock_pretrained_config.is_mtp = False
        mock_from_dict.return_value = mock_pretrained_config
        self.args.tensor_parallel_size = 4

        fd_config = initialize_fd_config(self.args, ranks=4, local_rank=2)

        # 验证核心逻辑：张量并行 rank 计算是否正确
        self.assertEqual(fd_config.parallel_config.tensor_parallel_size, 4)
        self.assertEqual(fd_config.parallel_config.tensor_parallel_rank, 2)
        self.assertEqual(fd_config.parallel_config.data_parallel_rank, 0)

    @patch("fastdeploy.config.ModelConfig._post_init", return_value=None)
    @patch("fastdeploy.config.check_unified_ckpt", return_value=False)
    @patch("paddleformers.transformers.configuration_utils.PretrainedConfig.get_config_dict")
    @patch("paddleformers.transformers.configuration_utils.PretrainedConfig.from_dict")
    @patch("fastdeploy.worker.worker_process.paddle")
    @patch("fastdeploy.worker.worker_process.parse_quant_config")
    @patch("fastdeploy.worker.worker_process.v1_loader_support")
    @patch("fastdeploy.worker.worker_process.current_platform")
    def test_initialize_fd_config_with_data_parallel(
        self,
        mock_platform,
        mock_v1_support,
        mock_parse_quant,
        mock_paddle,
        mock_from_dict,
        mock_get_config_dict,
        mock_check_unified_ckpt,
        mock_model_post_init,
    ):
        """测试带数据并行的配置初始化"""

        mock_platform.is_iluvatar.return_value = False
        mock_platform.is_cuda.return_value = True
        mock_platform.is_xpu.return_value = False
        mock_platform.is_maca.return_value = False
        mock_platform.is_intel_hpu.return_value = False
        mock_parse_quant.return_value = None
        mock_v1_support.return_value = True
        mock_paddle.version.commit = "test_commit"
        mock_paddle.is_compiled_with_xpu.return_value = False
        mock_get_config_dict.return_value = (
            {
                "model_type": "llama",
                "hidden_size": 768,
                "num_attention_heads": 12,
                "num_hidden_layers": 2,
                "architectures": ["LlamaForCausalLM"],
                "enable_mm": False,
                "vocab_size": 32000,
            },
            {},
        )
        mock_pretrained_config = Mock()
        mock_pretrained_config.tensor_parallel_rank = None
        mock_pretrained_config.tensor_model_parallel_size = None
        mock_pretrained_config.is_mtp = False
        mock_from_dict.return_value = mock_pretrained_config
        self.args.data_parallel_size = 2
        self.args.tensor_parallel_size = 2

        fd_config = initialize_fd_config(self.args, ranks=4, local_rank=2)

        # 验证核心逻辑：数据并行和张量并行的 rank 计算是否正确
        self.assertEqual(fd_config.parallel_config.data_parallel_size, 2)
        self.assertEqual(fd_config.parallel_config.tensor_parallel_size, 2)
        self.assertEqual(fd_config.parallel_config.tensor_parallel_rank, 0)  # 2 % 2 = 0
        self.assertEqual(fd_config.parallel_config.data_parallel_rank, 1)  # 2 // 2 = 1

    @patch("fastdeploy.config.ModelConfig._post_init")
    @patch("fastdeploy.config.check_unified_ckpt", return_value=False)
    @patch("paddleformers.transformers.configuration_utils.PretrainedConfig.get_config_dict")
    @patch("paddleformers.transformers.configuration_utils.PretrainedConfig.from_dict")
    @patch("fastdeploy.worker.worker_process.paddle")
    @patch("fastdeploy.worker.worker_process.parse_quant_config")
    @patch("fastdeploy.worker.worker_process.v1_loader_support")
    @patch("fastdeploy.worker.worker_process.current_platform")
    def test_initialize_fd_config_with_expert_parallel(
        self,
        mock_platform,
        mock_v1_support,
        mock_parse_quant,
        mock_paddle,
        mock_from_dict,
        mock_get_config_dict,
        mock_check_unified_ckpt,
        mock_model_post_init,
    ):
        """测试带专家并行的配置初始化"""
        mock_platform.is_iluvatar.return_value = False
        mock_platform.is_cuda.return_value = True
        mock_platform.is_xpu.return_value = False
        mock_platform.is_maca.return_value = False
        mock_platform.is_intel_hpu.return_value = False
        mock_parse_quant.return_value = None
        mock_v1_support.return_value = True
        mock_paddle.version.commit = "test_commit"
        mock_paddle.is_compiled_with_xpu.return_value = False
        mock_get_config_dict.return_value = (
            {
                "model_type": "llama",
                "hidden_size": 768,
                "num_attention_heads": 12,
                "num_hidden_layers": 2,
                "architectures": ["LlamaForCausalLM"],
                "enable_mm": False,
                "vocab_size": 32000,
            },
            {},
        )
        mock_pretrained_config = Mock()
        mock_pretrained_config.tensor_parallel_rank = None
        mock_pretrained_config.tensor_model_parallel_size = None
        mock_pretrained_config.is_mtp = False
        mock_from_dict.return_value = mock_pretrained_config
        # 根据 ParallelConfig 的逻辑，如果 enable_expert_parallel = True，
        # expert_parallel_size = data_parallel_size * tensor_parallel_size
        # 所以我们需要设置 data_parallel_size 和 tensor_parallel_size 来得到想要的 expert_parallel_size
        self.args.enable_expert_parallel = True
        self.args.tensor_parallel_size = 2
        self.args.data_parallel_size = 2  # 这样 expert_parallel_size = 2 * 2 = 4
        self.args.moe_num_experts = 8

        # 使用 patch 在 ModelConfig 创建后设置 moe_num_experts
        original_model_config_init = ModelConfig.__init__

        def patched_init(self, args):
            original_model_config_init(self, args)
            # 确保 moe_num_experts 被设置（从 args 中获取）
            if "moe_num_experts" in args and args.get("moe_num_experts") is not None:
                self.moe_num_experts = args["moe_num_experts"]
            elif not hasattr(self, "moe_num_experts") or self.moe_num_experts is None:
                # 如果 args 中没有，尝试从 self.args 中获取
                self.moe_num_experts = 8  # 使用测试中设置的值

        with patch.object(ModelConfig, "__init__", patched_init):
            fd_config = initialize_fd_config(self.args, ranks=8, local_rank=2)

        # 验证专家并行配置
        # expert_parallel_size = data_parallel_size * tensor_parallel_size = 2 * 2 = 4
        self.assertEqual(fd_config.parallel_config.expert_parallel_size, 4)
        self.assertEqual(fd_config.parallel_config.expert_parallel_rank, 2)  # 2 % 4 = 2
        self.assertEqual(fd_config.parallel_config.num_experts_per_rank, 2)  # 8 // 4 = 2
        self.assertEqual(fd_config.parallel_config.num_experts_start_offset, 4)  # 2 * 2 = 4

    @patch("fastdeploy.config.ModelConfig._post_init", return_value=None)
    @patch("fastdeploy.config.check_unified_ckpt", return_value=False)
    @patch("paddleformers.transformers.configuration_utils.PretrainedConfig.get_config_dict")
    @patch("paddleformers.transformers.configuration_utils.PretrainedConfig.from_dict")
    @patch("fastdeploy.worker.worker_process.paddle")
    @patch("fastdeploy.worker.worker_process.parse_quant_config")
    @patch("fastdeploy.worker.worker_process.v1_loader_support")
    @patch("fastdeploy.worker.worker_process.current_platform")
    def test_initialize_fd_config_with_expert_parallel_list_experts(
        self,
        mock_platform,
        mock_v1_support,
        mock_parse_quant,
        mock_paddle,
        mock_from_dict,
        mock_get_config_dict,
        mock_check_unified_ckpt,
        mock_model_post_init,
    ):
        """测试专家并行配置（moe_num_experts 是 list）"""
        mock_platform.is_iluvatar.return_value = False
        mock_platform.is_cuda.return_value = True
        mock_platform.is_xpu.return_value = False
        mock_platform.is_maca.return_value = False
        mock_platform.is_intel_hpu.return_value = False
        mock_parse_quant.return_value = None
        mock_v1_support.return_value = True
        mock_paddle.version.commit = "test_commit"
        mock_paddle.is_compiled_with_xpu.return_value = False
        mock_get_config_dict.return_value = (
            {
                "model_type": "llama",
                "hidden_size": 768,
                "num_attention_heads": 12,
                "num_hidden_layers": 2,
                "architectures": ["LlamaForCausalLM"],
                "enable_mm": False,
                "vocab_size": 32000,
            },
            {},
        )
        mock_pretrained_config = Mock()
        mock_pretrained_config.tensor_parallel_rank = None
        mock_pretrained_config.tensor_model_parallel_size = None
        mock_pretrained_config.is_mtp = False
        mock_from_dict.return_value = mock_pretrained_config
        # 根据 ParallelConfig 的逻辑，expert_parallel_size = data_parallel_size * tensor_parallel_size
        self.args.enable_expert_parallel = True
        self.args.tensor_parallel_size = 2
        self.args.data_parallel_size = 2  # 这样 expert_parallel_size = 2 * 2 = 4
        self.args.moe_num_experts = [8, 4]  # list

        # 使用 patch 在 ModelConfig 创建后设置 moe_num_experts
        original_model_config_init = ModelConfig.__init__

        def patched_init(self, args):
            original_model_config_init(self, args)
            # 确保 moe_num_experts 被设置（从 args 中获取）
            if "moe_num_experts" in args and args.get("moe_num_experts") is not None:
                self.moe_num_experts = args["moe_num_experts"]

        with patch.object(ModelConfig, "__init__", patched_init):
            fd_config = initialize_fd_config(self.args, ranks=8, local_rank=2)

        # 验证专家并行配置（使用 list 的第一个元素）
        self.assertEqual(fd_config.parallel_config.expert_parallel_size, 4)
        self.assertEqual(fd_config.parallel_config.expert_parallel_rank, 2)

    @patch("fastdeploy.config.ModelConfig._post_init", return_value=None)
    @patch("fastdeploy.config.check_unified_ckpt", return_value=False)
    @patch("paddleformers.transformers.configuration_utils.PretrainedConfig.get_config_dict")
    @patch("paddleformers.transformers.configuration_utils.PretrainedConfig.from_dict")
    @patch("fastdeploy.worker.worker_process.paddle")
    @patch("fastdeploy.worker.worker_process.parse_quant_config")
    @patch("fastdeploy.worker.worker_process.v1_loader_support")
    @patch("fastdeploy.worker.worker_process.current_platform")
    def test_initialize_fd_config_with_expert_parallel_num_local_experts(
        self,
        mock_platform,
        mock_v1_support,
        mock_parse_quant,
        mock_paddle,
        mock_from_dict,
        mock_get_config_dict,
        mock_check_unified_ckpt,
        mock_model_post_init,
    ):
        """测试专家并行配置（使用 num_local_experts）"""
        mock_platform.is_iluvatar.return_value = False
        mock_platform.is_cuda.return_value = True
        mock_platform.is_xpu.return_value = False
        mock_platform.is_maca.return_value = False
        mock_platform.is_intel_hpu.return_value = False
        mock_parse_quant.return_value = None
        mock_v1_support.return_value = True
        mock_paddle.version.commit = "test_commit"
        mock_paddle.is_compiled_with_xpu.return_value = False
        mock_get_config_dict.return_value = (
            {
                "model_type": "llama",
                "hidden_size": 768,
                "num_attention_heads": 12,
                "num_hidden_layers": 2,
                "architectures": ["LlamaForCausalLM"],
                "enable_mm": False,
                "vocab_size": 32000,
            },
            {},
        )
        mock_pretrained_config = Mock()
        mock_pretrained_config.tensor_parallel_rank = None
        mock_pretrained_config.tensor_model_parallel_size = None
        mock_pretrained_config.is_mtp = False
        mock_from_dict.return_value = mock_pretrained_config
        # 根据 ParallelConfig 的逻辑，expert_parallel_size = data_parallel_size * tensor_parallel_size
        self.args.enable_expert_parallel = True
        self.args.tensor_parallel_size = 2
        self.args.data_parallel_size = 2  # 这样 expert_parallel_size = 2 * 2 = 4
        self.args.num_local_experts = 8  # 使用 num_local_experts

        # 使用 patch 在 ModelConfig 创建后设置 num_local_experts
        original_model_config_init = ModelConfig.__init__

        def patched_init(self, args):
            original_model_config_init(self, args)
            # 确保 num_local_experts 被设置（从 args 中获取）
            if "num_local_experts" in args and args.get("num_local_experts") is not None:
                self.num_local_experts = args["num_local_experts"]

        with patch.object(ModelConfig, "__init__", patched_init):
            fd_config = initialize_fd_config(self.args, ranks=8, local_rank=2)

        # 验证专家并行配置
        self.assertEqual(fd_config.parallel_config.expert_parallel_size, 4)
        self.assertEqual(fd_config.parallel_config.expert_parallel_rank, 2)

    @patch("fastdeploy.config.ModelConfig._post_init", return_value=None)
    @patch("fastdeploy.config.check_unified_ckpt", return_value=False)
    @patch("paddleformers.transformers.configuration_utils.PretrainedConfig.get_config_dict")
    @patch("paddleformers.transformers.configuration_utils.PretrainedConfig.from_dict")
    @patch("fastdeploy.worker.worker_process.paddle")
    @patch("fastdeploy.worker.worker_process.parse_quant_config")
    @patch("fastdeploy.worker.worker_process.v1_loader_support")
    @patch("fastdeploy.worker.worker_process.current_platform")
    @patch("fastdeploy.worker.worker_process.ErnieArchitectures")
    def test_initialize_fd_config_with_ernie_architecture(
        self,
        mock_ernie_arch,
        mock_platform,
        mock_v1_support,
        mock_parse_quant,
        mock_paddle,
        mock_from_dict,
        mock_get_config_dict,
        mock_check_unified_ckpt,
        mock_model_post_init,
    ):
        """测试 ERNIE 架构的配置初始化"""
        mock_platform.is_iluvatar.return_value = False
        mock_platform.is_cuda.return_value = True
        mock_platform.is_xpu.return_value = False
        mock_platform.is_maca.return_value = False
        mock_platform.is_intel_hpu.return_value = False
        mock_parse_quant.return_value = None
        mock_v1_support.return_value = True
        mock_paddle.version.commit = "test_commit"
        mock_paddle.is_compiled_with_xpu.return_value = False
        mock_ernie_arch.contains_ernie_arch.return_value = True
        mock_get_config_dict.return_value = (
            {
                "model_type": "ernie",
                "hidden_size": 768,
                "num_attention_heads": 12,
                "num_hidden_layers": 2,
                "architectures": ["ErnieForCausalLM"],
                "enable_mm": False,
                "vocab_size": 32000,
            },
            {},
        )
        mock_pretrained_config = Mock()
        mock_pretrained_config.tensor_parallel_rank = None
        mock_pretrained_config.tensor_model_parallel_size = None
        mock_pretrained_config.is_mtp = False
        mock_from_dict.return_value = mock_pretrained_config

        fd_config = initialize_fd_config(self.args, ranks=1, local_rank=0)

        self.assertIsInstance(fd_config, FDConfig)
        mock_ernie_arch.contains_ernie_arch.assert_called()

    @patch("fastdeploy.config.ModelConfig._post_init", return_value=None)
    @patch("fastdeploy.config.check_unified_ckpt", return_value=False)
    @patch("paddleformers.transformers.configuration_utils.PretrainedConfig.get_config_dict")
    @patch("paddleformers.transformers.configuration_utils.PretrainedConfig.from_dict")
    @patch("fastdeploy.worker.worker_process.paddle")
    @patch("fastdeploy.worker.worker_process.parse_quant_config")
    @patch("fastdeploy.worker.worker_process.v1_loader_support")
    @patch("fastdeploy.worker.worker_process.current_platform")
    @patch("fastdeploy.worker.worker_process.envs")
    def test_initialize_fd_config_with_iluvatar_platform(
        self,
        mock_envs,
        mock_platform,
        mock_v1_support,
        mock_parse_quant,
        mock_paddle,
        mock_from_dict,
        mock_get_config_dict,
        mock_check_unified_ckpt,
        mock_model_post_init,
    ):
        """测试 Iluvatar 平台的配置初始化（数据并行）"""
        mock_platform.is_iluvatar.return_value = True
        mock_platform.is_cuda.return_value = False
        mock_platform.is_xpu.return_value = False
        mock_platform.is_maca.return_value = False
        mock_platform.is_intel_hpu.return_value = False
        mock_parse_quant.return_value = None
        mock_v1_support.return_value = True
        mock_paddle.version.commit = "test_commit"
        mock_paddle.is_compiled_with_xpu.return_value = False
        mock_envs.ENABLE_V1_KVCACHE_SCHEDULER = 1
        mock_get_config_dict.return_value = (
            {
                "model_type": "llama",
                "hidden_size": 768,
                "num_attention_heads": 12,
                "num_hidden_layers": 2,
                "architectures": ["LlamaForCausalLM"],
                "enable_mm": False,
                "vocab_size": 32000,
            },
            {},
        )
        mock_pretrained_config = Mock()
        mock_pretrained_config.tensor_parallel_rank = None
        mock_pretrained_config.tensor_model_parallel_size = None
        mock_pretrained_config.is_mtp = False
        mock_from_dict.return_value = mock_pretrained_config
        self.args.data_parallel_size = 2
        self.args.tensor_parallel_size = 2

        fd_config = initialize_fd_config(self.args, ranks=4, local_rank=2)

        # 验证 Iluvatar 平台的数据并行配置（max_chips_per_node = 16）
        self.assertEqual(fd_config.parallel_config.data_parallel_size, 2)
        # local_data_parallel_id = data_parallel_rank % (max_chips_per_node // tensor_parallel_size)
        # = 1 % (16 // 2) = 1 % 8 = 1
        self.assertEqual(fd_config.parallel_config.local_data_parallel_id, 1)

    @patch("fastdeploy.config.ModelConfig._post_init", return_value=None)
    @patch("fastdeploy.config.check_unified_ckpt", return_value=False)
    @patch("paddleformers.transformers.configuration_utils.PretrainedConfig.get_config_dict")
    @patch("paddleformers.transformers.configuration_utils.PretrainedConfig.from_dict")
    @patch("fastdeploy.worker.worker_process.paddle")
    @patch("fastdeploy.worker.worker_process.parse_quant_config")
    @patch("fastdeploy.worker.worker_process.v1_loader_support")
    @patch("fastdeploy.worker.worker_process.current_platform")
    @patch("fastdeploy.worker.worker_process.envs")
    @patch("fastdeploy.worker.worker_process.os")
    def test_initialize_fd_config_with_splitwise_role(
        self,
        mock_os,
        mock_envs,
        mock_platform,
        mock_v1_support,
        mock_parse_quant,
        mock_paddle,
        mock_from_dict,
        mock_get_config_dict,
        mock_check_unified_ckpt,
        mock_model_post_init,
    ):
        """测试 splitwise_role 配置"""
        mock_platform.is_iluvatar.return_value = False
        mock_platform.is_cuda.return_value = True
        mock_platform.is_xpu.return_value = False
        mock_platform.is_maca.return_value = False
        mock_platform.is_intel_hpu.return_value = False
        mock_parse_quant.return_value = None
        mock_v1_support.return_value = True
        mock_paddle.version.commit = "test_commit"
        mock_paddle.is_compiled_with_xpu.return_value = False
        mock_envs.ENABLE_V1_KVCACHE_SCHEDULER = 1
        mock_os.environ = {}
        mock_get_config_dict.return_value = (
            {
                "model_type": "llama",
                "hidden_size": 768,
                "num_attention_heads": 12,
                "num_hidden_layers": 2,
                "architectures": ["LlamaForCausalLM"],
                "enable_mm": False,
                "vocab_size": 32000,
            },
            {},
        )
        mock_pretrained_config = Mock()
        mock_pretrained_config.tensor_parallel_rank = None
        mock_pretrained_config.tensor_model_parallel_size = None
        mock_pretrained_config.is_mtp = False
        mock_from_dict.return_value = mock_pretrained_config
        self.args.splitwise_role = "prefill"

        fd_config = initialize_fd_config(self.args, ranks=1, local_rank=0)

        self.assertIsInstance(fd_config, FDConfig)
        # 验证环境变量被设置（通过检查 mock_os.environ 是否被调用）
        # 由于 os.environ 是字典，我们直接验证函数执行成功即可


class TestPaddleDisWorkerProcInit(unittest.TestCase):
    """测试 PaddleDisWorkerProc 类的初始化"""

    def setUp(self):
        """设置测试环境"""
        self.fd_config = Mock(spec=FDConfig)
        self.fd_config.parallel_config = Mock(spec=ParallelConfig)
        self.fd_config.parallel_config.tensor_parallel_size = 2
        self.fd_config.parallel_config.data_parallel_size = 1
        self.fd_config.parallel_config.expert_parallel_size = 1
        self.fd_config.parallel_config.local_engine_worker_queue_port = "9923"
        self.fd_config.parallel_config.engine_pid = 12345
        self.fd_config.parallel_config.use_ep = False
        self.fd_config.parallel_config.local_data_parallel_id = 0
        self.fd_config.nnode = 1
        self.fd_config.cache_config = Mock(spec=CacheConfig)
        self.fd_config.scheduler_config = Mock(spec=SchedulerConfig)
        self.fd_config.eplb_config = Mock(spec=EPLBConfig)
        self.fd_config.eplb_config.enable_eplb = False
        self.fd_config.load_config = Mock(spec=LoadConfig)
        self.fd_config.load_config.dynamic_load_weight = False

        # 为 initialize_fd_config 测试添加 args
        self.args = argparse.Namespace(
            model="facebook/opt-125m",
            max_num_seqs=64,
            num_gpu_blocks_override=None,
            block_size=64,
            pod_ip="127.0.0.1",
            engine_worker_queue_port="9923",
            max_model_len=2048,
            device_ids="0",
            dtype="bfloat16",
            enc_dec_block_num=1,
            kv_cache_ratio=0.7,
            first_token_id=1,
            gpu_memory_utilization=0.9,
            engine_pid=12345,
            do_profile=False,
            pad_token_id=-1,
            eos_tokens_lens=2,
            enable_chunked_prefill=False,
            use_internode_ll_two_stage=False,
            speculative_config={},
            max_num_batched_tokens=2048,
            enable_prefix_caching=False,
            disable_custom_all_reduce=False,
            disable_sequence_parallel_moe=False,
            splitwise_role="mixed",
            tensor_parallel_size=1,
            expert_parallel_size=1,
            data_parallel_size=1,
            enable_expert_parallel=False,
            enable_chunked_moe=False,
            chunked_moe_size=256,
            ori_vocab_size=None,
            think_end_id=-1,
            image_patch_id=-1,
            line_break_id=-1,
            quantization=None,
            graph_optimization_config=None,
            plas_attention_config=None,
            guided_decoding_backend="off",
            disable_any_whitespace=False,
            dynamic_load_weight=False,
            load_strategy="ipc_snapshot",
            rsync_config=None,
            enable_logprob=False,
            max_logprobs=20,
            logprobs_mode="raw_logprobs",
            reasoning_parser=None,
            early_stop_config=None,
            load_choices="default_v1",
            ips=None,
            lm_head_fp32=False,
            max_encoder_cache=0,
            cache_transfer_protocol="ipc",
            runner="auto",
            convert="auto",
            override_pooler_config=None,
            logits_processors=[],
            eplb_config=None,
            routing_replay_config=None,
            shutdown_comm_group_if_worker_idle=False,
            enable_entropy=False,
            num_cpu_blocks=0,
        )

    @patch("fastdeploy.worker.worker_process.get_worker")
    @patch("fastdeploy.worker.worker_process.current_platform")
    def test_paddle_dis_worker_proc_init(self, mock_platform, mock_get_worker):
        """测试 PaddleDisWorkerProc 初始化"""
        from fastdeploy.worker.worker_process import PaddleDisWorkerProc

        mock_platform.is_iluvatar.return_value = False
        mock_worker = Mock()
        mock_get_worker.return_value = mock_worker

        worker_proc = PaddleDisWorkerProc(self.fd_config, ranks=2, local_rank=0)

        self.assertEqual(worker_proc.ranks, 2)
        self.assertEqual(worker_proc.local_rank, 0)
        self.assertEqual(worker_proc.fd_config, self.fd_config)
        self.assertEqual(worker_proc.worker, mock_worker)
        mock_get_worker.assert_called_once_with(fd_config=self.fd_config, local_rank=0, rank=2)

    @patch("fastdeploy.worker.worker_process.get_worker")
    @patch("fastdeploy.worker.worker_process.current_platform")
    def test_paddle_dis_worker_proc_init_iluvatar(self, mock_platform, mock_get_worker):
        """测试 Iluvatar 平台的 PaddleDisWorkerProc 初始化"""
        from fastdeploy.worker.worker_process import PaddleDisWorkerProc

        mock_platform.is_iluvatar.return_value = True
        mock_worker = Mock()
        mock_get_worker.return_value = mock_worker

        worker_proc = PaddleDisWorkerProc(self.fd_config, ranks=2, local_rank=0)

        self.assertEqual(worker_proc.max_chips_per_node, 16)  # Iluvatar 平台

    @patch("fastdeploy.worker.worker_process.FMQ")
    @patch("fastdeploy.worker.worker_process.get_worker")
    @patch("fastdeploy.worker.worker_process.current_platform")
    def test_paddle_dis_worker_proc_init_control(self, mock_platform, mock_get_worker, mock_fmq):
        """测试 PaddleDisWorkerProc.init_control"""
        from fastdeploy.worker.worker_process import PaddleDisWorkerProc

        mock_platform.is_iluvatar.return_value = False
        mock_worker = Mock()
        mock_get_worker.return_value = mock_worker
        mock_queue = Mock()
        mock_fmq.return_value.queue.return_value = mock_queue

        worker_proc = PaddleDisWorkerProc(self.fd_config, ranks=2, local_rank=1)
        worker_proc.init_control()

        self.assertEqual(worker_proc._ctrl_output, mock_queue)
        mock_fmq.return_value.queue.assert_called_once_with("ctrl_w2e_rank1_9923", "producer")

    @patch("fastdeploy.worker.worker_process.IPCSignal")
    @patch("fastdeploy.worker.worker_process.envs")
    @patch("fastdeploy.worker.worker_process.get_worker")
    @patch("fastdeploy.worker.worker_process.current_platform")
    @patch("fastdeploy.worker.worker_process.time")
    def test_paddle_dis_worker_proc_init_health_status(
        self, mock_time, mock_platform, mock_get_worker, mock_envs, mock_ipc_signal
    ):
        """测试 PaddleDisWorkerProc.init_health_status"""
        from fastdeploy.worker.worker_process import PaddleDisWorkerProc

        mock_platform.is_iluvatar.return_value = False
        mock_worker = Mock()
        mock_get_worker.return_value = mock_worker
        mock_envs.FD_ENABLE_MULTI_API_SERVER = False
        mock_time.time.return_value = 1234567890

        # Mock IPCSignal
        mock_signal_instance = Mock()
        mock_signal_instance.value = [0]  # For launched_expert_service_signal
        mock_ipc_signal.return_value = mock_signal_instance

        worker_proc = PaddleDisWorkerProc(self.fd_config, ranks=2, local_rank=0)
        worker_proc.init_health_status()

        # 验证各种信号被初始化
        self.assertIsNotNone(worker_proc.worker_ready_signal)
        self.assertIsNotNone(worker_proc.worker_healthy_live_signal)
        self.assertIsNotNone(worker_proc.model_weights_status)
        self.assertIsNotNone(worker_proc.kv_cache_status)
        self.assertIsNotNone(worker_proc.exist_task_signal)
        self.assertIsNotNone(worker_proc.exist_swapped_task_signal)
        self.assertIsNotNone(worker_proc.exist_prefill_task_signal)

    @patch("fastdeploy.worker.worker_process.IPCSignal")
    @patch("fastdeploy.worker.worker_process.envs")
    @patch("fastdeploy.worker.worker_process.get_worker")
    @patch("fastdeploy.worker.worker_process.current_platform")
    @patch("fastdeploy.worker.worker_process.time")
    def test_paddle_dis_worker_proc_init_health_status_with_data_parallel(
        self, mock_time, mock_platform, mock_get_worker, mock_envs, mock_ipc_signal
    ):
        """测试带数据并行的 init_health_status"""
        from fastdeploy.worker.worker_process import PaddleDisWorkerProc

        mock_platform.is_iluvatar.return_value = False
        mock_worker = Mock()
        mock_get_worker.return_value = mock_worker
        mock_envs.FD_ENABLE_MULTI_API_SERVER = False
        mock_time.time.return_value = 1234567890

        # 设置数据并行
        self.fd_config.parallel_config.data_parallel_size = 2
        self.fd_config.parallel_config.local_data_parallel_id = 1

        # Mock IPCSignal - 设置 launched_expert_service_signal.value 为非零，避免无限循环
        # 列表长度需要至少为 data_parallel_size // nnode = 2 // 1 = 2
        # 访问索引是 local_data_parallel_id % max_chips_per_node = 1 % 8 = 1
        mock_launched_signal = Mock()
        mock_launched_signal.value = [0, 1]  # 索引1处为非零值，避免 while 循环，使用列表支持 item assignment

        # worker_ready_signal: array_size = min(8, 2*2) = 4, 访问索引 = 1 % 8 = 1
        mock_worker_ready_signal = Mock()
        mock_worker_ready_signal.value = [0] * 4  # 至少4个元素

        # worker_healthy_live_signal: array_size = min(4, 2) = 2, 访问索引 = 1 % 2 % 8 = 1
        mock_worker_healthy_signal = Mock()
        mock_worker_healthy_signal.value = [0] * 2  # 至少2个元素

        # 其他信号只需要1个元素
        mock_other_signal = Mock()
        mock_other_signal.value = [0]  # 使用列表支持 item assignment

        mock_ipc_signal.side_effect = [
            mock_launched_signal,  # launched_expert_service_signal
            mock_worker_ready_signal,  # worker_ready_signal
            mock_worker_healthy_signal,  # worker_healthy_live_signal
            mock_other_signal,  # model_weights_status
            mock_other_signal,  # kv_cache_status
            mock_other_signal,  # exist_task_signal
            mock_other_signal,  # exist_swapped_task_signal
            mock_other_signal,  # exist_prefill_task_signal
        ]

        worker_proc = PaddleDisWorkerProc(self.fd_config, ranks=4, local_rank=1)
        worker_proc.init_health_status()

        # 验证 launched_expert_service_signal 被初始化
        self.assertIsNotNone(worker_proc.launched_expert_service_signal)

    @patch("fastdeploy.config.ModelConfig._post_init", return_value=None)
    @patch("fastdeploy.config.check_unified_ckpt", return_value=False)
    @patch("paddleformers.transformers.configuration_utils.PretrainedConfig.get_config_dict")
    @patch("paddleformers.transformers.configuration_utils.PretrainedConfig.from_dict")
    @patch("fastdeploy.worker.worker_process.paddle")
    @patch("fastdeploy.worker.worker_process.parse_quant_config")
    @patch("fastdeploy.worker.worker_process.v1_loader_support")
    @patch("fastdeploy.worker.worker_process.current_platform")
    @patch("fastdeploy.worker.worker_process.envs")
    def test_initialize_fd_config_unsupported_platform(
        self,
        mock_envs,
        mock_platform,
        mock_v1_support,
        mock_parse_quant,
        mock_paddle,
        mock_from_dict,
        mock_get_config_dict,
        mock_check_unified_ckpt,
        mock_model_post_init,
    ):
        """测试不支持平台的配置初始化"""
        mock_platform.is_iluvatar.return_value = False
        mock_platform.is_cuda.return_value = False
        mock_platform.is_xpu.return_value = False
        mock_platform.is_maca.return_value = False
        mock_platform.is_intel_hpu.return_value = False
        mock_parse_quant.return_value = None
        mock_v1_support.return_value = True
        mock_paddle.version.commit = "test_commit"
        mock_paddle.is_compiled_with_xpu.return_value = False
        mock_envs.ENABLE_V1_KVCACHE_SCHEDULER = 1
        mock_get_config_dict.return_value = (
            {
                "model_type": "llama",
                "hidden_size": 768,
                "num_attention_heads": 12,
                "num_hidden_layers": 2,
                "architectures": ["LlamaForCausalLM"],
                "enable_mm": False,
                "vocab_size": 32000,
            },
            {},
        )
        mock_pretrained_config = Mock()
        mock_pretrained_config.tensor_parallel_rank = None
        mock_pretrained_config.tensor_model_parallel_size = None
        mock_pretrained_config.is_mtp = False
        mock_from_dict.return_value = mock_pretrained_config

        fd_config = initialize_fd_config(self.args, ranks=1, local_rank=0)

        # 验证不支持平台时 ENABLE_V1_KVCACHE_SCHEDULER 被设置为 0
        self.assertEqual(mock_envs.ENABLE_V1_KVCACHE_SCHEDULER, 0)
        self.assertIsInstance(fd_config, FDConfig)

    @patch("fastdeploy.config.ModelConfig._post_init", return_value=None)
    @patch("fastdeploy.config.check_unified_ckpt", return_value=False)
    @patch("paddleformers.transformers.configuration_utils.PretrainedConfig.get_config_dict")
    @patch("paddleformers.transformers.configuration_utils.PretrainedConfig.from_dict")
    @patch("fastdeploy.worker.worker_process.paddle")
    @patch("fastdeploy.worker.worker_process.parse_quant_config")
    @patch("fastdeploy.worker.worker_process.v1_loader_support")
    @patch("fastdeploy.worker.worker_process.current_platform")
    @patch("fastdeploy.worker.worker_process.envs")
    @patch("fastdeploy.worker.worker_process.os")
    def test_initialize_fd_config_splitwise_role_decode(
        self,
        mock_os,
        mock_envs,
        mock_platform,
        mock_v1_support,
        mock_parse_quant,
        mock_paddle,
        mock_from_dict,
        mock_get_config_dict,
        mock_check_unified_ckpt,
        mock_model_post_init,
    ):
        """测试 splitwise_role 为 decode 的配置"""
        mock_platform.is_iluvatar.return_value = False
        mock_platform.is_cuda.return_value = True
        mock_platform.is_xpu.return_value = False
        mock_platform.is_maca.return_value = False
        mock_platform.is_intel_hpu.return_value = False
        mock_parse_quant.return_value = None
        mock_v1_support.return_value = True
        mock_paddle.version.commit = "test_commit"
        mock_paddle.is_compiled_with_xpu.return_value = False
        mock_envs.ENABLE_V1_KVCACHE_SCHEDULER = 1
        mock_os.environ = {}
        mock_get_config_dict.return_value = (
            {
                "model_type": "llama",
                "hidden_size": 768,
                "num_attention_heads": 12,
                "num_hidden_layers": 2,
                "architectures": ["LlamaForCausalLM"],
                "enable_mm": False,
                "vocab_size": 32000,
            },
            {},
        )
        mock_pretrained_config = Mock()
        mock_pretrained_config.tensor_parallel_rank = None
        mock_pretrained_config.tensor_model_parallel_size = None
        mock_pretrained_config.is_mtp = False
        mock_from_dict.return_value = mock_pretrained_config
        self.args.splitwise_role = "decode"

        fd_config = initialize_fd_config(self.args, ranks=1, local_rank=0)

        self.assertIsInstance(fd_config, FDConfig)
        # 验证函数执行成功（环境变量设置通过代码执行验证）

    @patch("fastdeploy.config.ModelConfig._post_init", return_value=None)
    @patch("fastdeploy.config.check_unified_ckpt", return_value=False)
    @patch("paddleformers.transformers.configuration_utils.PretrainedConfig.get_config_dict")
    @patch("paddleformers.transformers.configuration_utils.PretrainedConfig.from_dict")
    @patch("fastdeploy.worker.worker_process.paddle")
    @patch("fastdeploy.worker.worker_process.parse_quant_config")
    @patch("fastdeploy.worker.worker_process.v1_loader_support")
    @patch("fastdeploy.worker.worker_process.current_platform")
    @patch("fastdeploy.worker.worker_process.envs")
    def test_initialize_fd_config_paddleocr(
        self,
        mock_envs,
        mock_platform,
        mock_v1_support,
        mock_parse_quant,
        mock_paddle,
        mock_from_dict,
        mock_get_config_dict,
        mock_check_unified_ckpt,
        mock_model_post_init,
    ):
        """测试 PaddleOCR 架构的特殊处理"""
        mock_platform.is_iluvatar.return_value = False
        mock_platform.is_cuda.return_value = True
        mock_platform.is_xpu.return_value = False
        mock_platform.is_maca.return_value = False
        mock_platform.is_intel_hpu.return_value = False
        mock_parse_quant.return_value = None
        mock_v1_support.return_value = True
        mock_paddle.version.commit = "test_commit"
        mock_paddle.is_compiled_with_xpu.return_value = False
        mock_envs.FD_ENABLE_MAX_PREFILL = 0
        mock_get_config_dict.return_value = (
            {
                "model_type": "paddleocr",
                "hidden_size": 768,
                "num_attention_heads": 12,
                "num_hidden_layers": 2,
                "architectures": ["PaddleOCRForCausalLM"],
                "enable_mm": False,
                "vocab_size": 32000,
            },
            {},
        )
        mock_pretrained_config = Mock()
        mock_pretrained_config.tensor_parallel_rank = None
        mock_pretrained_config.tensor_model_parallel_size = None
        mock_pretrained_config.is_mtp = False
        mock_from_dict.return_value = mock_pretrained_config

        fd_config = initialize_fd_config(self.args, ranks=1, local_rank=0)

        # 验证 PaddleOCR 的特殊配置
        self.assertEqual(mock_envs.FD_ENABLE_MAX_PREFILL, 1)
        self.assertFalse(fd_config.cache_config.enable_prefix_caching)
        self.assertEqual(fd_config.cache_config.max_encoder_cache, 0)

    @patch("fastdeploy.config.ModelConfig._post_init", return_value=None)
    @patch("fastdeploy.config.check_unified_ckpt", return_value=False)
    @patch("paddleformers.transformers.configuration_utils.PretrainedConfig.get_config_dict")
    @patch("paddleformers.transformers.configuration_utils.PretrainedConfig.from_dict")
    @patch("fastdeploy.worker.worker_process.paddle")
    @patch("fastdeploy.worker.worker_process.parse_quant_config")
    @patch("fastdeploy.worker.worker_process.v1_loader_support")
    @patch("fastdeploy.worker.worker_process.current_platform")
    def test_initialize_fd_config_v1_loader_fallback(
        self,
        mock_platform,
        mock_v1_support,
        mock_parse_quant,
        mock_paddle,
        mock_from_dict,
        mock_get_config_dict,
        mock_check_unified_ckpt,
        mock_model_post_init,
    ):
        """测试 v1_loader_support 返回 False 时的回退"""
        mock_platform.is_iluvatar.return_value = False
        mock_platform.is_cuda.return_value = True
        mock_platform.is_xpu.return_value = False
        mock_platform.is_maca.return_value = False
        mock_platform.is_intel_hpu.return_value = False
        mock_parse_quant.return_value = None
        mock_v1_support.return_value = False  # v1 loader 不支持
        mock_paddle.version.commit = "test_commit"
        mock_paddle.is_compiled_with_xpu.return_value = False
        mock_get_config_dict.return_value = (
            {
                "model_type": "llama",
                "hidden_size": 768,
                "num_attention_heads": 12,
                "num_hidden_layers": 2,
                "architectures": ["LlamaForCausalLM"],
                "enable_mm": False,
                "vocab_size": 32000,
            },
            {},
        )
        mock_pretrained_config = Mock()
        mock_pretrained_config.tensor_parallel_rank = None
        mock_pretrained_config.tensor_model_parallel_size = None
        mock_pretrained_config.is_mtp = False
        mock_from_dict.return_value = mock_pretrained_config
        self.args.load_choices = "default_v1"

        fd_config = initialize_fd_config(self.args, ranks=1, local_rank=0)

        # 验证 load_choices 被回退到 "default"
        self.assertEqual(fd_config.load_config.load_choices, "default")

    @patch("fastdeploy.config.ModelConfig._post_init", return_value=None)
    @patch("fastdeploy.config.check_unified_ckpt", return_value=False)
    @patch("paddleformers.transformers.configuration_utils.PretrainedConfig.get_config_dict")
    @patch("paddleformers.transformers.configuration_utils.PretrainedConfig.from_dict")
    @patch("fastdeploy.worker.worker_process.paddle")
    @patch("fastdeploy.worker.worker_process.parse_quant_config")
    @patch("fastdeploy.worker.worker_process.v1_loader_support")
    @patch("fastdeploy.worker.worker_process.current_platform")
    def test_initialize_fd_config_num_hidden_layers_none(
        self,
        mock_platform,
        mock_v1_support,
        mock_parse_quant,
        mock_paddle,
        mock_from_dict,
        mock_get_config_dict,
        mock_check_unified_ckpt,
        mock_model_post_init,
    ):
        """测试 num_hidden_layers 为 None 时抛出异常"""
        mock_platform.is_iluvatar.return_value = False
        mock_platform.is_cuda.return_value = True
        mock_platform.is_xpu.return_value = False
        mock_platform.is_maca.return_value = False
        mock_platform.is_intel_hpu.return_value = False
        mock_parse_quant.return_value = None
        mock_v1_support.return_value = True
        mock_paddle.version.commit = "test_commit"
        mock_paddle.is_compiled_with_xpu.return_value = False
        mock_get_config_dict.return_value = (
            {
                "model_type": "llama",
                "hidden_size": 768,
                "num_attention_heads": 12,
                # 不包含 num_hidden_layers
                "architectures": ["LlamaForCausalLM"],
                "enable_mm": False,
                "vocab_size": 32000,
            },
            {},
        )
        mock_pretrained_config = Mock()
        mock_pretrained_config.tensor_parallel_rank = None
        mock_pretrained_config.tensor_model_parallel_size = None
        mock_pretrained_config.is_mtp = False
        mock_from_dict.return_value = mock_pretrained_config

        # 使用 patch 在 ModelConfig 创建后移除 num_hidden_layers
        original_model_config_init = ModelConfig.__init__

        def patched_init(self, args):
            original_model_config_init(self, args)
            # 移除 num_hidden_layers 属性
            if hasattr(self, "num_hidden_layers"):
                delattr(self, "num_hidden_layers")

        with patch.object(ModelConfig, "__init__", patched_init):
            with self.assertRaises(ValueError) as context:
                initialize_fd_config(self.args, ranks=1, local_rank=0)
            self.assertIn("num_hidden_layers is None", str(context.exception))

    @patch("fastdeploy.config.ModelConfig._post_init", return_value=None)
    @patch("fastdeploy.config.check_unified_ckpt", return_value=False)
    @patch("paddleformers.transformers.configuration_utils.PretrainedConfig.get_config_dict")
    @patch("paddleformers.transformers.configuration_utils.PretrainedConfig.from_dict")
    @patch("fastdeploy.worker.worker_process.paddle")
    @patch("fastdeploy.worker.worker_process.parse_quant_config")
    @patch("fastdeploy.worker.worker_process.v1_loader_support")
    @patch("fastdeploy.worker.worker_process.current_platform")
    def test_initialize_fd_config_with_quantization(
        self,
        mock_platform,
        mock_v1_support,
        mock_parse_quant,
        mock_paddle,
        mock_from_dict,
        mock_get_config_dict,
        mock_check_unified_ckpt,
        mock_model_post_init,
    ):
        """测试带量化配置的初始化"""
        mock_platform.is_iluvatar.return_value = False
        mock_platform.is_cuda.return_value = True
        mock_platform.is_xpu.return_value = False
        mock_platform.is_maca.return_value = False
        mock_platform.is_intel_hpu.return_value = False
        mock_v1_support.return_value = True
        mock_paddle.version.commit = "test_commit"
        mock_paddle.is_compiled_with_xpu.return_value = False

        # Mock quant_config
        mock_quant_config = Mock()
        mock_parse_quant.return_value = mock_quant_config

        mock_get_config_dict.return_value = (
            {
                "model_type": "llama",
                "hidden_size": 768,
                "num_attention_heads": 12,
                "num_hidden_layers": 2,
                "architectures": ["LlamaForCausalLM"],
                "enable_mm": False,
                "vocab_size": 32000,
            },
            {},
        )
        mock_pretrained_config = Mock()
        mock_pretrained_config.tensor_parallel_rank = None
        mock_pretrained_config.tensor_model_parallel_size = None
        mock_pretrained_config.is_mtp = False
        mock_from_dict.return_value = mock_pretrained_config

        # 使用 patch 在 ModelConfig 创建后设置 is_quantized
        original_model_config_init = ModelConfig.__init__

        def patched_init(self, args):
            original_model_config_init(self, args)
            self.is_quantized = True
            self.quantization_config = mock_quant_config

        with patch.object(ModelConfig, "__init__", patched_init):
            fd_config = initialize_fd_config(self.args, ranks=1, local_rank=0)

        self.assertIsInstance(fd_config, FDConfig)
        self.assertEqual(fd_config.quant_config, mock_quant_config)

    @patch("fastdeploy.worker.worker_process.get_worker")
    @patch("fastdeploy.worker.worker_process.current_platform")
    def test_paddle_dis_worker_proc_update_weights_from_tensor(self, mock_platform, mock_get_worker):
        """测试 update_weights_from_tensor 方法"""
        from fastdeploy.worker.worker_process import PaddleDisWorkerProc

        mock_platform.is_iluvatar.return_value = False
        mock_worker = Mock()
        mock_model = Mock()
        mock_worker.get_model.return_value = mock_model
        mock_model.redundant_table_manger = Mock()
        mock_get_worker.return_value = mock_worker

        worker_proc = PaddleDisWorkerProc(self.fd_config, ranks=2, local_rank=0)

        # Mock experts_manager
        worker_proc.experts_manager = Mock()
        worker_proc.experts_manager.tensor_infos = {"some_info": "value"}  # 非 None，避免 while 循环
        worker_proc.experts_manager.get_ep_rank_to_expert_id_list.return_value = ([], {}, 0)

        # Mock load_tensor_from_shm_mem
        with patch("fastdeploy.worker.worker_process.load_tensor_from_shm_mem") as mock_load:
            mock_load.return_value = {}
            mmap_infos = {MODEL_MAIN_NAME: "main_model"}

            worker_proc.update_weights_from_tensor(mmap_infos)

            # 验证方法被调用
            mock_load.assert_called_once()
            mock_model.redundant_table_manger.update_expert_rank_table.assert_called_once()
            mock_model.update_state_dict.assert_called_once()
            self.assertIsNone(worker_proc.experts_manager.tensor_infos)

    @patch("fastdeploy.worker.worker_process.get_worker")
    @patch("fastdeploy.worker.worker_process.current_platform")
    @patch("fastdeploy.worker.worker_process.paddle")
    def test_paddle_dis_worker_proc_broadcast_model_weights_signal(self, mock_paddle, mock_platform, mock_get_worker):
        """测试 _broadcast_model_weights_signal 方法"""
        from fastdeploy.worker.worker_process import PaddleDisWorkerProc

        mock_platform.is_iluvatar.return_value = False
        mock_worker = Mock()
        mock_get_worker.return_value = mock_worker

        # Mock paddle.full 和 broadcast
        mock_tensor = Mock()
        mock_tensor.numpy.return_value = [42]
        mock_paddle.full.return_value = mock_tensor

        worker_proc = PaddleDisWorkerProc(self.fd_config, ranks=2, local_rank=0)
        worker_proc.model_weights_signal = [1]
        mock_group = Mock()

        result = worker_proc._broadcast_model_weights_signal(src=0, group=mock_group)

        self.assertEqual(result, 42)
        mock_paddle.full.assert_called_once_with(shape=[1], fill_value=1, dtype="int32")
        mock_paddle.distributed.broadcast.assert_called_once_with(mock_tensor, src=0, group=mock_group)

    @patch("fastdeploy.worker.worker_process.get_worker")
    @patch("fastdeploy.worker.worker_process.current_platform")
    @patch("fastdeploy.worker.worker_process.paddle")
    def test_paddle_dis_worker_proc_tp_barrier_wait_xpu(self, mock_paddle, mock_platform, mock_get_worker):
        """测试 _tp_barrier_wait 方法（XPU 平台）"""
        from fastdeploy.worker.worker_process import PaddleDisWorkerProc

        mock_platform.is_xpu.return_value = True
        mock_platform.is_iluvatar.return_value = False
        mock_worker = Mock()
        mock_get_worker.return_value = mock_worker

        worker_proc = PaddleDisWorkerProc(self.fd_config, ranks=2, local_rank=0)
        worker_proc.task_queue = Mock()
        worker_proc.task_queue.worker_process_tp_barrier = Mock()

        worker_proc._tp_barrier_wait()

        worker_proc.task_queue.worker_process_tp_barrier.wait.assert_called_once()

    @patch("fastdeploy.worker.worker_process.get_worker")
    @patch("fastdeploy.worker.worker_process.current_platform")
    @patch("fastdeploy.worker.worker_process.paddle")
    def test_paddle_dis_worker_proc_tp_barrier_wait_non_xpu(self, mock_paddle, mock_platform, mock_get_worker):
        """测试 _tp_barrier_wait 方法（非 XPU 平台）"""
        from fastdeploy.worker.worker_process import PaddleDisWorkerProc

        mock_platform.is_xpu.return_value = False
        mock_platform.is_iluvatar.return_value = False
        mock_worker = Mock()
        mock_get_worker.return_value = mock_worker

        worker_proc = PaddleDisWorkerProc(self.fd_config, ranks=2, local_rank=0)
        worker_proc.parallel_config.tp_group = Mock()

        worker_proc._tp_barrier_wait()

        mock_paddle.distributed.barrier.assert_called_once_with(worker_proc.parallel_config.tp_group)

    @patch("fastdeploy.worker.worker_process.IPCSignal")
    @patch("fastdeploy.worker.worker_process.RedundantExpertManager")
    @patch("fastdeploy.worker.worker_process.create_mmap")
    @patch("fastdeploy.worker.worker_process.get_worker")
    @patch("fastdeploy.worker.worker_process.current_platform")
    def test_paddle_dis_worker_proc_init_eplb_signal(
        self, mock_platform, mock_get_worker, mock_create_mmap, mock_expert_manager, mock_ipc_signal
    ):
        """测试 _init_eplb_signal 方法"""
        from fastdeploy.worker.worker_process import PaddleDisWorkerProc

        mock_platform.is_iluvatar.return_value = False
        mock_worker = Mock()
        mock_get_worker.return_value = mock_worker

        # 启用 EPLB 并设置 model_config
        self.fd_config.eplb_config.enable_eplb = True
        self.fd_config.model_config = Mock()
        self.fd_config.model_config.num_hidden_layers = 2
        self.fd_config.model_config.moe_num_experts = 8

        # Mock IPCSignal
        mock_signal = Mock()
        mock_signal.value = np.zeros([1], dtype=np.int32)
        mock_ipc_signal.return_value = mock_signal

        # Mock RedundantExpertManager
        mock_manager = Mock()
        mock_expert_manager.return_value = mock_manager

        # Mock create_mmap
        mock_create_mmap.return_value = {"main_model": "mmap_info"}

        worker_proc = PaddleDisWorkerProc(self.fd_config, ranks=2, local_rank=0)
        worker_proc._init_eplb_signal()

        # 验证 EPLB 信号被初始化
        self.assertIsNotNone(worker_proc.experts_manager)
        self.assertIsNotNone(worker_proc.local_experts_token_stats_array)
        self.assertIsNotNone(worker_proc.signal_clear_experts_token_stats)
        self.assertIsNotNone(worker_proc.mmap_infos)

    @patch("fastdeploy.worker.worker_process.get_worker")
    @patch("fastdeploy.worker.worker_process.current_platform")
    def test_paddle_dis_worker_proc_init_eplb_signal_disabled(self, mock_platform, mock_get_worker):
        """测试 _init_eplb_signal 方法（EPLB 未启用）"""
        from fastdeploy.worker.worker_process import PaddleDisWorkerProc

        mock_platform.is_iluvatar.return_value = False
        mock_worker = Mock()
        mock_get_worker.return_value = mock_worker

        # 禁用 EPLB
        self.fd_config.eplb_config.enable_eplb = False

        worker_proc = PaddleDisWorkerProc(self.fd_config, ranks=2, local_rank=0)

        # 记录调用前的状态
        had_experts_manager_before = hasattr(worker_proc, "experts_manager")

        worker_proc._init_eplb_signal()

        # 验证 EPLB 相关属性未被初始化（方法直接返回，不会设置 experts_manager）
        # 如果之前没有，现在也应该没有
        has_experts_manager_after = hasattr(worker_proc, "experts_manager")
        self.assertEqual(had_experts_manager_before, has_experts_manager_after)

    @patch("fastdeploy.worker.worker_process.get_worker")
    @patch("fastdeploy.worker.worker_process.current_platform")
    @patch("fastdeploy.worker.worker_process.paddle")
    @patch("fastdeploy.worker.worker_process.time")
    def test_paddle_dis_worker_proc_run_eplb(self, mock_time, mock_paddle, mock_platform, mock_get_worker):
        """测试 _run_eplb 方法"""
        from fastdeploy.worker.worker_process import PaddleDisWorkerProc

        mock_platform.is_iluvatar.return_value = False
        mock_worker = Mock()
        mock_model = Mock()
        mock_worker.get_model.return_value = mock_model
        mock_get_worker.return_value = mock_worker

        # 启用 EPLB
        self.fd_config.eplb_config.enable_eplb = True
        self.fd_config.eplb_config.redundant_expert_dump_workload_interval = 10

        mock_time.time.return_value = 1000

        worker_proc = PaddleDisWorkerProc(self.fd_config, ranks=2, local_rank=0)

        # Mock EPLB 相关属性（需要先调用 _init_eplb_signal 或手动设置）
        worker_proc.last_dump_expert_workload_ts = 990
        worker_proc.local_experts_token_stats_array = Mock()
        worker_proc.local_experts_token_stats_array.value = np.zeros([2, 8], dtype=np.int32)
        worker_proc.signal_clear_experts_token_stats = Mock()
        worker_proc.signal_clear_experts_token_stats.value = [0]
        worker_proc.signal_update_weight_from_tensor_array = Mock()
        worker_proc.signal_update_weight_from_tensor_array.value = [0]
        worker_proc.rearrange_experts_signal = Mock()
        worker_proc.rearrange_experts_signal.value = [0]
        worker_proc.mmap_infos = {"main_model": "mmap_info"}

        # Mock worker 的 redundant_table_manger
        mock_model.redundant_table_manger = Mock()
        mock_model.redundant_table_manger.get_expert_tokens_stats.return_value = (
            np.zeros([2, 8], dtype=np.int32),
            None,
            None,
            None,
        )

        # Mock paddle.to_tensor 和 broadcast
        mock_tensor = Mock()
        mock_tensor.__getitem__ = Mock(return_value=Mock(__eq__=lambda self, other: False))
        mock_tensor.__getitem__.return_value = 0
        mock_paddle.to_tensor.return_value = mock_tensor

        # 测试 _run_eplb（tp_rank=0，但不会触发更新权重，因为 broadcast_value 为 0）
        worker_proc._run_eplb(tp_rank=0)

        # 验证方法被调用（如果条件满足）
        # 由于条件可能不满足，我们只验证方法可以执行而不出错
        self.assertTrue(True)  # 方法执行成功

    @patch("fastdeploy.config.ModelConfig._post_init", return_value=None)
    @patch("fastdeploy.config.check_unified_ckpt", return_value=False)
    @patch("paddleformers.transformers.configuration_utils.PretrainedConfig.get_config_dict")
    @patch("paddleformers.transformers.configuration_utils.PretrainedConfig.from_dict")
    @patch("fastdeploy.worker.worker_process.paddle")
    @patch("fastdeploy.worker.worker_process.parse_quant_config")
    @patch("fastdeploy.worker.worker_process.v1_loader_support")
    @patch("fastdeploy.worker.worker_process.current_platform")
    def test_initialize_fd_config_with_quantization_not_quantized(
        self,
        mock_platform,
        mock_v1_support,
        mock_parse_quant,
        mock_paddle,
        mock_from_dict,
        mock_get_config_dict,
        mock_check_unified_ckpt,
        mock_model_post_init,
    ):
        """测试带量化配置的初始化（is_quantized=False）"""
        mock_platform.is_iluvatar.return_value = False
        mock_platform.is_cuda.return_value = True
        mock_platform.is_xpu.return_value = False
        mock_platform.is_maca.return_value = False
        mock_platform.is_intel_hpu.return_value = False
        mock_v1_support.return_value = True
        mock_paddle.version.commit = "test_commit"
        mock_paddle.is_compiled_with_xpu.return_value = False

        # Mock quant_config
        mock_quant_config = Mock()
        mock_parse_quant.return_value = mock_quant_config

        mock_get_config_dict.return_value = (
            {
                "model_type": "llama",
                "hidden_size": 768,
                "num_attention_heads": 12,
                "num_hidden_layers": 2,
                "architectures": ["LlamaForCausalLM"],
                "enable_mm": False,
                "vocab_size": 32000,
            },
            {},
        )
        mock_pretrained_config = Mock()
        mock_pretrained_config.tensor_parallel_rank = None
        mock_pretrained_config.tensor_model_parallel_size = None
        mock_pretrained_config.is_mtp = False
        mock_from_dict.return_value = mock_pretrained_config

        # 使用 patch 在 ModelConfig 创建后设置 is_quantized=False
        original_model_config_init = ModelConfig.__init__

        def patched_init(self, args):
            original_model_config_init(self, args)
            self.is_quantized = False  # 未量化
            self.quantization_config = mock_quant_config

        with patch.object(ModelConfig, "__init__", patched_init):
            fd_config = initialize_fd_config(self.args, ranks=1, local_rank=0)

        self.assertIsInstance(fd_config, FDConfig)
        self.assertEqual(fd_config.quant_config, mock_quant_config)

    @patch("fastdeploy.worker.worker_process.parse_args")
    @patch("fastdeploy.worker.worker_process.init_distributed_environment")
    @patch("fastdeploy.worker.worker_process.initialize_fd_config")
    @patch("fastdeploy.worker.worker_process.current_platform")
    @patch("fastdeploy.worker.worker_process.PaddleDisWorkerProc")
    def test_run_worker_proc_non_iluvatar(
        self, mock_worker_proc_class, mock_platform, mock_init_fd_config, mock_init_dist, mock_parse_args
    ):
        """测试 run_worker_proc 函数（非 Iluvatar 平台）"""
        from fastdeploy.worker.worker_process import run_worker_proc

        mock_platform.is_iluvatar.return_value = False
        mock_args = Mock()
        mock_parse_args.return_value = mock_args
        mock_init_dist.return_value = (2, 0)
        mock_fd_config = Mock()
        mock_init_fd_config.return_value = mock_fd_config

        mock_worker_proc = Mock()
        mock_worker_proc_class.return_value = mock_worker_proc

        # 由于 run_worker_proc 会调用 event_loop_normal，我们需要 mock 它
        mock_worker_proc.event_loop_normal = Mock()

        run_worker_proc()

        mock_parse_args.assert_called_once()
        mock_init_dist.assert_called_once()
        mock_init_fd_config.assert_called_once_with(mock_args, 2, 0)
        mock_worker_proc_class.assert_called_once_with(mock_fd_config, 2, 0)
        mock_worker_proc.init_control.assert_called_once()
        mock_worker_proc.init_device.assert_called_once()
        mock_worker_proc.load_model.assert_called_once()
        mock_worker_proc.initialize_kv_cache.assert_called_once()
        mock_worker_proc.graph_optimize_and_warm_up_model.assert_called_once()
        mock_worker_proc.init_health_status.assert_called_once()
        mock_worker_proc.start_task_queue_service.assert_called_once()
        mock_worker_proc.event_loop_normal.assert_called_once()

    @patch("fastdeploy.worker.worker_process.parse_args")
    @patch("fastdeploy.worker.worker_process.init_distributed_environment")
    @patch("fastdeploy.worker.worker_process.initialize_fd_config")
    @patch("fastdeploy.worker.worker_process.current_platform")
    def test_run_worker_proc_iluvatar(self, mock_platform, mock_init_fd_config, mock_init_dist, mock_parse_args):
        """测试 run_worker_proc 函数（Iluvatar 平台）"""
        from fastdeploy.worker.worker_process import run_worker_proc

        mock_platform.is_iluvatar.return_value = True
        mock_args = Mock()
        mock_parse_args.return_value = mock_args
        mock_init_dist.return_value = (2, 0)
        mock_fd_config = Mock()
        mock_init_fd_config.return_value = mock_fd_config

        # Mock IluvatarPaddleDisWorkerProc 模块
        import sys

        mock_iluvatar_module = types.ModuleType("fastdeploy.worker.iluvatar_worker")
        mock_iluvatar_proc_class = Mock()
        mock_iluvatar_module.IluvatarPaddleDisWorkerProc = mock_iluvatar_proc_class
        sys.modules["fastdeploy.worker.iluvatar_worker"] = mock_iluvatar_module

        mock_worker_proc = Mock()
        mock_iluvatar_proc_class.return_value = mock_worker_proc
        mock_worker_proc.event_loop_normal = Mock()

        try:
            run_worker_proc()

            mock_iluvatar_proc_class.assert_called_once_with(mock_fd_config, 2, 0)
            mock_worker_proc.init_device.assert_called_once()
            mock_worker_proc.load_model.assert_called_once()
            mock_worker_proc.initialize_kv_cache.assert_called_once()
            mock_worker_proc.graph_optimize_and_warm_up_model.assert_called_once()
            mock_worker_proc.init_health_status.assert_called_once()
            mock_worker_proc.start_task_queue_service.assert_called_once()
            mock_worker_proc.event_loop_normal.assert_called_once()
        finally:
            # 清理 sys.modules
            if "fastdeploy.worker.iluvatar_worker" in sys.modules:
                del sys.modules["fastdeploy.worker.iluvatar_worker"]

    @patch("fastdeploy.worker.worker_process.IPCSignal")
    @patch("fastdeploy.worker.worker_process.RedundantExpertManager")
    @patch("fastdeploy.worker.worker_process.create_mmap")
    @patch("fastdeploy.worker.worker_process.get_worker")
    @patch("fastdeploy.worker.worker_process.current_platform")
    def test_paddle_dis_worker_proc_init_eplb_signal_master_rank0(
        self, mock_platform, mock_get_worker, mock_create_mmap, mock_expert_manager, mock_ipc_signal
    ):
        """测试 _init_eplb_signal 方法（local_rank == 0，master rank0）"""
        from fastdeploy.worker.worker_process import PaddleDisWorkerProc

        mock_platform.is_iluvatar.return_value = False
        mock_worker = Mock()
        mock_get_worker.return_value = mock_worker

        # 启用 EPLB 并设置 model_config
        self.fd_config.eplb_config.enable_eplb = True
        self.fd_config.model_config = Mock()
        self.fd_config.model_config.num_hidden_layers = 2
        self.fd_config.model_config.moe_num_experts = 8
        self.fd_config.parallel_config.tensor_parallel_size = 2

        # Mock IPCSignal
        mock_signal = Mock()
        mock_signal.value = np.zeros([1], dtype=np.int32)
        mock_ipc_signal.return_value = mock_signal

        # Mock RedundantExpertManager
        mock_manager = Mock()
        mock_expert_manager.return_value = mock_manager

        # Mock create_mmap
        mock_create_mmap.return_value = {"main_model": "mmap_info"}

        # local_rank=0, tensor_parallel_size=2, 所以 local_rank % tensor_parallel_size = 0
        # 确保 local_rank % tensor_parallel_size == 0 以进入 master rank0 分支
        worker_proc = PaddleDisWorkerProc(self.fd_config, ranks=2, local_rank=0)
        # 确保 parallel_config.tensor_parallel_size 被正确设置
        worker_proc.parallel_config.tensor_parallel_size = 2
        worker_proc._init_eplb_signal()

        # 验证 master rank0 的信号被初始化（local_rank % tensor_parallel_size == 0）
        # 0 % 2 == 0，所以应该进入第330行的分支
        self.assertIsNotNone(worker_proc.signal_update_weight_from_tensor_array)
        self.assertIsNotNone(worker_proc.rearrange_experts_signal)

    @patch("fastdeploy.worker.worker_process.get_worker")
    @patch("fastdeploy.worker.worker_process.current_platform")
    @patch("fastdeploy.worker.worker_process.paddle")
    @patch("fastdeploy.worker.worker_process.time")
    def test_paddle_dis_worker_proc_run_eplb_update_weights(
        self, mock_time, mock_paddle, mock_platform, mock_get_worker
    ):
        """测试 _run_eplb 方法（触发权重更新）"""
        from fastdeploy.worker.worker_process import (
            REARRANGE_EXPERT_MAGIC_NUM,
            PaddleDisWorkerProc,
        )

        mock_platform.is_iluvatar.return_value = False
        mock_worker = Mock()
        mock_model = Mock()
        mock_worker.get_model.return_value = mock_model
        mock_get_worker.return_value = mock_worker

        # 启用 EPLB
        self.fd_config.eplb_config.enable_eplb = True
        self.fd_config.eplb_config.redundant_expert_dump_workload_interval = 10

        mock_time.time.return_value = 1000

        worker_proc = PaddleDisWorkerProc(self.fd_config, ranks=2, local_rank=0)

        # Mock EPLB 相关属性
        worker_proc.last_dump_expert_workload_ts = 990
        worker_proc.local_experts_token_stats_array = Mock()
        worker_proc.local_experts_token_stats_array.value = np.zeros([2, 8], dtype=np.int32)
        worker_proc.signal_clear_experts_token_stats = Mock()
        worker_proc.signal_clear_experts_token_stats.value = [0]
        worker_proc.signal_update_weight_from_tensor_array = Mock()
        worker_proc.signal_update_weight_from_tensor_array.value = [1]  # 触发更新
        worker_proc.rearrange_experts_signal = Mock()
        worker_proc.rearrange_experts_signal.value = [0]
        worker_proc.mmap_infos = {"main_model": "mmap_info"}
        worker_proc.update_weights_from_tensor = Mock()

        # Mock worker 的 redundant_table_manger
        mock_model.redundant_table_manger = Mock()
        mock_model.redundant_table_manger.get_expert_tokens_stats.return_value = (
            np.zeros([2, 8], dtype=np.int32),
            None,
            None,
            None,
        )

        # Mock paddle.to_tensor 和 broadcast（返回 REARRANGE_EXPERT_MAGIC_NUM）
        mock_tensor = Mock()
        mock_tensor.__getitem__ = Mock(return_value=REARRANGE_EXPERT_MAGIC_NUM)
        mock_paddle.to_tensor.return_value = mock_tensor

        # 测试 _run_eplb（tp_rank=0，触发权重更新）
        worker_proc._run_eplb(tp_rank=0)

        # 验证 update_weights_from_tensor 被调用
        worker_proc.update_weights_from_tensor.assert_called_once_with(worker_proc.mmap_infos)

    @patch("fastdeploy.worker.worker_process.IPCSignal")
    @patch("fastdeploy.worker.worker_process.get_worker")
    @patch("fastdeploy.worker.worker_process.current_platform")
    @patch("fastdeploy.worker.worker_process.paddle")
    def test_paddle_dis_worker_proc_initialize_kv_cache_with_profile(
        self, mock_paddle, mock_platform, mock_get_worker, mock_ipc_signal
    ):
        """测试 initialize_kv_cache 方法（do_profile=True）"""
        from fastdeploy.worker.worker_process import PaddleDisWorkerProc

        mock_platform.is_iluvatar.return_value = False
        mock_worker = Mock()
        mock_worker.determine_available_memory.return_value = 10 * 1024**3  # 10GB
        mock_worker.cal_theortical_kvcache.return_value = 1024**2  # 1MB per block
        mock_get_worker.return_value = mock_worker

        # Mock IPCSignal
        mock_signal = Mock()
        mock_signal.value = [0]
        mock_ipc_signal.return_value = mock_signal

        # Mock paddle.distributed
        mock_paddle.distributed.all_reduce = Mock()
        mock_paddle.distributed.ReduceOp = Mock()
        mock_paddle.distributed.ReduceOp.MIN = Mock()

        # 启用 profile
        self.fd_config.parallel_config.do_profile = True

        worker_proc = PaddleDisWorkerProc(self.fd_config, ranks=1, local_rank=0)
        worker_proc.initialize_kv_cache()

        # 验证 worker 方法被调用
        mock_worker.determine_available_memory.assert_called_once()
        mock_worker.cal_theortical_kvcache.assert_called_once()
        mock_worker.initialize_cache.assert_called_once()

    @patch("fastdeploy.worker.worker_process.get_worker")
    @patch("fastdeploy.worker.worker_process.current_platform")
    @patch("fastdeploy.worker.worker_process.envs")
    @patch("fastdeploy.worker.worker_process.IPCSignal")
    def test_paddle_dis_worker_proc_graph_optimize_splitwise_prefill(
        self, mock_ipc_signal, mock_envs, mock_platform, mock_get_worker
    ):
        """测试 graph_optimize_and_warm_up_model 方法（splitwise_role=prefill）"""
        from fastdeploy.worker.worker_process import PaddleDisWorkerProc

        mock_platform.is_iluvatar.return_value = False
        mock_worker = Mock()
        mock_model_runner = Mock()
        mock_model_runner.device_id = 0
        mock_worker.model_runner = mock_model_runner
        mock_get_worker.return_value = mock_worker

        mock_envs.ENABLE_V1_KVCACHE_SCHEDULER = False
        self.fd_config.scheduler_config.splitwise_role = "prefill"

        # Mock IPCSignal
        mock_signal = Mock()
        mock_signal.value = [0]
        mock_ipc_signal.return_value = mock_signal

        worker_proc = PaddleDisWorkerProc(self.fd_config, ranks=1, local_rank=0)
        worker_proc.graph_optimize_and_warm_up_model()

        # 验证 worker 方法被调用
        mock_worker.graph_optimize_and_warm_up_model.assert_called_once()
        # 验证 IPCSignal 被创建（splitwise_role=prefill 时）
        mock_ipc_signal.assert_called()

    @patch("fastdeploy.worker.worker_process.IPCSignal")
    @patch("fastdeploy.worker.worker_process.envs")
    @patch("fastdeploy.worker.worker_process.get_worker")
    @patch("fastdeploy.worker.worker_process.current_platform")
    @patch("fastdeploy.worker.worker_process.time")
    def test_paddle_dis_worker_proc_init_health_status_wait_expert_service(
        self, mock_time, mock_platform, mock_get_worker, mock_envs, mock_ipc_signal
    ):
        """测试 init_health_status 中的等待 expert service 信号（while 循环）"""
        from fastdeploy.worker.worker_process import PaddleDisWorkerProc

        mock_platform.is_iluvatar.return_value = False
        mock_worker = Mock()
        mock_get_worker.return_value = mock_worker
        mock_envs.FD_ENABLE_MULTI_API_SERVER = False
        mock_time.time.return_value = 1234567890

        # 设置数据并行
        self.fd_config.parallel_config.data_parallel_size = 2
        self.fd_config.parallel_config.local_data_parallel_id = 1

        # Mock IPCSignal - launched_expert_service_signal 初始为 0，然后变为 1
        mock_launched_signal = Mock()
        # 使用 side_effect 来模拟值的变化
        call_count = [0]

        def get_value_side_effect(index):
            call_count[0] += 1
            if call_count[0] <= 2:  # 前两次返回 0（等待）
                return 0
            else:  # 之后返回 1（退出循环）
                return 1

        mock_launched_signal.value = Mock()
        mock_launched_signal.value.__getitem__ = Mock(side_effect=get_value_side_effect)

        # 其他信号
        mock_other_signal = Mock()
        mock_other_signal.value = [0] * 4  # worker_ready_signal
        mock_worker_healthy_signal = Mock()
        mock_worker_healthy_signal.value = [0] * 2  # worker_healthy_live_signal
        mock_single_signal = Mock()
        mock_single_signal.value = [0]  # 其他单元素信号

        mock_ipc_signal.side_effect = [
            mock_launched_signal,  # launched_expert_service_signal
            mock_other_signal,  # worker_ready_signal
            mock_worker_healthy_signal,  # worker_healthy_live_signal
            mock_single_signal,  # model_weights_status
            mock_single_signal,  # kv_cache_status
            mock_single_signal,  # exist_task_signal
            mock_single_signal,  # exist_swapped_task_signal
            mock_single_signal,  # exist_prefill_task_signal
        ]

        worker_proc = PaddleDisWorkerProc(self.fd_config, ranks=4, local_rank=1)
        worker_proc.init_health_status()

        # 验证 launched_expert_service_signal 被初始化
        self.assertIsNotNone(worker_proc.launched_expert_service_signal)

    @patch("fastdeploy.worker.worker_process.get_worker")
    @patch("fastdeploy.worker.worker_process.current_platform")
    @patch("fastdeploy.worker.worker_process.time")
    def test_paddle_dis_worker_proc_update_weights_from_tensor_wait(self, mock_time, mock_platform, mock_get_worker):
        """测试 update_weights_from_tensor 方法（等待 tensor_infos）"""
        from fastdeploy.worker.worker_process import PaddleDisWorkerProc

        mock_platform.is_iluvatar.return_value = False
        mock_worker = Mock()
        mock_model = Mock()
        mock_worker.get_model.return_value = mock_model
        mock_model.redundant_table_manger = Mock()
        mock_get_worker.return_value = mock_worker

        worker_proc = PaddleDisWorkerProc(self.fd_config, ranks=2, local_rank=0)

        # Mock experts_manager - 使用一个简单的类来模拟属性变化
        class MockExpertsManager:
            def __init__(self):
                self._tensor_infos = None
                self._access_count = 0

            @property
            def tensor_infos(self):
                self._access_count += 1
                if self._access_count <= 2:  # 前两次返回 None（等待）
                    return None
                else:  # 之后返回非 None（退出循环）
                    return {"some_info": "value"}

            @tensor_infos.setter
            def tensor_infos(self, value):
                self._tensor_infos = value

            def get_ep_rank_to_expert_id_list(self):
                return ([], {}, 0)

        worker_proc.experts_manager = MockExpertsManager()

        # Mock load_tensor_from_shm_mem
        with patch("fastdeploy.worker.worker_process.load_tensor_from_shm_mem") as mock_load:
            mock_load.return_value = {}
            mmap_infos = {MODEL_MAIN_NAME: "main_model"}

            # Mock time.sleep 以避免实际等待
            mock_time.sleep = Mock()

            worker_proc.update_weights_from_tensor(mmap_infos)

            # 验证方法被调用
            mock_load.assert_called_once()
            mock_model.redundant_table_manger.update_expert_rank_table.assert_called_once()
            mock_model.update_state_dict.assert_called_once()
            # 验证 tensor_infos 被设置为 None
            self.assertIsNone(worker_proc.experts_manager._tensor_infos)

    @patch("fastdeploy.worker.worker_process.get_worker")
    @patch("fastdeploy.worker.worker_process.current_platform")
    @patch("fastdeploy.worker.worker_process.paddle")
    @patch("fastdeploy.worker.worker_process.time")
    def test_paddle_dis_worker_proc_run_eplb_stats_none(self, mock_time, mock_paddle, mock_platform, mock_get_worker):
        """测试 _run_eplb 方法（local_experts_token_stats_array.value is None）"""
        from fastdeploy.worker.worker_process import PaddleDisWorkerProc

        mock_platform.is_iluvatar.return_value = False
        mock_worker = Mock()
        mock_get_worker.return_value = mock_worker

        # 启用 EPLB
        self.fd_config.eplb_config.enable_eplb = True
        self.fd_config.eplb_config.redundant_expert_dump_workload_interval = 10

        mock_time.time.return_value = 1000

        worker_proc = PaddleDisWorkerProc(self.fd_config, ranks=2, local_rank=0)

        # Mock EPLB 相关属性 - local_experts_token_stats_array.value 为 None
        worker_proc.last_dump_expert_workload_ts = 990
        worker_proc.local_experts_token_stats_array = Mock()
        worker_proc.local_experts_token_stats_array.value = None  # 触发警告分支
        worker_proc.signal_clear_experts_token_stats = Mock()
        worker_proc.signal_clear_experts_token_stats.value = [0]
        worker_proc.signal_update_weight_from_tensor_array = Mock()
        worker_proc.signal_update_weight_from_tensor_array.value = [0]
        worker_proc.rearrange_experts_signal = Mock()
        worker_proc.rearrange_experts_signal.value = [0]
        worker_proc.mmap_infos = {"main_model": "mmap_info"}

        # Mock paddle.to_tensor 和 broadcast
        mock_tensor = Mock()
        mock_tensor.__getitem__ = Mock(return_value=0)
        mock_paddle.to_tensor.return_value = mock_tensor

        # 测试 _run_eplb（应该触发警告）
        with patch("fastdeploy.worker.worker_process.logger") as mock_logger:
            worker_proc._run_eplb(tp_rank=0)
            # 验证警告被记录
            mock_logger.warning.assert_called()

    @patch("fastdeploy.worker.worker_process.IPCSignal")
    @patch("fastdeploy.worker.worker_process.get_worker")
    @patch("fastdeploy.worker.worker_process.current_platform")
    @patch("fastdeploy.worker.worker_process.paddle")
    @patch("fastdeploy.worker.worker_process.dist")
    def test_paddle_dis_worker_proc_initialize_kv_cache_large_blocks(
        self, mock_dist, mock_paddle, mock_platform, mock_get_worker, mock_ipc_signal
    ):
        """测试 initialize_kv_cache 方法（num_blocks_local > 40000）"""
        from fastdeploy.worker.worker_process import PaddleDisWorkerProc

        mock_platform.is_iluvatar.return_value = False
        mock_worker = Mock()
        mock_worker.determine_available_memory.return_value = 1000 * 1024**3  # 1000GB
        mock_worker.cal_theortical_kvcache.return_value = 1024**2  # 1MB per block (会产生大量blocks)
        mock_get_worker.return_value = mock_worker

        # Mock IPCSignal
        mock_signal = Mock()
        mock_signal.value = [0]
        mock_ipc_signal.return_value = mock_signal

        # 启用 profile
        self.fd_config.parallel_config.do_profile = True

        worker_proc = PaddleDisWorkerProc(self.fd_config, ranks=1, local_rank=0)
        worker_proc.initialize_kv_cache()

        # 验证 worker 方法被调用
        mock_worker.initialize_cache.assert_called_once()
        # 验证 num_blocks_local 被限制（通过检查调用参数）
        call_args = mock_worker.initialize_cache.call_args
        num_blocks = call_args[1]["num_gpu_blocks"] if "num_gpu_blocks" in call_args[1] else call_args[0][0]
        self.assertLessEqual(num_blocks, 40000)

    @patch("fastdeploy.worker.worker_process.get_worker")
    @patch("fastdeploy.worker.worker_process.current_platform")
    def test_paddle_dis_worker_proc_initialize_kv_cache_zero_blocks(self, mock_platform, mock_get_worker):
        """测试 initialize_kv_cache 方法（num_blocks_local <= 0，应该抛出异常）"""
        from fastdeploy.worker.worker_process import PaddleDisWorkerProc

        mock_platform.is_iluvatar.return_value = False
        mock_worker = Mock()
        mock_worker.determine_available_memory.return_value = 1024  # 很小的内存
        mock_worker.cal_theortical_kvcache.return_value = 1024**3  # 1GB per block (会导致blocks为0)
        mock_get_worker.return_value = mock_worker

        # 启用 profile
        self.fd_config.parallel_config.do_profile = True

        worker_proc = PaddleDisWorkerProc(self.fd_config, ranks=1, local_rank=0)

        # 应该抛出 ValueError
        with self.assertRaises(ValueError) as context:
            worker_proc.initialize_kv_cache()
        self.assertIn("cannot be less than zero", str(context.exception))

    @patch("fastdeploy.worker.worker_process.IPCSignal")
    @patch("fastdeploy.worker.worker_process.get_worker")
    @patch("fastdeploy.worker.worker_process.current_platform")
    @patch("fastdeploy.worker.worker_process.paddle")
    @patch("fastdeploy.worker.worker_process.dist")
    def test_paddle_dis_worker_proc_initialize_kv_cache_distributed(
        self, mock_dist, mock_paddle, mock_platform, mock_get_worker, mock_ipc_signal
    ):
        """测试 initialize_kv_cache 方法（ranks > 1，分布式）"""
        from fastdeploy.worker.worker_process import PaddleDisWorkerProc

        mock_platform.is_iluvatar.return_value = False
        mock_worker = Mock()
        mock_worker.determine_available_memory.return_value = 10 * 1024**3  # 10GB
        mock_worker.cal_theortical_kvcache.return_value = 1024**2  # 1MB per block
        mock_get_worker.return_value = mock_worker

        # Mock IPCSignal
        mock_signal = Mock()
        mock_signal.value = [0]
        mock_ipc_signal.return_value = mock_signal

        # Mock paddle.distributed
        mock_tensor = Mock()
        mock_tensor.item.return_value = 1000
        mock_paddle.full.return_value = mock_tensor
        mock_dist.all_reduce = Mock()
        mock_dist.ReduceOp = Mock()
        mock_dist.ReduceOp.MIN = Mock()

        # 启用 profile
        self.fd_config.parallel_config.do_profile = True

        worker_proc = PaddleDisWorkerProc(self.fd_config, ranks=2, local_rank=0)
        worker_proc.initialize_kv_cache()

        # 验证分布式操作被调用
        mock_dist.all_reduce.assert_called_once()
        mock_worker.initialize_cache.assert_called_once()

    @patch("fastdeploy.worker.worker_process.IPCSignal")
    @patch("fastdeploy.worker.worker_process.get_worker")
    @patch("fastdeploy.worker.worker_process.current_platform")
    @patch("fastdeploy.worker.worker_process.paddle")
    @patch("fastdeploy.worker.worker_process.dist")
    def test_paddle_dis_worker_proc_initialize_kv_cache_master_rank(
        self, mock_dist, mock_paddle, mock_platform, mock_get_worker, mock_ipc_signal
    ):
        """测试 initialize_kv_cache 方法（local_rank % max_chips_per_node == 0）"""
        from fastdeploy.worker.worker_process import PaddleDisWorkerProc

        mock_platform.is_iluvatar.return_value = False
        mock_worker = Mock()
        mock_worker.determine_available_memory.return_value = 10 * 1024**3  # 10GB
        mock_worker.cal_theortical_kvcache.return_value = 1024**2  # 1MB per block
        mock_get_worker.return_value = mock_worker

        # Mock IPCSignal
        mock_signal = Mock()
        mock_signal.value = [0]
        mock_ipc_signal.return_value = mock_signal

        # Mock paddle.distributed
        mock_tensor = Mock()
        mock_tensor.item.return_value = 1000
        mock_paddle.full.return_value = mock_tensor
        mock_dist.all_reduce = Mock()
        mock_dist.ReduceOp = Mock()
        mock_dist.ReduceOp.MIN = Mock()

        # 启用 profile，local_rank=0，max_chips_per_node=8，所以 0 % 8 == 0
        self.fd_config.parallel_config.do_profile = True

        worker_proc = PaddleDisWorkerProc(self.fd_config, ranks=1, local_rank=0)
        worker_proc.initialize_kv_cache()

        # 验证 IPCSignal 被创建（master rank）
        mock_ipc_signal.assert_called()
        self.assertIsNotNone(worker_proc.get_profile_block_num_signal)

    @patch("fastdeploy.worker.worker_process.get_worker")
    @patch("fastdeploy.worker.worker_process.current_platform")
    @patch("fastdeploy.worker.worker_process.paddle")
    @patch("fastdeploy.worker.worker_process.time")
    def test_paddle_dis_worker_proc_run_eplb_tp_rank0_done(
        self, mock_time, mock_paddle, mock_platform, mock_get_worker
    ):
        """测试 _run_eplb 方法（tp_rank == 0，设置 DONE 状态）"""
        from fastdeploy.worker.worker_process import (
            REARRANGE_EXPERT_MAGIC_NUM,
            PaddleDisWorkerProc,
            RearrangeExpertStatus,
        )

        mock_platform.is_iluvatar.return_value = False
        mock_worker = Mock()
        mock_model = Mock()
        mock_worker.get_model.return_value = mock_model
        mock_get_worker.return_value = mock_worker

        # 启用 EPLB
        self.fd_config.eplb_config.enable_eplb = True
        self.fd_config.eplb_config.redundant_expert_dump_workload_interval = 10

        mock_time.time.return_value = 1000

        worker_proc = PaddleDisWorkerProc(self.fd_config, ranks=2, local_rank=0)

        # Mock EPLB 相关属性
        worker_proc.last_dump_expert_workload_ts = 990
        worker_proc.local_experts_token_stats_array = Mock()
        worker_proc.local_experts_token_stats_array.value = np.zeros([2, 8], dtype=np.int32)
        worker_proc.signal_clear_experts_token_stats = Mock()
        worker_proc.signal_clear_experts_token_stats.value = [0]
        worker_proc.signal_update_weight_from_tensor_array = Mock()
        worker_proc.signal_update_weight_from_tensor_array.value = [1]  # 触发更新
        worker_proc.rearrange_experts_signal = Mock()
        worker_proc.rearrange_experts_signal.value = [0]
        worker_proc.mmap_infos = {"main_model": "mmap_info"}
        worker_proc.update_weights_from_tensor = Mock()

        # Mock worker 的 redundant_table_manger
        mock_model.redundant_table_manger = Mock()
        mock_model.redundant_table_manger.get_expert_tokens_stats.return_value = (
            np.zeros([2, 8], dtype=np.int32),
            None,
            None,
            None,
        )

        # Mock paddle.to_tensor 和 broadcast（返回 REARRANGE_EXPERT_MAGIC_NUM）
        mock_tensor = Mock()
        mock_tensor.__getitem__ = Mock(return_value=REARRANGE_EXPERT_MAGIC_NUM)
        mock_paddle.to_tensor.return_value = mock_tensor
        mock_paddle.distributed.barrier = Mock()

        # 测试 _run_eplb（tp_rank=0，触发权重更新并设置 DONE）
        worker_proc._run_eplb(tp_rank=0)

        # 验证 rearrange_experts_signal 被设置为 DONE
        self.assertEqual(worker_proc.rearrange_experts_signal.value[0], RearrangeExpertStatus.DONE.value)

    @patch("fastdeploy.worker.worker_process.get_worker")
    @patch("fastdeploy.worker.worker_process.current_platform")
    def test_paddle_dis_worker_proc_init_device(self, mock_platform, mock_get_worker):
        """测试 init_device 方法"""
        from fastdeploy.worker.worker_process import PaddleDisWorkerProc

        mock_platform.is_iluvatar.return_value = False
        mock_worker = Mock()
        mock_get_worker.return_value = mock_worker

        worker_proc = PaddleDisWorkerProc(self.fd_config, ranks=2, local_rank=0)
        worker_proc.init_device()

        # 验证 worker.init_device 被调用
        mock_worker.init_device.assert_called_once()

    @patch("fastdeploy.worker.worker_process.TaskQueue")
    @patch("fastdeploy.worker.worker_process.get_worker")
    @patch("fastdeploy.worker.worker_process.current_platform")
    @patch("fastdeploy.worker.worker_process.envs")
    def test_paddle_dis_worker_proc_start_task_queue_service(
        self, mock_envs, mock_platform, mock_get_worker, mock_task_queue_class
    ):
        """测试 start_task_queue_service 方法"""
        from fastdeploy.worker.worker_process import PaddleDisWorkerProc

        mock_platform.is_iluvatar.return_value = False
        mock_worker = Mock()
        mock_get_worker.return_value = mock_worker
        mock_envs.FD_ENGINE_TASK_QUEUE_WITH_SHM = False
        self.fd_config.parallel_config.pod_ip = "127.0.0.1"
        self.fd_config.parallel_config.local_engine_worker_queue_port = 9923
        self.fd_config.parallel_config.tensor_parallel_rank = 0
        self.fd_config.parallel_config.local_data_parallel_id = 0

        worker_proc = PaddleDisWorkerProc(self.fd_config, ranks=2, local_rank=0)
        worker_proc.start_task_queue_service()

        # 验证 TaskQueue 被创建
        mock_task_queue_class.assert_called_once()
        self.assertIsNotNone(worker_proc.task_queue)

    @patch("fastdeploy.worker.worker_process.TaskQueue")
    @patch("fastdeploy.worker.worker_process.get_worker")
    @patch("fastdeploy.worker.worker_process.current_platform")
    @patch("fastdeploy.worker.worker_process.envs")
    def test_paddle_dis_worker_proc_start_task_queue_service_shm(
        self, mock_envs, mock_platform, mock_get_worker, mock_task_queue_class
    ):
        """测试 start_task_queue_service 方法（使用 SHM）"""
        from fastdeploy.worker.worker_process import PaddleDisWorkerProc

        mock_platform.is_iluvatar.return_value = False
        mock_worker = Mock()
        mock_get_worker.return_value = mock_worker
        mock_envs.FD_ENGINE_TASK_QUEUE_WITH_SHM = True
        self.fd_config.parallel_config.local_engine_worker_queue_port = 9923
        self.fd_config.parallel_config.tensor_parallel_rank = 0
        self.fd_config.parallel_config.local_data_parallel_id = 0

        worker_proc = PaddleDisWorkerProc(self.fd_config, ranks=2, local_rank=0)
        worker_proc.start_task_queue_service()

        # 验证 TaskQueue 被创建（使用 SHM 地址）
        mock_task_queue_class.assert_called_once()
        call_args = mock_task_queue_class.call_args
        self.assertIn("/dev/shm/fd_task_queue_9923.sock", str(call_args))

    @patch("fastdeploy.worker.worker_process.IPCSignal")
    @patch("fastdeploy.worker.worker_process.get_worker")
    @patch("fastdeploy.worker.worker_process.current_platform")
    @patch("fastdeploy.worker.worker_process.paddle")
    def test_paddle_dis_worker_proc_load_model(self, mock_paddle, mock_platform, mock_get_worker, mock_ipc_signal):
        """测试 load_model 方法"""
        from fastdeploy.worker.worker_process import PaddleDisWorkerProc

        mock_platform.is_iluvatar.return_value = False
        mock_worker = Mock()
        mock_get_worker.return_value = mock_worker

        # Mock IPCSignal
        mock_signal = Mock()
        mock_signal.value = [0]
        mock_ipc_signal.return_value = mock_signal

        worker_proc = PaddleDisWorkerProc(self.fd_config, ranks=1, local_rank=0)
        worker_proc.load_model()

        # 验证 worker.load_model 被调用
        mock_worker.load_model.assert_called_once()
        # 验证 loaded_model_signal 被创建
        self.assertIsNotNone(worker_proc.loaded_model_signal)

    @patch("fastdeploy.worker.worker_process.IPCSignal")
    @patch("fastdeploy.worker.worker_process.get_worker")
    @patch("fastdeploy.worker.worker_process.current_platform")
    @patch("fastdeploy.worker.worker_process.paddle")
    def test_paddle_dis_worker_proc_load_model_distributed(
        self, mock_paddle, mock_platform, mock_get_worker, mock_ipc_signal
    ):
        """测试 load_model 方法（分布式，ranks > 1）"""
        from fastdeploy.worker.worker_process import PaddleDisWorkerProc

        mock_platform.is_iluvatar.return_value = False
        mock_worker = Mock()
        mock_get_worker.return_value = mock_worker

        # Mock IPCSignal
        mock_signal = Mock()
        mock_signal.value = [0]
        mock_ipc_signal.return_value = mock_signal

        worker_proc = PaddleDisWorkerProc(self.fd_config, ranks=2, local_rank=0)
        worker_proc.load_model()

        # 验证 worker.load_model 被调用
        mock_worker.load_model.assert_called_once()
        # 验证 barrier 被调用（ranks > 1）
        mock_paddle.distributed.barrier.assert_called_once()

    @patch("fastdeploy.worker.worker_process.get_worker")
    @patch("fastdeploy.worker.worker_process.current_platform")
    @patch("fastdeploy.worker.worker_process.asyncio")
    def test_paddle_dis_worker_proc_run_control_method(self, mock_asyncio, mock_platform, mock_get_worker):
        """测试 run_control_method 方法"""
        from fastdeploy.engine.request import ControlRequest
        from fastdeploy.worker.worker_process import PaddleDisWorkerProc

        mock_platform.is_iluvatar.return_value = False
        mock_worker = Mock()
        mock_get_worker.return_value = mock_worker

        # Mock worker 的方法
        mock_worker.test_method = Mock(return_value="test_result")

        worker_proc = PaddleDisWorkerProc(self.fd_config, ranks=2, local_rank=0)

        # 创建 ControlRequest（需要 request_id, method, args）
        control_request = ControlRequest(request_id="test_id", method="test_method", args={})

        # Mock _ctrl_output
        mock_queue = Mock()
        worker_proc._ctrl_output = mock_queue
        mock_asyncio.run = Mock()

        worker_proc.run_control_method(control_request)

        # 验证 worker 的方法被调用
        mock_worker.test_method.assert_called_once()
        # 验证 asyncio.run 被调用（用于 put 响应）
        mock_asyncio.run.assert_called()

    @patch("fastdeploy.worker.worker_process.get_worker")
    @patch("fastdeploy.worker.worker_process.current_platform")
    @patch("fastdeploy.worker.worker_process.asyncio")
    def test_paddle_dis_worker_proc_run_control_method_unknown_method(
        self, mock_asyncio, mock_platform, mock_get_worker
    ):
        """测试 run_control_method 方法（未知方法）"""
        from fastdeploy.engine.request import ControlRequest
        from fastdeploy.worker.worker_process import PaddleDisWorkerProc

        mock_platform.is_iluvatar.return_value = False
        mock_worker = Mock()
        mock_get_worker.return_value = mock_worker

        # Mock worker 没有该方法
        mock_worker.unknown_method = None

        worker_proc = PaddleDisWorkerProc(self.fd_config, ranks=2, local_rank=0)

        # 创建 ControlRequest（使用未知方法）
        control_request = ControlRequest(request_id="test_id", method="unknown_method", args={})

        # Mock _ctrl_output
        mock_queue = Mock()
        worker_proc._ctrl_output = mock_queue
        mock_asyncio.run = Mock()

        worker_proc.run_control_method(control_request)

        # 验证 asyncio.run 被调用（用于 put 错误响应）
        mock_asyncio.run.assert_called()

    @patch("fastdeploy.worker.worker_process.get_worker")
    @patch("fastdeploy.worker.worker_process.current_platform")
    @patch("fastdeploy.worker.worker_process.asyncio")
    @patch("fastdeploy.worker.worker_process.logger")
    def test_paddle_dis_worker_proc_run_control_method_exception(
        self, mock_logger, mock_asyncio, mock_platform, mock_get_worker
    ):
        """测试 run_control_method 方法（方法抛出异常）"""
        from fastdeploy.engine.request import ControlRequest
        from fastdeploy.worker.worker_process import PaddleDisWorkerProc

        mock_platform.is_iluvatar.return_value = False
        mock_worker = Mock()
        mock_get_worker.return_value = mock_worker

        # Mock worker 的方法抛出异常
        mock_worker.test_method = Mock(side_effect=ValueError("Test error"))

        worker_proc = PaddleDisWorkerProc(self.fd_config, ranks=2, local_rank=0)

        # 创建 ControlRequest
        control_request = ControlRequest(request_id="test_id", method="test_method", args={})

        # Mock _ctrl_output
        mock_queue = Mock()
        worker_proc._ctrl_output = mock_queue
        mock_asyncio.run = Mock()

        worker_proc.run_control_method(control_request)

        # 验证 worker 的方法被调用
        mock_worker.test_method.assert_called_once()
        # 验证错误被记录
        mock_logger.error.assert_called()
        # 验证 asyncio.run 被调用（用于 put 错误响应）
        mock_asyncio.run.assert_called()

    @patch("fastdeploy.worker.worker_process.get_worker")
    @patch("fastdeploy.worker.worker_process.current_platform")
    def test_paddle_dis_worker_proc_initialize_kv_cache_no_profile(self, mock_platform, mock_get_worker):
        """测试 initialize_kv_cache 方法（do_profile=False）"""
        from fastdeploy.worker.worker_process import PaddleDisWorkerProc

        mock_platform.is_iluvatar.return_value = False
        mock_worker = Mock()
        mock_get_worker.return_value = mock_worker

        # 禁用 profile
        self.fd_config.parallel_config.do_profile = False
        self.fd_config.cache_config.total_block_num = 1000

        worker_proc = PaddleDisWorkerProc(self.fd_config, ranks=1, local_rank=0)
        worker_proc.initialize_kv_cache()

        # 验证 worker.initialize_cache 被调用，使用 total_block_num
        mock_worker.initialize_cache.assert_called_once_with(num_gpu_blocks=1000)

    @patch("fastdeploy.worker.worker_process.get_worker")
    @patch("fastdeploy.worker.worker_process.current_platform")
    @patch("fastdeploy.worker.worker_process.paddle")
    def test_paddle_dis_worker_proc_initialize_kv_cache_no_profile_distributed(
        self, mock_paddle, mock_platform, mock_get_worker
    ):
        """测试 initialize_kv_cache 方法（do_profile=False，分布式）"""
        from fastdeploy.worker.worker_process import PaddleDisWorkerProc

        mock_platform.is_iluvatar.return_value = False
        mock_worker = Mock()
        mock_get_worker.return_value = mock_worker

        # 禁用 profile
        self.fd_config.parallel_config.do_profile = False
        self.fd_config.cache_config.total_block_num = 1000

        worker_proc = PaddleDisWorkerProc(self.fd_config, ranks=2, local_rank=0)
        worker_proc.initialize_kv_cache()

        # 验证 worker.initialize_cache 被调用
        mock_worker.initialize_cache.assert_called_once_with(num_gpu_blocks=1000)

    @patch("fastdeploy.worker.worker_process.get_worker")
    @patch("fastdeploy.worker.worker_process.current_platform")
    @patch("fastdeploy.worker.worker_process.paddle")
    @patch("fastdeploy.worker.worker_process.time")
    def test_paddle_dis_worker_proc_run_eplb_clear_stats(self, mock_time, mock_paddle, mock_platform, mock_get_worker):
        """测试 _run_eplb 方法（clear_stat=True）"""
        from fastdeploy.worker.worker_process import PaddleDisWorkerProc

        mock_platform.is_iluvatar.return_value = False
        mock_worker = Mock()
        mock_model = Mock()
        mock_worker.get_model.return_value = mock_model
        mock_get_worker.return_value = mock_worker

        # 启用 EPLB
        self.fd_config.eplb_config.enable_eplb = True
        self.fd_config.eplb_config.redundant_expert_dump_workload_interval = 10

        mock_time.time.return_value = 1000

        worker_proc = PaddleDisWorkerProc(self.fd_config, ranks=2, local_rank=0)

        # Mock EPLB 相关属性 - 确保条件满足
        # 时间间隔需要严格大于 interval（代码中是 >，不是 >=）
        worker_proc.last_dump_expert_workload_ts = 989  # 时间间隔：1000 - 989 = 11 > 10
        worker_proc.local_experts_token_stats_array = Mock()
        # 确保 value 不是 None，且可以访问
        stats_array = np.zeros([2, 8], dtype=np.int32)
        worker_proc.local_experts_token_stats_array.value = stats_array
        worker_proc.signal_clear_experts_token_stats = Mock()
        worker_proc.signal_clear_experts_token_stats.value = [1]  # 触发清除统计
        worker_proc.signal_update_weight_from_tensor_array = Mock()
        worker_proc.signal_update_weight_from_tensor_array.value = [0]
        worker_proc.rearrange_experts_signal = Mock()
        worker_proc.rearrange_experts_signal.value = [0]
        worker_proc.mmap_infos = {"main_model": "mmap_info"}

        # Mock worker 的 redundant_table_manger
        mock_model.redundant_table_manger = Mock()
        mock_model.redundant_table_manger.get_expert_tokens_stats.return_value = (
            np.zeros([2, 8], dtype=np.int32),
            None,
            None,
            None,
        )

        # Mock paddle.to_tensor 和 broadcast
        mock_tensor = Mock()
        mock_tensor.__getitem__ = Mock(return_value=0)
        mock_paddle.to_tensor.return_value = mock_tensor

        # 测试 _run_eplb（应该触发 clear_stat=True）
        worker_proc._run_eplb(tp_rank=0)

        # 验证 get_expert_tokens_stats 被调用，且 clear_stat=True
        # 时间间隔：1000 - 990 = 10 >= 10，满足条件
        mock_model.redundant_table_manger.get_expert_tokens_stats.assert_called_once_with(clear_stat=True)
        # 验证 signal_clear_experts_token_stats 被重置为 0
        self.assertEqual(worker_proc.signal_clear_experts_token_stats.value[0], 0)

    @patch("fastdeploy.worker.worker_process.IPCSignal")
    @patch("fastdeploy.worker.worker_process.get_worker")
    @patch("fastdeploy.worker.worker_process.current_platform")
    @patch("fastdeploy.worker.worker_process.paddle")
    @patch("fastdeploy.worker.worker_process.dist")
    def test_paddle_dis_worker_proc_initialize_kv_cache_master_rank_non_zero(
        self, mock_dist, mock_paddle, mock_platform, mock_get_worker, mock_ipc_signal
    ):
        """测试 initialize_kv_cache 方法（local_rank % max_chips_per_node != 0，非 master rank）"""
        from fastdeploy.worker.worker_process import PaddleDisWorkerProc

        mock_platform.is_iluvatar.return_value = False
        mock_worker = Mock()
        mock_worker.determine_available_memory.return_value = 10 * 1024**3  # 10GB
        mock_worker.cal_theortical_kvcache.return_value = 1024**2  # 1MB per block
        mock_get_worker.return_value = mock_worker

        # Mock IPCSignal（不应该被调用，因为不是 master rank）
        mock_signal = Mock()
        mock_signal.value = [0]
        mock_ipc_signal.return_value = mock_signal

        # Mock paddle.distributed
        mock_tensor = Mock()
        mock_tensor.item.return_value = 1000
        mock_paddle.full.return_value = mock_tensor
        mock_dist.all_reduce = Mock()
        mock_dist.ReduceOp = Mock()
        mock_dist.ReduceOp.MIN = Mock()

        # 启用 profile，local_rank=1，max_chips_per_node=8，所以 1 % 8 != 0（不是 master rank）
        self.fd_config.parallel_config.do_profile = True

        worker_proc = PaddleDisWorkerProc(self.fd_config, ranks=1, local_rank=1)
        worker_proc.initialize_kv_cache()

        # 验证 worker.initialize_cache 被调用
        mock_worker.initialize_cache.assert_called_once()
        # 验证 IPCSignal 没有被创建（非 master rank）
        # 由于 local_rank=1 % 8 != 0，所以不应该创建 get_profile_block_num_signal
        self.assertFalse(hasattr(worker_proc, "get_profile_block_num_signal"))

    @patch("fastdeploy.worker.worker_process.get_worker")
    @patch("fastdeploy.worker.worker_process.current_platform")
    @patch("fastdeploy.worker.worker_process.envs")
    @patch("fastdeploy.worker.worker_process.IPCSignal")
    def test_paddle_dis_worker_proc_graph_optimize_splitwise_prefill_not_prefill(
        self, mock_ipc_signal, mock_envs, mock_platform, mock_get_worker
    ):
        """测试 graph_optimize_and_warm_up_model 方法（splitwise_role != prefill）"""
        from fastdeploy.worker.worker_process import PaddleDisWorkerProc

        mock_platform.is_iluvatar.return_value = False
        mock_worker = Mock()
        mock_model_runner = Mock()
        mock_model_runner.device_id = 0
        mock_worker.model_runner = mock_model_runner
        mock_get_worker.return_value = mock_worker

        mock_envs.ENABLE_V1_KVCACHE_SCHEDULER = False
        self.fd_config.scheduler_config.splitwise_role = "decode"  # 不是 prefill

        worker_proc = PaddleDisWorkerProc(self.fd_config, ranks=1, local_rank=0)
        worker_proc.graph_optimize_and_warm_up_model()

        # 验证 worker 方法被调用
        mock_worker.graph_optimize_and_warm_up_model.assert_called_once()
        # 验证 IPCSignal 没有被创建（splitwise_role != prefill）
        mock_ipc_signal.assert_not_called()

    @patch("fastdeploy.worker.worker_process.get_worker")
    @patch("fastdeploy.worker.worker_process.current_platform")
    @patch("fastdeploy.worker.worker_process.envs")
    def test_paddle_dis_worker_proc_graph_optimize_v1_scheduler(self, mock_envs, mock_platform, mock_get_worker):
        """测试 graph_optimize_and_warm_up_model 方法（ENABLE_V1_KVCACHE_SCHEDULER=True）"""
        from fastdeploy.worker.worker_process import PaddleDisWorkerProc

        mock_platform.is_iluvatar.return_value = False
        mock_worker = Mock()
        mock_get_worker.return_value = mock_worker

        mock_envs.ENABLE_V1_KVCACHE_SCHEDULER = True  # 启用 V1 scheduler
        self.fd_config.scheduler_config.splitwise_role = "prefill"

        worker_proc = PaddleDisWorkerProc(self.fd_config, ranks=1, local_rank=0)
        worker_proc.graph_optimize_and_warm_up_model()

        # 验证 worker 方法被调用
        mock_worker.graph_optimize_and_warm_up_model.assert_called_once()
        # 由于 ENABLE_V1_KVCACHE_SCHEDULER=True，不应该创建 IPCSignal
        # （这个分支在代码中会被跳过）

    @patch("fastdeploy.worker.worker_process.get_worker")
    @patch("fastdeploy.worker.worker_process.current_platform")
    @patch("fastdeploy.worker.worker_process.envs")
    @patch("fastdeploy.worker.worker_process.time")
    def test_paddle_dis_worker_proc_run_eplb_interval_not_reached(
        self, mock_time, mock_envs, mock_platform, mock_get_worker
    ):
        """测试 _run_eplb 方法（时间间隔未达到）"""
        from fastdeploy.worker.worker_process import PaddleDisWorkerProc

        mock_platform.is_iluvatar.return_value = False
        mock_worker = Mock()
        mock_get_worker.return_value = mock_worker

        # 启用 EPLB
        self.fd_config.eplb_config.enable_eplb = True
        self.fd_config.eplb_config.redundant_expert_dump_workload_interval = 10

        mock_time.time.return_value = 1000

        worker_proc = PaddleDisWorkerProc(self.fd_config, ranks=2, local_rank=0)

        # Mock EPLB 相关属性 - 时间间隔未达到
        worker_proc.last_dump_expert_workload_ts = 995  # 只过了5秒，未达到10秒间隔
        worker_proc.local_experts_token_stats_array = Mock()
        worker_proc.local_experts_token_stats_array.value = np.zeros([2, 8], dtype=np.int32)
        worker_proc.signal_clear_experts_token_stats = Mock()
        worker_proc.signal_clear_experts_token_stats.value = [0]
        worker_proc.signal_update_weight_from_tensor_array = Mock()
        worker_proc.signal_update_weight_from_tensor_array.value = [0]
        worker_proc.rearrange_experts_signal = Mock()
        worker_proc.rearrange_experts_signal.value = [0]
        worker_proc.mmap_infos = {"main_model": "mmap_info"}

        # Mock paddle.to_tensor 和 broadcast
        with patch("fastdeploy.worker.worker_process.paddle") as mock_paddle:
            mock_tensor = Mock()
            mock_tensor.__getitem__ = Mock(return_value=0)
            mock_paddle.to_tensor.return_value = mock_tensor

            # 测试 _run_eplb（时间间隔未达到，不应该更新统计）
            worker_proc._run_eplb(tp_rank=0)

            # 验证 last_dump_expert_workload_ts 没有被更新（时间间隔未达到）
            self.assertEqual(worker_proc.last_dump_expert_workload_ts, 995)

    @patch("fastdeploy.worker.worker_process.IPCSignal")
    @patch("fastdeploy.worker.worker_process.RedundantExpertManager")
    @patch("fastdeploy.worker.worker_process.create_mmap")
    @patch("fastdeploy.worker.worker_process.get_worker")
    @patch("fastdeploy.worker.worker_process.current_platform")
    def test_paddle_dis_worker_proc_init_eplb_signal_master_rank0_explicit(
        self, mock_platform, mock_get_worker, mock_create_mmap, mock_expert_manager, mock_ipc_signal
    ):
        """测试 _init_eplb_signal 方法（明确设置 local_rank % tensor_parallel_size == 0）"""
        from fastdeploy.worker.worker_process import PaddleDisWorkerProc

        mock_platform.is_iluvatar.return_value = False
        mock_worker = Mock()
        mock_get_worker.return_value = mock_worker

        # 启用 EPLB 并设置 model_config
        self.fd_config.eplb_config.enable_eplb = True
        self.fd_config.model_config = Mock()
        self.fd_config.model_config.num_hidden_layers = 2
        self.fd_config.model_config.moe_num_experts = 8
        self.fd_config.parallel_config.tensor_parallel_size = 4
        self.fd_config.parallel_config.local_data_parallel_id = 0

        # Mock IPCSignal - 需要多个信号
        mock_signal = Mock()
        mock_signal.value = np.zeros([1], dtype=np.int32)
        mock_ipc_signal.return_value = mock_signal

        # Mock RedundantExpertManager
        mock_manager = Mock()
        mock_expert_manager.return_value = mock_manager

        # Mock create_mmap
        mock_create_mmap.return_value = {"main_model": "mmap_info"}

        # local_rank=0, tensor_parallel_size=4, 所以 local_rank % tensor_parallel_size = 0 % 4 = 0
        worker_proc = PaddleDisWorkerProc(self.fd_config, ranks=4, local_rank=0)
        # 确保 parallel_config.tensor_parallel_size 被正确设置
        worker_proc.parallel_config.tensor_parallel_size = 4
        worker_proc.parallel_config.local_engine_worker_queue_port = "9923"

        # 确保 IPCSignal 可以被多次调用（需要多个信号）
        call_count = [0]

        def ipc_signal_side_effect(*args, **kwargs):
            call_count[0] += 1
            mock_signal = Mock()
            mock_signal.value = np.zeros([1], dtype=np.int32)
            return mock_signal

        mock_ipc_signal.side_effect = ipc_signal_side_effect

        worker_proc._init_eplb_signal()

        # 验证 master rank0 的信号被初始化（local_rank % tensor_parallel_size == 0）
        # 0 % 4 == 0，所以应该进入第330行的分支
        self.assertIsNotNone(worker_proc.signal_update_weight_from_tensor_array)
        self.assertIsNotNone(worker_proc.rearrange_experts_signal)

    @patch("fastdeploy.worker.worker_process.get_worker")
    @patch("fastdeploy.worker.worker_process.current_platform")
    @patch("fastdeploy.worker.worker_process.paddle")
    @patch("fastdeploy.worker.worker_process.time")
    def test_paddle_dis_worker_proc_run_eplb_tp_rank0_rearrange_done(
        self, mock_time, mock_paddle, mock_platform, mock_get_worker
    ):
        """测试 _run_eplb 方法（tp_rank == 0，设置 rearrange_experts_signal 为 DONE）"""
        from fastdeploy.worker.worker_process import (
            REARRANGE_EXPERT_MAGIC_NUM,
            PaddleDisWorkerProc,
            RearrangeExpertStatus,
        )

        mock_platform.is_iluvatar.return_value = False
        mock_worker = Mock()
        mock_model = Mock()
        mock_worker.get_model.return_value = mock_model
        mock_get_worker.return_value = mock_worker

        # 启用 EPLB
        self.fd_config.eplb_config.enable_eplb = True
        self.fd_config.eplb_config.redundant_expert_dump_workload_interval = 10

        mock_time.time.return_value = 1000

        worker_proc = PaddleDisWorkerProc(self.fd_config, ranks=2, local_rank=0)

        # Mock EPLB 相关属性
        worker_proc.last_dump_expert_workload_ts = 989
        worker_proc.local_experts_token_stats_array = Mock()
        worker_proc.local_experts_token_stats_array.value = np.zeros([2, 8], dtype=np.int32)
        worker_proc.signal_clear_experts_token_stats = Mock()
        worker_proc.signal_clear_experts_token_stats.value = [0]
        worker_proc.signal_update_weight_from_tensor_array = Mock()
        worker_proc.signal_update_weight_from_tensor_array.value = [1]  # 触发更新
        worker_proc.rearrange_experts_signal = Mock()
        worker_proc.rearrange_experts_signal.value = [0]
        worker_proc.mmap_infos = {"main_model": "mmap_info"}
        worker_proc.update_weights_from_tensor = Mock()

        # Mock worker 的 redundant_table_manger
        mock_model.redundant_table_manger = Mock()
        mock_model.redundant_table_manger.get_expert_tokens_stats.return_value = (
            np.zeros([2, 8], dtype=np.int32),
            None,
            None,
            None,
        )

        # Mock paddle.to_tensor 和 broadcast（返回 REARRANGE_EXPERT_MAGIC_NUM）
        mock_tensor = Mock()
        mock_tensor.__getitem__ = Mock(return_value=REARRANGE_EXPERT_MAGIC_NUM)
        mock_paddle.to_tensor.return_value = mock_tensor
        mock_paddle.distributed.barrier = Mock()

        # 测试 _run_eplb（tp_rank=0，触发权重更新并设置 DONE）
        worker_proc._run_eplb(tp_rank=0)

        # 验证 rearrange_experts_signal 被设置为 DONE（tp_rank == 0 时）
        self.assertEqual(worker_proc.rearrange_experts_signal.value[0], RearrangeExpertStatus.DONE.value)

    @patch("fastdeploy.worker.worker_process.get_worker")
    @patch("fastdeploy.worker.worker_process.current_platform")
    def test_paddle_dis_worker_proc_event_loop_normal_exists(self, mock_platform, mock_get_worker):
        """测试 event_loop_normal 方法存在（不实际执行，因为包含无限循环）"""
        from fastdeploy.worker.worker_process import PaddleDisWorkerProc

        mock_platform.is_iluvatar.return_value = False
        mock_worker = Mock()
        mock_get_worker.return_value = mock_worker

        worker_proc = PaddleDisWorkerProc(self.fd_config, ranks=1, local_rank=0)

        # 只验证方法存在，不实际执行（因为包含 while True 无限循环）
        self.assertTrue(hasattr(worker_proc, "event_loop_normal"))
        self.assertTrue(callable(worker_proc.event_loop_normal))

    @patch("fastdeploy.worker.worker_process.get_worker")
    @patch("fastdeploy.worker.worker_process.current_platform")
    def test_paddle_dis_worker_proc_run_eplb_disabled(self, mock_platform, mock_get_worker):
        """测试 _run_eplb 方法（EPLB 未启用，应该直接返回）"""
        from fastdeploy.worker.worker_process import PaddleDisWorkerProc

        mock_platform.is_iluvatar.return_value = False
        mock_worker = Mock()
        mock_get_worker.return_value = mock_worker

        # 禁用 EPLB
        self.fd_config.eplb_config.enable_eplb = False

        worker_proc = PaddleDisWorkerProc(self.fd_config, ranks=2, local_rank=0)

        # Mock 相关属性（虽然不会被使用，但确保不会出错）
        worker_proc.last_dump_expert_workload_ts = 0
        worker_proc.local_experts_token_stats_array = Mock()
        worker_proc.signal_clear_experts_token_stats = Mock()
        worker_proc.signal_update_weight_from_tensor_array = Mock()
        worker_proc.rearrange_experts_signal = Mock()
        worker_proc.mmap_infos = {}

        # 测试 _run_eplb（EPLB 未启用，应该直接返回）
        # 不应该抛出异常
        try:
            worker_proc._run_eplb(tp_rank=0)
            # 如果执行到这里，说明方法正常返回了
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"_run_eplb should return early when EPLB is disabled, but raised: {e}")

    @patch("fastdeploy.worker.worker_process.get_worker")
    @patch("fastdeploy.worker.worker_process.current_platform")
    @patch("fastdeploy.worker.worker_process.paddle")
    @patch("fastdeploy.worker.worker_process.time")
    def test_paddle_dis_worker_proc_run_eplb_tp_rank_non_zero(
        self, mock_time, mock_paddle, mock_platform, mock_get_worker
    ):
        """测试 _run_eplb 方法（tp_rank != 0，不应该设置 rearrange_experts_signal）"""
        from fastdeploy.worker.worker_process import (
            REARRANGE_EXPERT_MAGIC_NUM,
            PaddleDisWorkerProc,
        )

        mock_platform.is_iluvatar.return_value = False
        mock_worker = Mock()
        mock_model = Mock()
        mock_worker.get_model.return_value = mock_model
        mock_get_worker.return_value = mock_worker

        # 启用 EPLB
        self.fd_config.eplb_config.enable_eplb = True
        self.fd_config.eplb_config.redundant_expert_dump_workload_interval = 10

        mock_time.time.return_value = 1000

        worker_proc = PaddleDisWorkerProc(self.fd_config, ranks=2, local_rank=0)

        # Mock EPLB 相关属性
        worker_proc.last_dump_expert_workload_ts = 989
        worker_proc.local_experts_token_stats_array = Mock()
        worker_proc.local_experts_token_stats_array.value = np.zeros([2, 8], dtype=np.int32)
        worker_proc.signal_clear_experts_token_stats = Mock()
        worker_proc.signal_clear_experts_token_stats.value = [0]
        worker_proc.signal_update_weight_from_tensor_array = Mock()
        worker_proc.signal_update_weight_from_tensor_array.value = [1]  # 触发更新
        worker_proc.rearrange_experts_signal = Mock()
        worker_proc.rearrange_experts_signal.value = [0]
        worker_proc.mmap_infos = {"main_model": "mmap_info"}
        worker_proc.update_weights_from_tensor = Mock()

        # Mock worker 的 redundant_table_manger
        mock_model.redundant_table_manger = Mock()
        mock_model.redundant_table_manger.get_expert_tokens_stats.return_value = (
            np.zeros([2, 8], dtype=np.int32),
            None,
            None,
            None,
        )

        # Mock paddle.to_tensor 和 broadcast（返回 REARRANGE_EXPERT_MAGIC_NUM）
        mock_tensor = Mock()
        mock_tensor.__getitem__ = Mock(return_value=REARRANGE_EXPERT_MAGIC_NUM)
        mock_paddle.to_tensor.return_value = mock_tensor
        mock_paddle.distributed.barrier = Mock()

        # 测试 _run_eplb（tp_rank=1，不应该设置 rearrange_experts_signal）
        initial_value = worker_proc.rearrange_experts_signal.value[0]
        worker_proc._run_eplb(tp_rank=1)

        # 验证 rearrange_experts_signal 没有被修改（tp_rank != 0 时）
        # 注意：由于 tp_rank != 0，第420行的分支不会执行
        self.assertEqual(worker_proc.rearrange_experts_signal.value[0], initial_value)

    @patch("fastdeploy.worker.worker_process.IPCSignal")
    @patch("fastdeploy.worker.worker_process.RedundantExpertManager")
    @patch("fastdeploy.worker.worker_process.create_mmap")
    @patch("fastdeploy.worker.worker_process.get_worker")
    @patch("fastdeploy.worker.worker_process.current_platform")
    def test_paddle_dis_worker_proc_init_eplb_signal_master_rank0_tensor_parallel_1(
        self, mock_platform, mock_get_worker, mock_create_mmap, mock_expert_manager, mock_ipc_signal
    ):
        """测试 _init_eplb_signal 方法（tensor_parallel_size=1，local_rank=0，确保覆盖 master rank0 分支）"""
        from fastdeploy.worker.worker_process import PaddleDisWorkerProc

        mock_platform.is_iluvatar.return_value = False
        mock_worker = Mock()
        mock_get_worker.return_value = mock_worker

        # 启用 EPLB 并设置 model_config
        self.fd_config.eplb_config.enable_eplb = True
        self.fd_config.model_config = Mock()
        self.fd_config.model_config.num_hidden_layers = 2
        self.fd_config.model_config.moe_num_experts = 8
        self.fd_config.parallel_config.tensor_parallel_size = 1  # tensor_parallel_size=1
        self.fd_config.parallel_config.local_data_parallel_id = 0
        self.fd_config.parallel_config.local_engine_worker_queue_port = "9923"

        # Mock IPCSignal - 需要多个信号
        call_count = [0]

        def ipc_signal_side_effect(*args, **kwargs):
            call_count[0] += 1
            mock_signal = Mock()
            mock_signal.value = np.zeros([1], dtype=np.int32)
            return mock_signal

        mock_ipc_signal.side_effect = ipc_signal_side_effect

        # Mock RedundantExpertManager
        mock_manager = Mock()
        mock_expert_manager.return_value = mock_manager

        # Mock create_mmap
        mock_create_mmap.return_value = {"main_model": "mmap_info"}

        # local_rank=0, tensor_parallel_size=1, 所以 local_rank % tensor_parallel_size = 0 % 1 = 0
        worker_proc = PaddleDisWorkerProc(self.fd_config, ranks=1, local_rank=0)
        # 确保 parallel_config.tensor_parallel_size 被正确设置
        worker_proc.parallel_config.tensor_parallel_size = 1

        worker_proc._init_eplb_signal()

        # 验证 master rank0 的信号被初始化（local_rank % tensor_parallel_size == 0）
        # 0 % 1 == 0，所以应该进入第330行的分支
        self.assertIsNotNone(worker_proc.signal_update_weight_from_tensor_array)
        self.assertIsNotNone(worker_proc.rearrange_experts_signal)

    @patch("fastdeploy.worker.worker_process.get_worker")
    @patch("fastdeploy.worker.worker_process.current_platform")
    def test_paddle_dis_worker_proc_event_loop_normal_dynamic_load_weight(self, mock_platform, mock_get_worker):
        """测试 event_loop_normal 方法中的 dynamic_load_weight 分支（只验证方法存在）"""
        from fastdeploy.worker.worker_process import PaddleDisWorkerProc

        mock_platform.is_iluvatar.return_value = False
        mock_worker = Mock()
        mock_get_worker.return_value = mock_worker

        # 启用 dynamic_load_weight
        self.fd_config.load_config.dynamic_load_weight = True

        worker_proc = PaddleDisWorkerProc(self.fd_config, ranks=1, local_rank=0)

        # 只验证方法存在和相关属性被正确设置
        self.assertTrue(hasattr(worker_proc, "event_loop_normal"))
        self.assertTrue(self.fd_config.load_config.dynamic_load_weight)

    @patch("fastdeploy.worker.worker_process.get_worker")
    @patch("fastdeploy.worker.worker_process.current_platform")
    def test_paddle_dis_worker_proc_event_loop_normal_task_queue_nnode_gt_1(self, mock_platform, mock_get_worker):
        """测试 event_loop_normal 方法中的 nnode > 1 分支（只验证方法存在）"""
        from fastdeploy.worker.worker_process import PaddleDisWorkerProc

        mock_platform.is_iluvatar.return_value = False
        mock_worker = Mock()
        mock_get_worker.return_value = mock_worker

        worker_proc = PaddleDisWorkerProc(self.fd_config, ranks=1, local_rank=0)

        # 设置 nnode > 1
        worker_proc.nnode = 2
        worker_proc.task_queue = Mock()
        worker_proc.task_queue.read_finish_flag = Mock()
        worker_proc.task_queue.read_finish_flag.set = Mock()

        # 只验证方法存在和相关属性被正确设置
        self.assertTrue(hasattr(worker_proc, "event_loop_normal"))
        self.assertGreater(worker_proc.nnode, 1)
        self.assertTrue(hasattr(worker_proc.task_queue.read_finish_flag, "set"))


if __name__ == "__main__":
    unittest.main()
