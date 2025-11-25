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

"""Configuration classes for XPU CI tests."""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ServerConfig:
    """Configuration for FastDeploy API server."""

    model_path: str
    port: int = 8188
    tensor_parallel_size: int = 4
    max_model_len: int = 32768
    max_num_seqs: int = 128
    quantization: str = "wint4"

    # Optional parameters
    enable_prefix_caching: bool = False
    enable_chunked_prefill: bool = False
    enable_expert_parallel: bool = False
    data_parallel_size: int = 1
    enable_mm: bool = False
    mm_processor_kwargs: Optional[str] = None
    limit_mm_per_prompt: Optional[str] = None
    reasoning_parser: Optional[str] = None
    num_gpu_blocks_override: Optional[int] = None
    gpu_memory_utilization: float = 0.9
    load_choices: str = "default"
    disable_sequence_parallel_moe: bool = False

    # Flags to control whether to explicitly pass certain parameters
    # Only EP tests need these parameters in original script
    use_gpu_memory_utilization: bool = False
    use_load_choices: bool = False

    # Port offsets
    engine_worker_queue_port_offset: int = 1
    metrics_port_offset: int = 2
    cache_queue_port_offset: int = 47873

    # For data_parallel_size > 1, each worker needs its own queue port
    # This controls the interval between ports (e.g., +10, +20, +30, +40)
    engine_worker_queue_port_interval: int = 10

    @property
    def engine_worker_queue_port(self) -> str:
        """Get engine worker queue port(s).

        For data_parallel_size > 1, returns comma-separated ports.
        E.g., with port=8188, offset=10, interval=10, dp_size=4:
        Returns "8198,8208,8218,8228"
        """
        if self.data_parallel_size > 1:
            base = self.port + self.engine_worker_queue_port_offset
            ports = [
                str(base + i * self.engine_worker_queue_port_interval)
                for i in range(self.data_parallel_size)
            ]
            return ",".join(ports)
        return str(self.port + self.engine_worker_queue_port_offset)

    @property
    def metrics_port(self) -> int:
        return self.port + self.metrics_port_offset

    @property
    def cache_queue_port(self) -> int:
        return self.port + self.cache_queue_port_offset

    def to_args(self) -> list:
        """Convert config to command line arguments."""
        args = [
            "--model", self.model_path,
            "--port", str(self.port),
            "--tensor-parallel-size", str(self.tensor_parallel_size),
            "--max-model-len", str(self.max_model_len),
            "--max-num-seqs", str(self.max_num_seqs),
            "--quantization", self.quantization,
            "--engine-worker-queue-port", self.engine_worker_queue_port,
            "--metrics-port", str(self.metrics_port),
            "--cache-queue-port", str(self.cache_queue_port),
        ]

        if self.num_gpu_blocks_override:
            args.extend(["--num-gpu-blocks-override", str(self.num_gpu_blocks_override)])

        if self.enable_prefix_caching:
            args.append("--enable-prefix-caching")

        if self.enable_chunked_prefill:
            args.append("--enable-chunked-prefill")

        if self.enable_expert_parallel:
            args.append("--enable-expert-parallel")
            args.extend(["--data-parallel-size", str(self.data_parallel_size)])

        if self.enable_mm:
            args.append("--enable-mm")

        if self.mm_processor_kwargs:
            args.extend(["--mm-processor-kwargs", self.mm_processor_kwargs])

        if self.limit_mm_per_prompt:
            args.extend(["--limit-mm-per-prompt", self.limit_mm_per_prompt])

        if self.reasoning_parser:
            args.extend(["--reasoning-parser", self.reasoning_parser])

        # Only pass these parameters if explicitly requested (EP tests only)
        if self.use_gpu_memory_utilization:
            args.extend(["--gpu-memory-utilization", str(self.gpu_memory_utilization)])

        if self.use_load_choices:
            args.extend(["--load-choices", self.load_choices])

        if self.disable_sequence_parallel_moe:
            args.append("--disable-sequence-parallel-moe")

        return args


@dataclass
class EnvironmentConfig:
    """Environment configuration for XPU tests."""

    xpu_id: int = 0
    xpu_visible_devices: str = "0,1,2,3"

    # BKCL environment variables for Expert Parallel
    bkcl_enable_xdr: bool = False
    bkcl_rdma_nics: str = "xgbe1,xgbe2,xgbe3,xgbe4"
    bkcl_trace_topo: bool = False
    bkcl_pcie_ring: bool = False
    xshmem_mode: bool = False
    xshmem_qp_num_per_rank: int = 32
    bkcl_rdma_verbs: bool = False

    def apply(self):
        """Apply environment variables."""
        os.environ["XPU_VISIBLE_DEVICES"] = self.xpu_visible_devices

        if self.bkcl_enable_xdr:
            os.environ["BKCL_ENABLE_XDR"] = "1"
            os.environ["BKCL_RDMA_NICS"] = self.bkcl_rdma_nics
            os.environ["BKCL_TRACE_TOPO"] = "1" if self.bkcl_trace_topo else "0"
            os.environ["BKCL_PCIE_RING"] = "1" if self.bkcl_pcie_ring else "0"
            os.environ["XSHMEM_MODE"] = "1" if self.xshmem_mode else "0"
            os.environ["XSHMEM_QP_NUM_PER_RANK"] = str(self.xshmem_qp_num_per_rank)
            os.environ["BKCL_RDMA_VERBS"] = "1" if self.bkcl_rdma_verbs else "0"

    def cleanup(self):
        """Remove environment variables."""
        env_vars = [
            "BKCL_ENABLE_XDR", "BKCL_RDMA_NICS", "BKCL_TRACE_TOPO",
            "BKCL_PCIE_RING", "XSHMEM_MODE", "XSHMEM_QP_NUM_PER_RANK",
            "BKCL_RDMA_VERBS"
        ]
        for var in env_vars:
            os.environ.pop(var, None)


@dataclass
class TestConfig:
    """Configuration for a test case."""

    name: str
    description: str
    server_config: ServerConfig
    env_config: EnvironmentConfig = field(default_factory=EnvironmentConfig)

    # Test parameters
    health_check_timeout: int = 900  # 15 minutes
    health_check_interval: int = 10
    startup_wait: int = 60

    # Expected keywords in response for validation
    expected_keywords: list = field(default_factory=lambda: ["人工智能", "文心一言"])

    @classmethod
    def create_v1_test(cls, model_path: str, xpu_id: int = 0) -> "TestConfig":
        """Create V1 mode test configuration."""
        port = 8188 + xpu_id * 100
        xpu_devices = "0,1,2,3" if xpu_id == 0 else "4,5,6,7"

        return cls(
            name="v1_mode",
            description="V1模式测试 (ERNIE-4.5-300B with wint4)",
            server_config=ServerConfig(
                model_path=os.path.join(model_path, "ERNIE-4.5-300B-A47B-Paddle"),
                port=port,
                tensor_parallel_size=4,
                max_num_seqs=128,
                quantization="wint4",
                num_gpu_blocks_override=16384,
                enable_prefix_caching=True,
                enable_chunked_prefill=True,
            ),
            env_config=EnvironmentConfig(
                xpu_id=xpu_id,
                xpu_visible_devices=xpu_devices,
            ),
        )

    @classmethod
    def create_w4a8_test(cls, model_path: str, xpu_id: int = 0) -> "TestConfig":
        """Create W4A8 quantization test configuration."""
        port = 8188 + xpu_id * 100
        xpu_devices = "0,1,2,3" if xpu_id == 0 else "4,5,6,7"

        return cls(
            name="w4a8",
            description="W4A8量化测试",
            server_config=ServerConfig(
                model_path=os.path.join(model_path, "ERNIE-4.5-300B-A47B-W4A8C8-TP4-Paddle"),
                port=port,
                tensor_parallel_size=4,
                max_num_seqs=64,
                quantization="W4A8",
                num_gpu_blocks_override=16384,
            ),
            env_config=EnvironmentConfig(
                xpu_id=xpu_id,
                xpu_visible_devices=xpu_devices,
            ),
        )

    @classmethod
    def create_vl_test(cls, model_path: str, xpu_id: int = 0) -> "TestConfig":
        """Create VL model test configuration."""
        port = 8188 + xpu_id * 100
        xpu_devices = "0,1,2,3" if xpu_id == 0 else "4,5,6,7"

        return cls(
            name="vl_model",
            description="VL多模态模型测试",
            server_config=ServerConfig(
                model_path=os.path.join(model_path, "ERNIE-4.5-VL-28B-A3B-Paddle"),
                port=port,
                tensor_parallel_size=4,
                max_num_seqs=10,
                quantization="wint8",
                enable_mm=True,
                mm_processor_kwargs='{"video_max_frames": 30}',
                limit_mm_per_prompt='{"image": 10, "video": 3}',
                reasoning_parser="ernie-45-vl",
                enable_chunked_prefill=True,
            ),
            env_config=EnvironmentConfig(
                xpu_id=xpu_id,
                xpu_visible_devices=xpu_devices,
            ),
            expected_keywords=["北魏", "北齐", "释迦牟尼"],
        )

    @classmethod
    def create_ep4tp4_test(cls, model_path: str, xpu_id: int = 0) -> "TestConfig":
        """Create EP4TP4 Expert Parallel test configuration."""
        port = 8188 + xpu_id * 100
        xpu_devices = "0,1,2,3" if xpu_id == 0 else "4,5,6,7"

        return cls(
            name="ep4tp4",
            description="EP4TP4在线服务测试",
            server_config=ServerConfig(
                model_path=os.path.join(model_path, "ERNIE-4.5-300B-A47B-Paddle"),
                port=port,
                tensor_parallel_size=4,
                max_num_seqs=64,
                quantization="wint4",
                enable_expert_parallel=True,
                data_parallel_size=1,
                gpu_memory_utilization=0.9,
                load_choices="default",
                use_gpu_memory_utilization=True,
                use_load_choices=True,
                disable_sequence_parallel_moe=True,
                engine_worker_queue_port_offset=10,
            ),
            env_config=EnvironmentConfig(
                xpu_id=xpu_id,
                xpu_visible_devices=xpu_devices,
                bkcl_enable_xdr=True,
                bkcl_trace_topo=True,
                bkcl_pcie_ring=True,
                xshmem_mode=True,
                bkcl_rdma_verbs=True,
            ),
        )

    @classmethod
    def create_ep4tp1_test(cls, model_path: str, xpu_id: int = 0) -> "TestConfig":
        """Create EP4TP1 Expert Parallel test configuration."""
        port = 8188 + xpu_id * 100
        xpu_devices = "0,1,2,3" if xpu_id == 0 else "4,5,6,7"

        return cls(
            name="ep4tp1",
            description="EP4TP1在线服务测试",
            server_config=ServerConfig(
                model_path=os.path.join(model_path, "ERNIE-4.5-300B-A47B-Paddle"),
                port=port,
                tensor_parallel_size=1,
                max_num_seqs=64,
                quantization="wint4",
                enable_expert_parallel=True,
                data_parallel_size=4,
                gpu_memory_utilization=0.9,
                load_choices="default",
                use_gpu_memory_utilization=True,
                use_load_choices=True,
                engine_worker_queue_port_offset=10,
            ),
            env_config=EnvironmentConfig(
                xpu_id=xpu_id,
                xpu_visible_devices=xpu_devices,
                bkcl_enable_xdr=True,
                bkcl_trace_topo=True,
                bkcl_pcie_ring=True,
                xshmem_mode=True,
                bkcl_rdma_verbs=True,
            ),
        )

    @classmethod
    def create_ep4tp4_all2all_test(cls, model_path: str, xpu_id: int = 0) -> "TestConfig":
        """Create EP4TP4 all2all test configuration."""
        port = 8188 + xpu_id * 100
        xpu_devices = "0,1,2,3" if xpu_id == 0 else "4,5,6,7"

        return cls(
            name="ep4tp4_all2all",
            description="EP4TP4 all2all测试",
            server_config=ServerConfig(
                model_path=os.path.join(model_path, "ERNIE-4.5-300B-A47B-Paddle"),
                port=port,
                tensor_parallel_size=4,
                max_num_seqs=64,
                quantization="wint4",
                enable_expert_parallel=True,
                data_parallel_size=1,
                gpu_memory_utilization=0.9,
                load_choices="default",
                use_gpu_memory_utilization=True,
                use_load_choices=True,
                engine_worker_queue_port_offset=10,
            ),
            env_config=EnvironmentConfig(
                xpu_id=xpu_id,
                xpu_visible_devices=xpu_devices,
                bkcl_enable_xdr=True,
                bkcl_trace_topo=True,
                bkcl_pcie_ring=True,
                xshmem_mode=True,
                bkcl_rdma_verbs=True,
            ),
        )
