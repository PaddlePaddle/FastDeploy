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

from types import SimpleNamespace

import paddle

architectures = "Ernie4_5_MoeForCausalLM"
import sys
from pathlib import Path

from fastdeploy.config import ErnieArchitectures, ModelConfig
from fastdeploy.model_executor.layers.moe.fused_moe_cutlass_backend import (
    CutlassWeightOnlyMoEMethod,
)
from fastdeploy.model_executor.layers.quantization import parse_quant_config
from fastdeploy.model_executor.models.model_base import ModelRegistry
from fastdeploy.worker.worker_process import init_distributed_environment, parse_args

TEST_DIR = Path(__file__).resolve().parent
TEST_MODEL_DIR = TEST_DIR / "test_model"


def make_fd_config(
    *,
    model_format="paddle",
    tensor_parallel_size=1,
    tensor_parallel_rank=0,
    splitwise_role="prefill",
    use_sequence_parallel_moe=False,
    load_choices="default_v1",
    model=str(TEST_MODEL_DIR),
):
    argv_backup = sys.argv
    try:
        sys.argv = ["fastdeploy"]
        args = parse_args()
    finally:
        sys.argv = argv_backup
    args.model = model
    model_config = ModelConfig(vars(args))
    return SimpleNamespace(
        model_config=model_config,
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tensor_parallel_size,
            tensor_parallel_rank=tensor_parallel_rank,
            expert_parallel_size=1,
            expert_parallel_rank=0,
            tp_group=None,
            use_sequence_parallel_moe=use_sequence_parallel_moe,
        ),
        scheduler_config=SimpleNamespace(splitwise_role=splitwise_role, max_num_seqs=1),
        load_config=SimpleNamespace(dynamic_load_weight=False, load_choices=load_choices),
        quant_config=parse_quant_config(
            args,
            model_config,
            is_ernie=ErnieArchitectures.contains_ernie_arch(model_config.architectures),
            is_v1_loader=True,
        ),
        eplb_config=SimpleNamespace(enable_eplb=False),
        plas_attention_config=None,
        routing_replay_config=SimpleNamespace(enable_routing_replay=False),
        graph_opt_config=SimpleNamespace(graph_opt_level=0, use_cudagraph=False),
    )


baseline = {
    "ernie.layers.24.mlp.experts",
    "ernie.layers.23.mlp.experts",
    "ernie.layers.11.mlp.experts",
    "ernie.layers.5.mlp.experts",
    "ernie.layers.22.mlp.experts",
    "ernie.layers.15.mlp.experts",
    "ernie.layers.3.mlp.experts",
    "ernie.layers.21.mlp.experts",
    "ernie.layers.4.mlp.experts",
    "ernie.layers.13.mlp.experts",
    "ernie.layers.6.mlp.experts",
    "ernie.layers.7.mlp.experts",
    "ernie.layers.14.mlp.experts",
    "ernie.layers.12.mlp.experts",
    "ernie.layers.10.mlp.experts",
    "ernie.layers.20.mlp.experts",
}


def collect_cutlass_moe_layers(model) -> set[str]:
    matched_keys = set()

    for name, layer in model.named_sublayers():
        quant_method = getattr(layer, "quant_method", None)
        if isinstance(quant_method, CutlassWeightOnlyMoEMethod):
            matched_keys.add(name)

    return matched_keys


def test_skip_layer_mixed_quantization():
    ranks, local_rank = init_distributed_environment()
    context = paddle.LazyGuard()
    with context:
        model_cls = ModelRegistry.get_class(architectures)
        model = model_cls(make_fd_config())
    res = collect_cutlass_moe_layers(model)
    assert res == baseline
