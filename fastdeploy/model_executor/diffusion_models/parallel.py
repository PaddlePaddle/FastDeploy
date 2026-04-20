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
Tensor-parallel and quantization adaptation for diffusion transformers.

Provides utilities to replace standard ``nn.Linear`` layers in Flux/SD3 DiT
blocks with FastDeploy's ``ColumnParallelLinear`` / ``RowParallelLinear``
when running under a multi-GPU ``ParallelConfig``.

Usage (single-GPU, default — no-op):
    engine = DiffusionEngine(config)
    engine.load()   # uses plain nn.Linear everywhere

Usage (tensor-parallel, future):
    from fastdeploy.model_executor.diffusion_models.parallel import (
        apply_tensor_parallel,
    )
    engine = DiffusionEngine(config)
    engine.load()
    apply_tensor_parallel(engine.transformer, fd_config)

Quantization hooks follow the same replacement pattern — see
``apply_weight_quantization`` below.

The separation into *stubs* (this file) keeps the core model code clean and
framework-agnostic, while allowing FD-native parallel/quant to be wired in
without modifying the DiT forward pass.
"""

from __future__ import annotations

import logging
from typing import Optional

from paddle import nn

logger = logging.getLogger(__name__)

# TP layer names that should be split along the output dimension (column-parallel).
# These are QKV projections and MLP gate/up projections in Flux/SD3 DiT blocks.
_COLUMN_PARALLEL_PATTERNS = (
    "attn_qkv",  # Flux/SD3: joint QKV projection
    "attn_qkv_context",  # Flux/SD3: context stream QKV
    "mlp.0",  # MLP gate (first linear in Sequential)
    "mlp_context.0",  # Context MLP gate
)

# TP layer names that should be split along the input dimension (row-parallel).
# These are attention output projections and MLP down projections.
_ROW_PARALLEL_PATTERNS = (
    "attn_out",  # Flux/SD3: attention output projection
    "attn_out_context",  # Flux/SD3: context attention output
    "mlp.2",  # MLP down projection (third in Sequential)
    "mlp_context.2",  # Context MLP down
    "proj_out",  # SD3: final projection
)


def apply_tensor_parallel(
    model: nn.Layer,
    fd_config: "FDConfig",  # noqa: F821 — lazy import avoids circular
    prefix: str = "",
) -> None:
    """Replace ``nn.Linear`` layers in *model* with TP-parallel equivalents.

    This is a **Phase 3 stub** — the replacement logic is wired up but
    activating it requires a live ``FDConfig`` with ``tensor_parallel_size > 1``
    and the ``paddle.distributed.fleet`` backend initialised.  On single-GPU
    (the hackathon default), this function is a no-op.

    Args:
        model: A ``FluxForImageGeneration`` or ``SD3Transformer2DModel``.
        fd_config: FastDeploy configuration (carries ``ParallelConfig``).
        prefix: Weight-name prefix for checkpoint loading.
    """
    tp_size = getattr(
        getattr(fd_config, "parallel_config", None),
        "tensor_parallel_size",
        1,
    )
    if tp_size <= 1:
        logger.debug("TP size=1 — skipping tensor-parallel conversion for DiT.")
        return

    # TODO(Phase 3): Walk model.named_modules(), match against _COLUMN/_ROW
    # patterns, replace nn.Linear → ColumnParallelLinear / RowParallelLinear
    # from fastdeploy.model_executor.layers.linear.  Requires fleet init +
    # FDConfig integration in diffusion engine.
    replaced = 0
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(pat in name for pat in _COLUMN_PARALLEL_PATTERNS):
            logger.info("TP column-parallel candidate: %s", name)
            replaced += 1
        elif any(pat in name for pat in _ROW_PARALLEL_PATTERNS):
            logger.info("TP row-parallel candidate: %s", name)
            replaced += 1

    logger.info(
        "Tensor-parallel scan: %d layers eligible for TP=%d sharding in %s",
        replaced,
        tp_size,
        model.__class__.__name__,
    )


def apply_weight_quantization(
    model: nn.Layer,
    quant_method: Optional[str] = None,
    quant_bits: int = 8,
) -> None:
    """Apply weight-only quantization to DiT linear layers.

    Integrates with FastDeploy's quantization infrastructure
    (``fastdeploy.model_executor.layers.quantization``).

    This is a **Phase 3 stub**.  The actual replacement requires:
    1. A ``QuantConfigBase`` instance (e.g., from ``parse_quant_config``).
    2. Calling ``QuantMethodBase.create_weights`` on each eligible layer.

    Args:
        model: DiT model (Flux or SD3).
        quant_method: Quantization algorithm name (e.g., ``"w8a8"``, ``"w4a16"``).
            ``None`` means no quantization (no-op).
        quant_bits: Weight bit-width for the quantization scheme.
    """
    if quant_method is None:
        logger.debug("No quantization requested — skipping.")
        return

    # TODO(Phase 3): Replace eligible nn.Linear with quantised equivalents
    # using fastdeploy.model_executor.layers.quantization infrastructure
    # (QuantConfigBase + QuantMethodBase.create_weights).
    eligible = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            in_f = module.weight.shape[0]
            out_f = module.weight.shape[1]
            # Skip small layers (embeddings, norms) — only quantise ≥256 columns
            if min(in_f, out_f) >= 256:
                eligible += 1

    logger.info(
        "Quantization scan: %d linear layers eligible for %s (bits=%d) in %s",
        eligible,
        quant_method,
        quant_bits,
        model.__class__.__name__,
    )
