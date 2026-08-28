"""
Elastic-Attention integration for Qwen3 on FastDeploy.

Importing this package registers the ``PawQwen3ForCausalLM`` architecture
in :class:`fastdeploy.model_executor.models.model_base.ModelRegistry`.

It also patches ``attention_selecter.get_attention_backend`` /
``_get_attn_backend`` so that :class:`Qwen3ElasticAttentionBackend` is
returned **only when the caller's fd_config corresponds to a PawQwen3
model**. For any other architecture the selector falls through to its
original behaviour, which means dense Qwen3 (or any other FD model) is
unaffected even though FastDeploy's ``auto_models_registry`` always
imports this package on startup.
"""

from .modeling_elastic_qwen3 import (  # noqa: F401
    AttentionRouter,
    PawQwen3ForCausalLM,
    Qwen3ElasticAttention,
    Qwen3ElasticDecoderLayer,
    Qwen3ElasticModel,
)

# ---- Architecture-aware elastic attention backend patch ----
# FastDeploy's default dispatch only consults
# ``current_platform.get_attention_backend_cls`` (which on CUDA returns
# ``AppendAttentionBackend``) and ignores the model class's
# ``_get_attn_backend_cls``. Without this patch the elastic backend is
# dead code and PawQwen3 silently runs as plain dense Qwen3.
#
# However, ``fastdeploy.model_executor.models.__init__.auto_models_registry``
# walks every model package and triggers this package's import for ANY
# launch (including dense Qwen3). A blind global patch would therefore
# return the elastic backend for dense models too -- whose Attention
# layers lack ``mask_allocator`` -- causing AttributeError.
#
# Solution: patch the selector but make it architecture-aware. We walk
# the call stack to find the caller's ``self.fd_config`` (every call site
# in FD lives on an object that owns ``fd_config``), and only return the
# elastic backend when ``architectures[0] == "PawQwen3ForCausalLM"``.
# Other models fall through to the original selector unchanged.
import sys as _sys  # noqa: E402

from fastdeploy.model_executor.layers.attention import (  # noqa: E402
    attention_selecter as _selecter,
)
from fastdeploy.model_executor.layers.attention.elastic_attn_backend import (  # noqa: E402
    Qwen3ElasticAttentionBackend as _ElasticBackend,
)

_orig_get_attn_backend = _selecter._get_attn_backend
_orig_get_attention_backend = _selecter.get_attention_backend


def _caller_arch():
    """Walk the call stack to find ``self.fd_config.model_config.architectures``."""
    frame = _sys._getframe(2)  # skip this fn + the patched selector fn
    while frame is not None:
        local_self = frame.f_locals.get("self")
        if local_self is not None:
            fd_config = getattr(local_self, "fd_config", None)
            if fd_config is not None:
                model_config = getattr(fd_config, "model_config", None)
                archs = getattr(model_config, "architectures", None) or []
                if archs:
                    return archs[0]
        frame = frame.f_back
    return None


def _patched_get_attn_backend(selected_backend=None):
    if _caller_arch() == "PawQwen3ForCausalLM":
        return _ElasticBackend
    return _orig_get_attn_backend(selected_backend)


def _patched_get_attention_backend():
    if _caller_arch() == "PawQwen3ForCausalLM":
        return _ElasticBackend
    return _orig_get_attention_backend()


try:
    _orig_get_attn_backend.cache_clear()
except AttributeError:
    pass
_selecter._get_attn_backend = _patched_get_attn_backend
_selecter.get_attention_backend = _patched_get_attention_backend


# ---- Force Qwen-style RoPE for PawQwen3 architecture ----
# ``InputBatch`` builds ``rope_emb`` during ``GpuModelRunner.__init__``,
# which runs **before** ``PawQwen3ForCausalLM.__init__`` gets a chance to
# rewrite ``architectures[0]``. At that point the architecture name is
# still ``"PawQwen3ForCausalLM"`` (does not start with "Qwen"), so
# ``get_rope_impl`` falls through to ``ErnieRotaryEmbedding`` which
# produces ``rope_emb`` with last-dim ``head_dim/2 = 64``. The neox-style
# ``gqa_rope_write_cache`` kernel then asserts
# ``rotary_embs.dims()[4] == head_dim`` (128) or ``head_dim/4`` (32) and
# crashes.
#
# Additionally, the PawQwen3 4B / 64K / 262K checkpoints ship with
# ``rope_scaling = {"type": "yarn", "factor": 8.0,
# "original_max_position_embeddings": 40960}`` and were TRAINED with YaRN.
# The plain ``QwenRotaryEmbedding`` ignores ``rope_scaling`` and produces
# vanilla RoPE with no inv-freq interpolation and no ``mscale`` magnitude
# correction. The mismatch between training-time YaRN cos/sin and
# inference-time vanilla cos/sin is enough to flip the per-layer K
# distribution to which the elastic router/attention is highly
# sensitive, producing pure-garbage outputs (``"The 』The 』..."``).
#
# Patch ``get_rope_impl`` so PawQwen3 + yarn rope_scaling routes through
# ``GptOssScalingRotaryEmbedding`` (``use_neox_rotary_style=True``), which
# implements the same YaRN math as DeepseekScalingRotaryEmbedding but
# emits the ``(2, 1, T, 1, head_dim)`` rope_emb layout that the neox
# ``gqa_rope_write_cache`` / ``append_attention`` kernels expect.
from fastdeploy.model_executor.layers import rotary_embedding as _rope_mod  # noqa: E402

_orig_get_rope_impl = _rope_mod.get_rope_impl


def _is_pawqwen3(model_config):
    archs = getattr(model_config, "architectures", None) or []
    return bool(archs) and "Qwen" in archs[0] and not archs[0].startswith("Qwen")


def _yarn_rope_scaling(model_config):
    rs = getattr(model_config, "rope_scaling", None)
    if not isinstance(rs, dict):
        return None
    rope_type = rs.get("rope_type") or rs.get("type")
    if rope_type != "yarn":
        return None
    return rs


def _patched_get_rope_impl(rotary_dim, base, position_ids, model_config=None, partial_rotary_factor=1):
    if _is_pawqwen3(model_config):
        rs = _yarn_rope_scaling(model_config)
        if rs is not None:
            # Build YaRN cos/sin cache on the fly. Matches training-time
            # transformers ``Qwen3RotaryEmbedding`` with rope_type=yarn.
            yarn_layer = _rope_mod.GptOssScalingRotaryEmbedding(
                rotary_dim=model_config.head_dim,
                base=model_config.rope_theta,
                original_max_position_embeddings=int(rs["original_max_position_embeddings"]),
                scale=float(rs["factor"]),
                mscale=float(rs.get("mscale", 1.0)),
                attn_factor=float(rs.get("attn_factor", 1.0)),
                beta_fast=int(rs.get("beta_fast", 32)),
                beta_slow=int(rs.get("beta_slow", 1)),
                extrapolation_factor=float(rs.get("extrapolation_factor", 1.0)),
                use_neox_rotary_style=True,
            )
            return yarn_layer(position_ids)
        # No yarn scaling -> fall through to plain Qwen RoPE by temporarily
        # prefixing the architecture name so the upstream impl picks
        # ``QwenRotaryEmbedding``.
        original = model_config.architectures[0]
        try:
            model_config.architectures[0] = "Qwen3" + original
            return _orig_get_rope_impl(rotary_dim, base, position_ids, model_config, partial_rotary_factor)
        finally:
            model_config.architectures[0] = original
    return _orig_get_rope_impl(rotary_dim, base, position_ids, model_config, partial_rotary_factor)


_rope_mod.get_rope_impl = _patched_get_rope_impl

