"""Bridge ckpt config.json fields into FastDeploy ``model_config``.

ckpt fields (PawQwen3) that runtime needs:

- toggle_type / retrieval_mode / enable_ada_sparsity
- pooling_mode / use_softmax
- sink_size / local_window_size
- xattn_stride / xattn_threshold / xattn_norm
- block_size

Standard Qwen3 fields (hidden_size, head_dim, num_kv_heads, RoPE/YaRN, ...)
are handled by paddleformers' ``Qwen3Config`` already; this module only adds
the ELASTIC fields and provides default values for missing entries.
"""

from __future__ import annotations

# Mapping: model_config attribute name -> (ckpt field name, default value)
ELASTIC_CONFIG_FIELDS = {
    "local_window_size":   ("local_window_size",   2048),
    "sink_size":           ("sink_size",           128),
    "toggle_type":         ("toggle_type",         "xattn"),
    "retrieval_mode":      ("retrieval_mode",      "full"),
    "enable_ada_sparsity": ("enable_ada_sparsity", True),
    "pooling_mode":        ("pooling_mode",        "ctx_q"),
    "use_softmax":         ("use_softmax",         True),
    "xattn_stride":        ("xattn_stride",        16),
    "xattn_threshold":     ("xattn_threshold",     0.9),
    "xattn_norm":          ("xattn_norm",          1),
    "block_size":          ("block_size",          128),
}

# Training-only fields that runtime must silently ignore:
ELASTIC_TRAIN_ONLY = {
    "enable_lambda_task",
    "enable_layerwise_sparsity",
    "disable_linear_regularization_term",
    "layerwise_sparsity_first",
    "layerwise_sparsity_last",
    "layerwise_sparsity_pattern",
    "erank_analysis_path",
    "suggested_sparsity",
    "suggested_threshold",
    "topk_k",
    "triangle_n_last",
    "use_task_emb_for_mask",
    "pooling_seq",
    "max_window_layers",
}


def populate_elastic_fields(model_config) -> None:
    """Read ELASTIC_CONFIG_FIELDS off ``pretrained_config`` and lift them to
    ``model_config`` attributes, falling back to defaults.  Idempotent."""
    raw = getattr(model_config, "pretrained_config", None) or model_config
    for attr, (ckpt_key, default) in ELASTIC_CONFIG_FIELDS.items():
        if hasattr(model_config, attr):
            continue
        val = getattr(raw, ckpt_key, default)
        setattr(model_config, attr, val)
