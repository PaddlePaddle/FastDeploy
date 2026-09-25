"""Elastic-Attention kernel wrappers (paddle)."""

from .block_sparse_attn import block_sparse_attn_paddle  # noqa: F401
from .find_blocks import find_blocks_chunked  # noqa: F401
from .xattention import Xattention_prefill_dim4, xattn_estimate  # noqa: F401
