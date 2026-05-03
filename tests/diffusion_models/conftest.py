"""Unit tests for Hackathon 10th Spring No.48."""

import types

import paddle

if not hasattr(paddle, "compat"):
    paddle.compat = types.ModuleType("paddle.compat")
if not hasattr(paddle.compat, "enable_torch_proxy"):
    paddle.compat.enable_torch_proxy = lambda *a, **kw: None
