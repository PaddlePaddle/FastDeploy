"""Conftest for diffusion model tests — patches paddle.compat for AI Studio."""

import types

import paddle

if not hasattr(paddle, "compat"):
    paddle.compat = types.ModuleType("paddle.compat")
if not hasattr(paddle.compat, "enable_torch_proxy"):
    paddle.compat.enable_torch_proxy = lambda *a, **kw: None
