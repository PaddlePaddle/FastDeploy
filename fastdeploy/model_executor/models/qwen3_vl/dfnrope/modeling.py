from __future__ import annotations

import paddle
from paddle import nn
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from paddleformers.transformers.model_utils import PretrainedModel
from paddleformers.utils.log import logger

from fastdeploy.model_executor.layers.utils import get_tensor
from fastdeploy.model_executor.models.qwen2_5_vl.dfnrope.modeling import (
    VisionFlashAttention2,
    VisionRotaryEmbedding,
)
from fastdeploy.model_executor.utils import set_weight_attrs

from .activation import get_activation_fn
from .configuration import Qwen3VisionTransformerConfig


class Qwen3VisionPatchEmbed(nn.Layer):
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        hidden_size: int = 1152,
        model_format: str = "",
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size

        kernel_size = (temporal_patch_size, patch_size, patch_size)
        self.proj = nn.Conv3D(
            in_channels,
            hidden_size,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias_attr=True,
        )

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        # L, C = hidden_states.shape
        # hidden_states = hidden_states.view(L, -1, self.temporal_patch_size, self.patch_size, self.patch_size)
        # hidden_states = self.proj(hidden_states).view(L, self.hidden_size)
        target_dtype = self.proj.weight.dtype
        sequence_length = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(
            [-1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size]
        )
        hidden_states = self.proj(paddle.cast(hidden_states, target_dtype)).reshape(
            [sequence_length, self.hidden_size]
        )
        return hidden_states


class Qwen3VisionMLP(nn.Layer):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        hidden_act: str = "gelu_tanh",
        tensor_model_parallel_size: int = 1,
        model_format: str = "",
    ) -> None:
        super().__init__()
        self.tensor_model_parallel_size = tensor_model_parallel_size

        if tensor_model_parallel_size > 1:
            self.linear_fc1 = ColumnParallelLinear(
                dim,
                hidden_dim,
                mp_group=fleet.get_hybrid_communicate_group().get_model_parallel_group(),
                gather_output=False,
                has_bias=True,
            )
            self.linear_fc2 = RowParallelLinear(
                hidden_dim,
                dim,
                mp_group=fleet.get_hybrid_communicate_group().get_model_parallel_group(),
                input_is_parallel=True,
                has_bias=True,
            )
            set_weight_attrs(self.linear_fc1.weight, {"output_dim": True})
            set_weight_attrs(self.linear_fc2.weight, {"output_dim": False})

            set_weight_attrs(self.linear_fc1.bias, {"output_dim": True})
        else:
            self.linear_fc1 = nn.Linear(dim, hidden_dim, bias_attr=True)
            self.linear_fc2 = nn.Linear(hidden_dim, dim, bias_attr=True)

        set_weight_attrs(self.linear_fc1.weight, {"weight_need_transpose": model_format == "torch"})
        set_weight_attrs(self.linear_fc2.weight, {"weight_need_transpose": model_format == "torch"})
        self.act = get_activation_fn(hidden_act)

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        hidden_states = self.linear_fc2(self.act(self.linear_fc1(hidden_states)))
        return hidden_states


class Qwen3VisionPatchMerger(nn.Layer):
    def __init__(
        self,
        d_model: int,
        context_dim: int,
        spatial_merge_size: int,
        tensor_model_parallel_size: int,
        use_postshuffle_norm: bool = False,
        norm_eps: float = 1e-6,
        model_format: str = "",
    ) -> None:
        super().__init__()
        self.tensor_model_parallel_size = tensor_model_parallel_size
        self.spatial_merge_size = spatial_merge_size
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm

        if self.use_postshuffle_norm:
            context_dim = self.hidden_size
        self.norm = nn.LayerNorm(context_dim, epsilon=norm_eps)

        if tensor_model_parallel_size > 1:
            self.linear_fc1 = ColumnParallelLinear(
                self.hidden_size,
                self.hidden_size,
                mp_group=fleet.get_hybrid_communicate_group().get_model_parallel_group(),
                gather_output=False,
                has_bias=True,
            )
            self.linear_fc2 = RowParallelLinear(
                self.hidden_size,
                d_model,
                mp_group=fleet.get_hybrid_communicate_group().get_model_parallel_group(),
                input_is_parallel=True,
                has_bias=True,
            )
            set_weight_attrs(self.linear_fc1.weight, {"output_dim": True})  # Column segmentation
            set_weight_attrs(self.linear_fc2.weight, {"output_dim": False})

            set_weight_attrs(self.linear_fc1.bias, {"output_dim": True})
        else:
            self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size, bias_attr=True)
            self.linear_fc2 = nn.Linear(self.hidden_size, d_model, bias_attr=True)

        set_weight_attrs(self.linear_fc1.weight, {"weight_need_transpose": model_format == "torch"})
        set_weight_attrs(self.linear_fc2.weight, {"weight_need_transpose": model_format == "torch"})

        self.act_fn = nn.GELU()

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        if self.use_postshuffle_norm:
            hidden_states = self.norm(hidden_states.view(-1, self.hidden_size))
        else:
            hidden_states = self.norm(hidden_states).view(-1, self.hidden_size)

        hidden_states_parallel = self.linear_fc1(hidden_states)
        hidden_states_parallel = self.act_fn(hidden_states_parallel)
        out = self.linear_fc2(hidden_states_parallel)
        return out


class Qwen3VisionBlock(nn.Layer):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        hidden_act: str = "gelu_tanh",
        tensor_model_parallel_size: int = 1,
        tensor_parallel_rank: int = 0,
        model_format: str = "",
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, epsilon=1e-6)
        self.norm2 = nn.LayerNorm(dim, epsilon=1e-6)
        self.attn = VisionFlashAttention2(
            dim=dim,
            num_heads=num_heads,
            tensor_model_parallel_size=tensor_model_parallel_size,
            tensor_parallel_rank=tensor_parallel_rank,
            model_format=model_format,
        )
        self.mlp = Qwen3VisionMLP(
            dim=dim,
            hidden_dim=mlp_hidden_dim,
            hidden_act=hidden_act,
            tensor_model_parallel_size=tensor_model_parallel_size,
            model_format=model_format,
        )

    def forward(
        self,
        hidden_states: paddle.Tensor,
        cu_seqlens: paddle.Tensor,
        max_seqlen: int,
        rotary_pos_emb: paddle.Tensor,
    ) -> paddle.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens,
            max_seqlen,
            rotary_pos_emb,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen3VisionTransformerPretrainedModel(PretrainedModel):
    """Qwen3 vision encoder."""

    config_class = Qwen3VisionTransformerConfig

    def __init__(self, config, prefix_name: str = "") -> None:
        vision_config = config.vision_config
        super().__init__(vision_config)
        self.prefix_name = prefix_name
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.temporal_patch_size = vision_config.temporal_patch_size
        self.deepstack_visual_indexes = vision_config.deepstack_visual_indexes
        self.num_position_embeddings = vision_config.num_position_embeddings
        self.num_grid_per_side = int(max(self.num_position_embeddings, 1) ** 0.5)
        model_format = getattr(config, "model_format", "")
        self.patch_embed = Qwen3VisionPatchEmbed(
            patch_size=vision_config.patch_size,
            temporal_patch_size=vision_config.temporal_patch_size,
            in_channels=vision_config.in_channels,
            hidden_size=vision_config.hidden_size,
            model_format=model_format,
        )
        self.pos_embed = nn.Embedding(self.num_position_embeddings, vision_config.hidden_size)

        head_dim = vision_config.hidden_size // vision_config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        self.merger = Qwen3VisionPatchMerger(
            d_model=vision_config.out_hidden_size,
            context_dim=vision_config.hidden_size,
            spatial_merge_size=self.spatial_merge_size,
            tensor_model_parallel_size=config.pretrained_config.tensor_model_parallel_size,
            use_postshuffle_norm=False,
            norm_eps=1e-6,
            model_format=model_format,
        )

        self.deepstack_merger_list = nn.LayerList(
            [
                Qwen3VisionPatchMerger(
                    d_model=vision_config.out_hidden_size,
                    context_dim=vision_config.hidden_size,
                    spatial_merge_size=self.spatial_merge_size,
                    tensor_model_parallel_size=config.pretrained_config.tensor_model_parallel_size,
                    use_postshuffle_norm=True,
                    norm_eps=1e-6,
                    model_format=model_format,
                )
                for _ in self.deepstack_visual_indexes
            ]
        )

        self.blocks = nn.LayerList(
            [
                Qwen3VisionBlock(
                    dim=vision_config.hidden_size,
                    num_heads=vision_config.num_heads,
                    mlp_hidden_dim=vision_config.intermediate_size,
                    hidden_act=vision_config.hidden_act,
                    tensor_model_parallel_size=config.pretrained_config.tensor_model_parallel_size,
                    tensor_parallel_rank=config.pretrained_config.tensor_parallel_rank,
                    model_format=model_format,
                )
                for _ in range(vision_config.depth)
            ]
        )

        self.out_hidden_size = vision_config.out_hidden_size * (1 + len(self.deepstack_visual_indexes))
        # self._set_model_format_attrs(model_format)

    def _set_model_format_attrs(self, model_format):
        if model_format is None:
            return
        for name, param in self.named_parameters():
            if "weight" in name and len(param.shape) == 2:
                logger.info(f"[Vision] {name} need to be transposed weight.")
                set_weight_attrs(param, {"weight_need_transpose": model_format == "torch"})

    @property
    def dtype(self) -> paddle.dtype:
        return self.patch_embed.proj.weight.dtype

    def fast_pos_embed_interpolate(self, grid_thw: list[list[int]]) -> paddle.Tensor:
        num_grid_per_side = self.num_grid_per_side
        merge_size = self.spatial_merge_size
        hidden_dim = self.pos_embed.weight.shape[-1]
        outputs = []

        for t, h, w in grid_thw:
            h_idxs = paddle.linspace(0, num_grid_per_side - 1, h, dtype="float32")
            w_idxs = paddle.linspace(0, num_grid_per_side - 1, w, dtype="float32")

            h_floor = paddle.floor(h_idxs).astype("int64")
            w_floor = paddle.floor(w_idxs).astype("int64")
            h_ceil = paddle.clip(h_floor + 1, max=num_grid_per_side - 1)
            w_ceil = paddle.clip(w_floor + 1, max=num_grid_per_side - 1)

            dh = h_idxs - paddle.cast(h_floor, "float32")
            dw = w_idxs - paddle.cast(w_floor, "float32")

            dh_grid, dw_grid = paddle.meshgrid(dh, dw)
            h_floor_grid, w_floor_grid = paddle.meshgrid(h_floor, w_floor)
            h_ceil_grid, w_ceil_grid = paddle.meshgrid(h_ceil, w_ceil)

            w11 = dh_grid * dw_grid
            w10 = dh_grid - w11
            w01 = dw_grid - w11
            w00 = 1.0 - dh_grid - w01

            h_grid = paddle.stack([h_floor_grid, h_floor_grid, h_ceil_grid, h_ceil_grid])
            w_grid = paddle.stack([w_floor_grid, w_ceil_grid, w_floor_grid, w_ceil_grid])
            h_grid_idx = h_grid * num_grid_per_side

            indices = (h_grid_idx + w_grid).reshape([4, -1])
            weights = paddle.stack([w00, w01, w10, w11], axis=0).reshape([4, -1, 1]).astype(self.dtype)

            embeds = self.pos_embed(indices)
            weighted = embeds * weights
            combined = weighted.sum(axis=0)

            combined = combined.reshape([h // merge_size, merge_size, w // merge_size, merge_size, hidden_dim])
            combined = combined.transpose([0, 2, 1, 3, 4]).reshape([1, -1, hidden_dim])
            combined = combined.tile([t, 1, 1]).reshape([-1, hidden_dim])
            outputs.append(combined)

        return paddle.concat(outputs, axis=0)

    def rot_pos_emb(self, grid_thw: list[list[int]]) -> paddle.Tensor:
        pos_ids = []
        max_grid_size = 0
        for t, h, w in grid_thw:
            max_grid_size = max(max_grid_size, h, w)
            hpos_ids = paddle.arange(h).unsqueeze(1).tile([1, w])
            hpos_ids = hpos_ids.reshape(
                [
                    h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    w // self.spatial_merge_size,
                    self.spatial_merge_size,
                ]
            )
            hpos_ids = hpos_ids.transpose([0, 2, 1, 3]).reshape([-1])

            wpos_ids = paddle.arange(w).unsqueeze(0).tile([h, 1])
            wpos_ids = wpos_ids.reshape(
                [
                    h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    w // self.spatial_merge_size,
                    self.spatial_merge_size,
                ]
            )
            wpos_ids = wpos_ids.transpose([0, 2, 1, 3]).reshape([-1])
            pos_ids.append(paddle.stack([hpos_ids, wpos_ids], axis=-1).tile([t, 1]))

        pos_ids = paddle.concat(pos_ids, axis=0)
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].reshape([pos_ids.shape[0], -1])
        return rotary_pos_emb

    def _build_cu_seqlens(self, grid_thw: list[list[int]]) -> paddle.Tensor:
        grid_tensor = paddle.to_tensor(grid_thw, dtype="int32")
        per_frame = paddle.repeat_interleave(grid_tensor[:, 1] * grid_tensor[:, 2], grid_tensor[:, 0])
        cu_seqlens = paddle.cumsum(per_frame, axis=0, dtype="int32")
        cu_seqlens = paddle.concat([paddle.zeros([1], dtype="int32"), cu_seqlens])
        return cu_seqlens

    def compute_attn_mask_seqlen(self, cu_seqlens: paddle.Tensor) -> int:
        if cu_seqlens.shape[0] <= 1:
            return 0
        diffs = cu_seqlens[1:] - cu_seqlens[:-1]
        return diffs.max().item()

    def forward(self, hidden_states: paddle.Tensor, grid_thw: paddle.Tensor | list, num_pad: int = 0) -> paddle.Tensor:
        if isinstance(grid_thw, paddle.Tensor):
            grid_list = grid_thw.astype("int32").numpy().tolist()
        else:
            grid_list = grid_thw

        hidden_states = self.patch_embed(hidden_states)
        pos_embeds = self.fast_pos_embed_interpolate(grid_list)
        hidden_states = hidden_states + paddle.cast(pos_embeds, hidden_states.dtype)
        rotary_pos_emb = self.rot_pos_emb(grid_list)

        cu_seqlens = self._build_cu_seqlens(grid_list)
        max_seqlen = self.compute_attn_mask_seqlen(cu_seqlens)

        deepstack_features = []
        for layer_id, block in enumerate(self.blocks):
            hidden_states = block(hidden_states, cu_seqlens, max_seqlen, rotary_pos_emb)
            if layer_id in self.deepstack_visual_indexes:
                ds_idx = self.deepstack_visual_indexes.index(layer_id)
                deepstack_features.append(self.deepstack_merger_list[ds_idx](hidden_states))

        hidden_states = self.merger(hidden_states)
        if deepstack_features:
            hidden_states = paddle.concat([hidden_states] + deepstack_features, axis=1)
        return hidden_states

    def extract_feature(self, hidden_states: paddle.Tensor, grid_thw: paddle.Tensor) -> paddle.Tensor:
        return self.forward(hidden_states, grid_thw)

    @classmethod
    def _get_tensor_parallel_mappings(cls, config, is_split=True):
        return {}
        # from paddleformers.transformers.conversion_utils import split_or_merge_func

        # from fastdeploy.model_executor.models.tp_utils import build_expanded_keys

        # fn = split_or_merge_func(
        #     is_split=is_split,
        #     tensor_model_parallel_size=config.tensor_model_parallel_size,
        #     tensor_parallel_rank=config.tensor_parallel_rank,
        # )

        # vision_config = config.vision_config
        # tp_degree = getattr(config, "tensor_model_parallel_size", 1)
        # tp_rank = getattr(config, "tensor_parallel_rank", 0)

        # def split_qkv_weight(weight):
        #     hidden = vision_config.hidden_size
        #     head_dim = hidden // vision_config.num_heads
        #     weight = weight.reshape([hidden, 3, vision_config.num_heads, head_dim])
        #     weight = np.split(weight, tp_degree, axis=2)[tp_rank]
        #     return weight.reshape([hidden, -1])

        # def split_qkv_bias(bias):
        #     head_dim = vision_config.hidden_size // vision_config.num_heads
        #     bias = bias.reshape([3, vision_config.num_heads, head_dim])
        #     bias = np.split(bias, tp_degree, axis=1)[tp_rank]
        #     return bias.reshape([-1])

        # base_actions = {
        #     "visual.blocks.0.attn.proj.weight": partial(fn, is_column=False),
        #     "visual.blocks.0.mlp.linear_fc1.weight": partial(fn, is_column=True),
        #     "visual.blocks.0.mlp.linear_fc1.bias": partial(fn, is_column=True),
        #     "visual.blocks.0.mlp.linear_fc2.weight": partial(fn, is_column=False),
        #     "visual.blocks.0.attn.qkv.weight": split_qkv_weight,
        #     "visual.blocks.0.attn.qkv.bias": split_qkv_bias,
        #     "visual.merger.linear_fc1.weight": partial(fn, is_column=True),
        #     "visual.merger.linear_fc1.bias": partial(fn, is_column=True),
        #     "visual.merger.linear_fc2.weight": partial(fn, is_column=False),
        # }

        # for idx in range(len(vision_config.deepstack_visual_indexes)):
        #     base_actions[f"visual.deepstack_merger_list.{idx}.linear_fc1.weight"] = partial(fn, is_column=True)
        #     base_actions[f"visual.deepstack_merger_list.{idx}.linear_fc1.bias"] = partial(fn, is_column=True)
        #     base_actions[f"visual.deepstack_merger_list.{idx}.linear_fc2.weight"] = partial(fn, is_column=False)

        # final_actions = {}
        # final_actions.update(
        #     build_expanded_keys(
        #         {k: v for k, v in base_actions.items() if "visual.blocks.0." in k},
        #         vision_config.depth,
        #     )
        # )
        # for k, v in base_actions.items():
        #     if "visual.blocks.0." not in k:
        #         final_actions[k] = v
        # return final_actions

    def load_state_dict(self, state_dict):
        params_dict = dict(self.named_parameters())
        buffers_dict = dict(self.named_buffers())

        prefix = f"{self.prefix_name}." if self.prefix_name else ""

        for name, param in params_dict.items():
            key = prefix + name
            if key not in state_dict:
                raise ValueError(f"Missing parameter {key} in state_dict")
            tensor = get_tensor(state_dict.pop(key))
            if tensor.shape != param.shape:
                raise ValueError(f"Shape mismatch for {key}: expected {param.shape}, got {tensor.shape}")
            param.copy_(tensor, False)

        for name, buffer in buffers_dict.items():
            key = prefix + name
            if key not in state_dict:
                continue
            tensor = get_tensor(state_dict.pop(key))
            if tensor.shape != buffer.shape:
                raise ValueError(f"Shape mismatch for buffer {key}: expected {buffer.shape}, got {tensor.shape}")
            buffer.copy_(tensor, False)
