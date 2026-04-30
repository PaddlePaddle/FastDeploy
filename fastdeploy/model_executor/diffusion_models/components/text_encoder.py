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
Text encoding pipeline for Flux / SD3.

Flux uses two text encoders:
  - CLIP-L (clip_l): pooled embeddings → timestep conditioning
  - T5-XXL (t5): sequence embeddings → cross-attention

SD3 adds CLIP-G as a third encoder (Phase 2).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import paddle
import paddle.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class TextEncoderOutput:
    """Output container for the text encoding pipeline.

    Attributes:
        prompt_embeds: Sequence embeddings for cross-attention [B, seq_len, dim].
        pooled_prompt_embeds: Pooled embeddings for timestep conditioning [B, pooled_dim].
    """

    prompt_embeds: paddle.Tensor
    pooled_prompt_embeds: paddle.Tensor


class CLIPTextEncoder(nn.Layer):
    """Wrapper for a single CLIP text encoder (CLIP-L or CLIP-G).

    Loads from PaddleNLP / HuggingFace checkpoint and provides:
      - Tokenization via the associated tokenizer
      - Forward pass returning both sequence and pooled embeddings
    """

    def __init__(self) -> None:
        super().__init__()
        self.model: Optional[nn.Layer] = None
        self.tokenizer = None
        self.max_length: int = 77  # CLIP default

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        subfolder: str = "text_encoder",
        dtype: paddle.dtype = paddle.float32,
        max_length: int = 77,
    ) -> "CLIPTextEncoder":
        """Load a pretrained CLIP text encoder.

        Args:
            model_path: Root model directory.
            subfolder: Subfolder name for this encoder.
            dtype: Weight dtype.
            max_length: Maximum token sequence length.

        Returns:
            Initialized CLIPTextEncoder.
        """
        encoder = cls()
        encoder.max_length = max_length
        encoder_path = os.path.join(model_path, subfolder)

        if not os.path.isdir(encoder_path):
            logger.warning("CLIP encoder path not found: %s", encoder_path)
            return encoder

        # 尝试加载 PaddleNLP CLIPTextModel (Try loading PaddleNLP CLIPTextModel)
        try:
            from paddlenlp.transformers import CLIPTextModel, CLIPTokenizer

            encoder.tokenizer = CLIPTokenizer.from_pretrained(encoder_path)
            encoder.model = CLIPTextModel.from_pretrained(encoder_path, dtype=dtype)
            encoder.model.eval()
            logger.info("Loaded CLIP encoder from %s", encoder_path)
        except (ImportError, OSError, ValueError) as e:
            logger.warning("Failed to load CLIP encoder from %s: %s", encoder_path, e)

        return encoder

    def forward(self, text: List[str]) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """Encode text prompts.

        Args:
            text: List of prompt strings.

        Returns:
            Tuple of (sequence_embeds [B, seq_len, dim], pooled_embeds [B, dim]).
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("CLIP encoder not loaded. Call from_pretrained first.")

        tokens = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pd",
        )
        outputs = self.model(**tokens)

        # CLIPTextModel returns (last_hidden_state, pooler_output)
        sequence_embeds = outputs[0]
        pooled_embeds = outputs[1]
        return sequence_embeds, pooled_embeds


class T5TextEncoder(nn.Layer):
    """Wrapper for T5-XXL text encoder (sequence embeddings for cross-attention)."""

    def __init__(self) -> None:
        super().__init__()
        self.model: Optional[nn.Layer] = None
        self.tokenizer = None
        self.max_length: int = 512

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        subfolder: str = "text_encoder_2",
        dtype: paddle.dtype = paddle.float32,
        max_length: int = 512,
    ) -> "T5TextEncoder":
        """Load a pretrained T5 text encoder.

        Args:
            model_path: Root model directory.
            subfolder: Subfolder name for T5 encoder.
            dtype: Weight dtype.
            max_length: Maximum token sequence length.

        Returns:
            Initialized T5TextEncoder.
        """
        encoder = cls()
        encoder.max_length = max_length
        encoder_path = os.path.join(model_path, subfolder)

        if not os.path.isdir(encoder_path):
            logger.warning("T5 encoder path not found: %s", encoder_path)
            return encoder

        try:
            from paddlenlp.transformers import T5EncoderModel, T5Tokenizer

            encoder.tokenizer = T5Tokenizer.from_pretrained(encoder_path)
            encoder.model = T5EncoderModel.from_pretrained(encoder_path, dtype=dtype)
            encoder.model.eval()
            logger.info("Loaded T5 encoder from %s", encoder_path)
        except (ImportError, OSError, ValueError) as e:
            logger.warning("Failed to load T5 encoder from %s: %s", encoder_path, e)

        return encoder

    def forward(self, text: List[str]) -> paddle.Tensor:
        """Encode text prompts to sequence embeddings.

        Args:
            text: List of prompt strings.

        Returns:
            Sequence embeddings [B, seq_len, dim].
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("T5 encoder not loaded. Call from_pretrained first.")

        tokens = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pd",
        )
        outputs = self.model(**tokens)
        return outputs[0]  # last_hidden_state


class TextEncoderPipeline:
    """Combined text encoding pipeline for Flux / SD3.

    Flux uses two text encoders:
      - CLIP-L: pooled embeddings (768d) → timestep conditioning
      - T5-XXL: sequence embeddings → cross-attention

    SD3 uses three text encoders:
      - CLIP-L: pooled embeddings (768d)
      - CLIP-G: pooled embeddings (1280d)
      → CLIP-L + CLIP-G concatenated = 2048d pooled projection
      - T5-XXL: sequence embeddings → cross-attention
    """

    def __init__(
        self,
        clip_encoder: Optional[CLIPTextEncoder] = None,
        clip_g_encoder: Optional[CLIPTextEncoder] = None,
        t5_encoder: Optional[T5TextEncoder] = None,
        max_sequence_length: int = 512,
    ) -> None:
        self.clip_encoder = clip_encoder
        self.clip_g_encoder = clip_g_encoder
        self.t5_encoder = t5_encoder
        self.max_sequence_length = max_sequence_length

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        dtype: paddle.dtype = paddle.float32,
        max_sequence_length: int = 512,
        model_type: str = "flux",
    ) -> "TextEncoderPipeline":
        """Load all text encoders for a Flux or SD3 model.

        Flux layout:
          - text_encoder/   → CLIP-L
          - text_encoder_2/ → T5-XXL

        SD3 layout:
          - text_encoder/   → CLIP-L
          - text_encoder_2/ → CLIP-G
          - text_encoder_3/ → T5-XXL

        Args:
            model_path: Root model directory.
            dtype: Weight dtype.
            max_sequence_length: Max T5 sequence length.
            model_type: "flux" or "sd3".

        Returns:
            Initialized TextEncoderPipeline.
        """
        # CLIP-L is always text_encoder/ for both Flux and SD3
        clip_encoder = CLIPTextEncoder.from_pretrained(
            model_path,
            subfolder="text_encoder",
            dtype=dtype,
        )

        clip_g_encoder = None
        if model_type == "sd3":
            # SD3: text_encoder_2 = CLIP-G, text_encoder_3 = T5-XXL
            clip_g_encoder = CLIPTextEncoder.from_pretrained(
                model_path,
                subfolder="text_encoder_2",
                dtype=dtype,
                max_length=77,
            )
            t5_encoder = T5TextEncoder.from_pretrained(
                model_path,
                subfolder="text_encoder_3",
                dtype=dtype,
                max_length=max_sequence_length,
            )
        else:
            # Flux: text_encoder_2 = T5-XXL
            t5_encoder = T5TextEncoder.from_pretrained(
                model_path,
                subfolder="text_encoder_2",
                dtype=dtype,
                max_length=max_sequence_length,
            )

        return cls(
            clip_encoder=clip_encoder,
            clip_g_encoder=clip_g_encoder,
            t5_encoder=t5_encoder,
            max_sequence_length=max_sequence_length,
        )

    @paddle.no_grad()
    def encode(
        self,
        prompt: List[str],
        dtype: paddle.dtype = paddle.float32,
    ) -> TextEncoderOutput:
        """Encode prompts through all text encoders.

        Args:
            prompt: List of text prompts.
            dtype: Output tensor dtype.

        Returns:
            TextEncoderOutput with prompt_embeds and pooled_prompt_embeds.
        """
        # CLIP-L → pooled embeddings for timestep conditioning
        pooled_prompt_embeds = None
        if self.clip_encoder is not None and self.clip_encoder.model is not None:
            _, pooled_prompt_embeds = self.clip_encoder(prompt)
            pooled_prompt_embeds = pooled_prompt_embeds.cast(dtype)
        elif self.clip_encoder is not None and self.clip_encoder.model is None:
            logger.warning(
                "CLIP-L encoder was requested but failed to load. "
                "Falling back to zero tensors — generation quality will be degraded."
            )

        # CLIP-G → pooled embeddings (SD3 only, concat with CLIP-L)
        if self.clip_g_encoder is not None and self.clip_g_encoder.model is not None:
            _, pooled_g = self.clip_g_encoder(prompt)
            pooled_g = pooled_g.cast(dtype)
            if pooled_prompt_embeds is not None:
                # SD3: CLIP-L (768d) + CLIP-G (1280d) = 2048d
                pooled_prompt_embeds = paddle.concat([pooled_prompt_embeds, pooled_g], axis=-1)
            else:
                pooled_prompt_embeds = pooled_g
        elif self.clip_g_encoder is not None and self.clip_g_encoder.model is None:
            logger.warning(
                "CLIP-G encoder was requested but failed to load. "
                "SD3 pooled embeddings will be incomplete — generation quality will be degraded."
            )
            # Pad CLIP-L (768d) → 2048d to match SD3 text_proj input dimension
            if pooled_prompt_embeds is not None:
                pad_dim = 2048 - pooled_prompt_embeds.shape[-1]
                if pad_dim > 0:
                    pooled_prompt_embeds = paddle.concat(
                        [pooled_prompt_embeds, paddle.zeros([pooled_prompt_embeds.shape[0], pad_dim], dtype=dtype)],
                        axis=-1,
                    )

        # T5-XXL → sequence embeddings for cross-attention
        prompt_embeds = None
        if self.t5_encoder is not None and self.t5_encoder.model is not None:
            prompt_embeds = self.t5_encoder(prompt)
            prompt_embeds = prompt_embeds.cast(dtype)
        elif self.t5_encoder is not None and self.t5_encoder.model is None:
            logger.warning(
                "T5 encoder was requested but failed to load. "
                "Falling back to zero tensors — generation quality will be degraded."
            )

        # 回退：生成零张量 (Fallback: generate zero tensors if encoders missing)
        batch_size = len(prompt)
        if pooled_prompt_embeds is None:
            # SD3 needs 2048d (CLIP-L 768 + CLIP-G 1280), Flux needs 768d
            pooled_dim = 2048 if self.clip_g_encoder is not None else 768
            pooled_prompt_embeds = paddle.zeros([batch_size, pooled_dim], dtype=dtype)
        if prompt_embeds is None:
            prompt_embeds = paddle.zeros([batch_size, self.max_sequence_length, 4096], dtype=dtype)

        return TextEncoderOutput(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
        )
