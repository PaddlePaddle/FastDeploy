# Diffusion Models — Flux & SD3 Implementation

FastDeploy supports text-to-image generation via two diffusion model architectures:
**Flux** (Black Forest Labs) and **Stable Diffusion 3** (Stability AI).

## Supported Models

| Model | Type | Architecture | Parameters |
|-------|------|-------------|------------|
| FLUX.1-dev | `flux` | Double/Single-stream DiT | 11.89B |
| FLUX.1-schnell | `flux` | Double/Single-stream DiT | 11.89B |
| SD3-Medium | `sd3` | Joint MMDiT | 2B |
| SD3.5-Large | `sd3` | Joint MMDiT | 8B |

## Quick Start

### Flux Example

```python
from fastdeploy.model_executor.diffusion_models import DiffusionConfig, DiffusionEngine

config = DiffusionConfig(
    model_name_or_path="black-forest-labs/FLUX.1-dev",
    model_type="flux",
    dtype="bfloat16",
    image_height=1024,
    image_width=1024,
    num_inference_steps=28,
    guidance_scale=3.5,
)

engine = DiffusionEngine(config)
engine.load()

images = engine.generate(
    prompt="A photorealistic cat sitting on a cloud at sunset",
    seed=42,
)
images[0].save("flux_output.png")
```

### SD3 Example

```python
from fastdeploy.model_executor.diffusion_models import DiffusionConfig, DiffusionEngine

config = DiffusionConfig(
    model_name_or_path="stabilityai/stable-diffusion-3-medium",
    model_type="sd3",
    dtype="float16",
    image_height=1024,
    image_width=1024,
    num_inference_steps=28,
    guidance_scale=7.0,
)

engine = DiffusionEngine(config)
engine.load()

images = engine.generate(
    prompt="A watercolor painting of a mountain village",
    seed=42,
)
images[0].save("sd3_output.png")
```

## Configuration

`DiffusionConfig` accepts:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name_or_path` | `str` | — | Path to model directory (HuggingFace format) |
| `model_type` | `"flux"` / `"sd3"` | `"flux"` | Architecture type |
| `dtype` | `str` | `"bfloat16"` | Weight precision (`float16`, `bfloat16`, `float32`) |
| `image_height` | `int` | `1024` | Output image height |
| `image_width` | `int` | `1024` | Output image width |
| `num_inference_steps` | `int` | `28` | Denoising steps |
| `guidance_scale` | `float` | `3.5` | CFG scale (Flux: 3.5, SD3: 7.0 recommended) |
| `max_sequence_length` | `int` | `512` | T5 text encoder max tokens |
| `vae_path` | `str` | `None` | Override VAE directory (default: `{model_path}/vae`) |

## Architecture Overview

### Flux (Double/Single-Stream Transformer)

```
Text Prompts → CLIP-L (pooled) + T5-XXL (sequence)
                  ↓
Noise (packed: [B, seq, 64]) + RoPE position IDs
                  ↓
┌─ 19× Double-Stream Blocks (joint text+image attention) ─┐
│   txt_attn ← concat(txt, img) → img_attn                │
│   txt_ff ─── separate FFN ──── img_ff                    │
└──────────────────────────────────────────────────────────┘
                  ↓
┌─ 38× Single-Stream Blocks (fused attention) ────────────┐
│   concat(img, txt) → self-attention → FFN                │
└──────────────────────────────────────────────────────────┘
                  ↓
Unpack latents → VAE decode → PIL Image
```

### SD3 (Joint MMDiT — Multi-Modal Diffusion Transformer)

```
Text Prompts → CLIP-L+G (pooled: 2048d) + T5-XXL (sequence: 4096d)
                  ↓
Noise (spatial: [B, 16, H/8, W/8]) → PatchEmbed → [B, N, 1536]
                  ↓
┌─ 24× Joint Transformer Blocks ─────────────────────────┐
│   AdaLN-Zero modulation (6 params from timestep embed)   │
│   Joint attention: concat(context, hidden) QKV            │
│   QK RMSNorm → scaled_dot_product_attention               │
│   Split output → separate FFN for context + hidden        │
│   (Last block: context_pre_only — no context output)      │
└──────────────────────────────────────────────────────────┘
                  ↓
AdaLN final norm → Linear projection → Unpatchify
                  ↓
Spatial latents [B, 16, H/8, W/8] → VAE decode → PIL Image
```

## Weight Format

The module supports two weight formats:

1. **PaddlePaddle native** (`.pdparams`): Loaded directly via `paddle.load()`
2. **SafeTensors** (`.safetensors`): Loaded via `safetensors` library with automatic
   PyTorch → Paddle key mapping

Directory structure expected:
```
model_root/
├── config.json              # Model config
├── transformer/
│   ├── config.json          # Transformer config
│   └── diffusion_pytorch_model.safetensors  # or model_state.pdparams
├── vae/
│   ├── config.json          # VAE config
│   └── diffusion_pytorch_model.safetensors
├── text_encoder/            # CLIP-L
├── text_encoder_2/          # CLIP-G (SD3) or T5-XXL (Flux)
└── text_encoder_3/          # T5-XXL (SD3 only)
```

## VAE Architecture

Both Flux and SD3 use a 16-channel KL-VAE with ResNet blocks and attention:

| Component | Details |
|-----------|---------|
| Encoder | Conv2D → 4 downsample stages × 2 ResBlocks → Mid (ResNet + Attn + ResNet) → GroupNorm → Conv2D |
| Decoder | Conv2D → Mid → 4 upsample stages × 3 ResBlocks → GroupNorm → Conv2D |
| Channels | 128 → 256 → 512 → 512 |
| Latent | 16 channels, 8× spatial compression |

Scaling:
- Flux: `scaling_factor=0.3611`, `shift_factor=0.0`
- SD3: `scaling_factor=1.5305`, `shift_factor=0.0609`

## Parallel and Quantization Adaptation

The `parallel.py` module provides integration hooks for FastDeploy's tensor-parallel
and weight-quantization infrastructure.

### Tensor Parallelism

DiT blocks contain attention QKV projections and MLP layers that are natural
candidates for tensor-parallel sharding:

| Layer Pattern | TP Strategy | Description |
|---------------|-------------|-------------|
| `attn_qkv`, `attn_qkv_context` | Column-parallel | Split QKV output across TP ranks |
| `mlp.0`, `mlp_context.0` | Column-parallel | Split MLP gate/up projection |
| `attn_out`, `attn_out_context` | Row-parallel | Reduce attention output across ranks |
| `mlp.2`, `mlp_context.2` | Row-parallel | Reduce MLP down projection |
| `proj_out` | Row-parallel | Final output projection (SD3) |

```python
from fastdeploy.model_executor.diffusion_models.parallel import apply_tensor_parallel

engine = DiffusionEngine(config)
engine.load()
apply_tensor_parallel(engine.transformer, fd_config)
```

On single-GPU (the default), `apply_tensor_parallel` is a no-op.

### Weight Quantization

Weight-only quantization (W8A8, W4A16) can be applied to DiT linear layers
≥256 columns, following the same pattern as LLM model quantization:

```python
from fastdeploy.model_executor.diffusion_models.parallel import apply_weight_quantization

engine = DiffusionEngine(config)
engine.load()
apply_weight_quantization(engine.transformer, quant_method="w8a8")
```

The VAE and text encoders are typically NOT quantized (small relative to the
transformer and sensitive to precision loss).
