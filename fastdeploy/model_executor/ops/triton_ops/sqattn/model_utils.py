import paddle
import paddleformers
from paddleformers.transformers.llama import LlamaForCausalLM
from paddleformers.transformers.qwen2 import Qwen2ForCausalLM

def move_embed(model, device):
    if isinstance(model, (LlamaForCausalLM, Qwen2ForCausalLM)):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        model.model.rotary_emb = model.model.rotary_emb.to(device)
    elif isinstance(model, paddleformers.transformers.opt.OPTForCausalLM):
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(device)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(
            device
        )
    elif isinstance(model, paddleformers.transformers.bloom.BloomForCausalLM):
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(device)
        model.transformer.word_embeddings_layernorm = (
            model.transformer.word_embeddings_layernorm.to(device)
        )
    elif "mpt" in str(model.__class__).lower():
        model.transformer.wte = model.transformer.wte.to(device)
        model.transformer.emb_drop = model.transformer.emb_drop.to(device)
    elif "falcon" in str(model.__class__).lower():
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(device)
    elif "bigcode" in str(model.__class__).lower():
        model.transformer.wte = model.transformer.wte.to(device)
        model.transformer.wpe = model.transformer.wpe.to(device)
        model.transformer.drop = model.transformer.drop.to(device)
    elif "neox" in str(model.__class__).lower():
        model.gpt_neox.embed_in = model.gpt_neox.embed_in.to(device)
        model.gpt_neox.emb_dropout = model.gpt_neox.emb_dropout.to(device)
        model.embed_out = model.embed_out.to(device)
    elif "llavallamamodel" in str(model.__class__).lower():
        model.llm.model.embed_tokens = model.llm.model.embed_tokens.to(device)
    else:
        raise NotImplementedError(type(model))


def get_blocks(model):
    if isinstance(model, (LlamaForCausalLM, Qwen2ForCausalLM)):
        layers = model.model.layers
    elif model.__class__.__name__ == "LlavaLlamaForCausalLM":
        layers = [
            model.model.layers,
            model.model.vision_tower.vision_tower.vision_model.encoder.layers,
        ]
    elif isinstance(model, paddleformers.transformers.opt.OPTForCausalLM):
        layers = model.model.decoder.layers
    elif isinstance(model, paddleformers.transformers.bloom.BloomForCausalLM):
        layers = model.transformer.h
    elif "mpt" in str(model.__class__).lower():
        layers = model.transformer.blocks
    elif "falcon" in str(model.__class__).lower():
        layers = model.transformer.h
    elif "bigcode" in str(model.__class__).lower():
        layers = model.transformer.h
    elif "neox" in str(model.__class__).lower():
        layers = model.gpt_neox.layers
    elif model.__class__.__name__ == "LlavaLlamaModel":
        layers = model.llm.model.layers
    else:
        raise NotImplementedError(type(model))
    return layers


def get_named_linears(module):
    return {
        name: m
        for name, m in module.named_sublayers(include_self=True)
        if isinstance(m, paddle.nn.Linear)
    }


@paddle.no_grad()
def batch_layer_infer(layer, batch_inps, batch_layer_kwargs, args):
    assert len(batch_inps) == len(batch_layer_kwargs)
    batch_oups = []
    for i in range(len(batch_inps)):
        inps = batch_inps[i]
        layer_kwargs = batch_layer_kwargs[i]
        oups = layer(inps, **layer_kwargs)#[0]
        batch_oups.append(oups)
    return batch_oups
