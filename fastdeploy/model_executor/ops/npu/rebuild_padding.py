import paddle
from paddle.base import core
import inspect
#import paddlenlp_ops 

def rebuild_padding(
    model_output,
    cum_offsets,
    seq_lens_this_time,
    seq_lens_decoder,
    seq_lens_encoder,
    padding_offset,
    max_model_len
):  
    # Cast to float16 for NPU kernel as required, then cast back to original dtype
    # FIXME: guozr need furter check if type cast needed
    original_dtype = model_output.dtype
    model_output = paddle.cast(model_output, paddle.float16)
    
    out = core.eager._run_custom_op(
        "rebuild_padding_v2",
        model_output,
        cum_offsets,
        seq_lens_decoder,
        seq_lens_encoder,
        max_model_len
    )[0]
    
    # Cast back to original dtype to maintain consistency
    out = paddle.cast(out, original_dtype)
    

    return out
