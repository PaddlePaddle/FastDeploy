from paddle.base import core
import paddlenlp_ops

def rms_norm_npu(
    x,
    norm_weight,
    norm_bias,
    epsilon,
    begin_norm_axis,
    bias,
    residual,
    quant_scale,
    quant_round_type,
    quant_max_bound,
    quant_min_bound,
):

    out, residual_out = core.eager._run_custom_op(
        "atb_rms_norm",
        x,
        norm_weight,
        residual,
        epsilon
    )

    return out, residual_out
