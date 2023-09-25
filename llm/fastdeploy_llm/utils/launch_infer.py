import os
import subprocess
from fastdeploy_llm.utils.logging_util import logger

infer_script_path = os.path.join(os.path.dirname(__file__), "..", 'engine.py')


def launch(device_ids, **kwargs: dict):
    keywords = [
        'model_dir', 'batch_size', 'max_seq_len', 'max_dec_len', 'num_layers',
        'num_attention_heads', 'hidden_size', 'architecture',
        'is_static_model', "decode_strategy", "is_ptuning",
        "model_prompt_dir_path"
    ]
    args = []
    missing_args = []
    for keyword in keywords:
        if keyword not in kwargs:
            missing_args.append(keyword)
        else:
            args.append('--' + keyword)
            args.append(str(kwargs[keyword]))
    args.append("--serving_pid {}".format(os.getpid()))
    if missing_args:
        raise RuntimeError(
            "Lauch paddle inference engine failed due to missing keyword args {} which required by "
            + "launch_infer.launch, please set them and try again".format(
                missing_args))

    pd_cmd = "python3 -m paddle.distributed.launch --devices {} {} {}".format(
        device_ids, infer_script_path, ' '.join(args))
    logger.info("Launch model with command: {}".format(pd_cmd))
    logger.info("Model is initializing...")
    p = subprocess.Popen(
        pd_cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid)
    return p
