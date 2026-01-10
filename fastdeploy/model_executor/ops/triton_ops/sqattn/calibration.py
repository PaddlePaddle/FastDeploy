import pickle

import paddle
import os

# from paddle_utils import *
# from .pred import build_chat, post_process

# from paddlenlp import datasets

def get_static_calib_dataset_longbench(
    tokenizer=None
):
    # 当前路径下的longbench_samples.pkl
    current_path = os.path.dirname(os.path.abspath(__file__))
    longbench_samples_path = os.path.join(current_path, "longbench_samples.pkl")
    with open(longbench_samples_path, "rb") as f:
        samples = []
        loaded_data = pickle.load(f)
        for data in loaded_data:
            tokenized_data = tokenizer(data, truncation=False, return_tensors="pd")
            samples.append(tokenized_data["input_ids"])
        return samples, None
