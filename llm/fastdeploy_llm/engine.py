# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import os
import paddle
import paddle.distributed.fleet as fleet
import paddle.distributed as dist
import argparse
import paddlenlp
from paddlenlp_ops import save_with_output  # NOLINT
from multiprocessing import shared_memory
import numpy as np
import pickle
import logging
import signal

from fastdeploy_llm.utils.prefix_utils import get_prefix_caches, pad_prefix_caches, LimitedSizeDict


def is_process_running(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True


def parse_args():
    parser = argparse.ArgumentParser("ernie inference")
    parser.add_argument(
        '-sp',
        '--serving_pid',
        type=int,
        default=-1,
        help='Process id of serving framework, will terminate infer while serving pid is not running.'
    )
    parser.add_argument(
        '-m', '--model_dir', type=str, default='./output', help='model dir')
    parser.add_argument(
        '-mp', '--mp_degree', type=int, default=8, help='mp degree')
    parser.add_argument(
        '-bs', '--batch_size', type=int, default=34, help='batch size')
    parser.add_argument(
        '--max_seq_len', type=int, default=3072, help='max_seq_len')
    parser.add_argument(
        '--max_dec_len', type=int, default=1024, help='max_dec_len')
    parser.add_argument(
        '--num_layers', type=int, default=80, help='num_layers')
    parser.add_argument(
        '--num_attention_heads',
        type=int,
        default=64,
        help='num_attention_heads')
    parser.add_argument(
        '--hidden_size', type=int, default=8192, help='hidden_size')
    parser.add_argument(
        '--architecture',
        type=str,
        default='llama',
        help='llama/chatglm/bloom')
    parser.add_argument(
        '--decode_strategy',
        type=str,
        default='sampling',
        help='sampling/greedy_search')
    parser.add_argument(
        '--is_static_model',
        type=int,
        default=1,
        help="define the model is a static model or dygraph model.")
    parser.add_argument(
        "--quant_bits",
        type=int,
        default=-1,
        help="if enable weight only optimization, -1 means disable.")
    parser.add_argument("--dtype", type=str, default="float16")
    # Below are ptuning specified arguments
    parser.add_argument(
        '--is_ptuning',
        type=int,
        default=False,
        help='inference for ptuning or not')
    parser.add_argument(
        '--model_prompt_dir_path',
        type=str,
        default='./prompt_embedding',
        help='directory for storing prefix caches')
    parser.add_argument(
        '--max_prefix_len',
        type=int,
        default=128,
        help='max length for prefix caches')
    args = parser.parse_args()
    return args


args = parse_args()


def init_dist_env(world_size, seed=20):
    # start to init distributed env
    strategy = fleet.DistributedStrategy()

    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": world_size,
        "pp_degree": 1,
        "sharding_degree": 1,
    }

    # Set control in tensor parallel
    strategy.tensor_parallel_configs = {"tensor_init_seed": seed}
    fleet.init(is_collective=True, strategy=strategy)


nranks = dist.get_world_size()
init_dist_env(nranks)
rank = fleet.worker_index()
decode_strategy = args.decode_strategy
quant_bits = args.quant_bits
serving_pid = args.serving_pid

if args.is_ptuning:
    size_limit = 50 * fleet.worker_num()
    model_prompt_dict = LimitedSizeDict(size_limit=size_limit)
    max_prefix_len = args.max_prefix_len
    prompt_num = args.max_prefix_len

cache_kvs = []
for _ in range(args.num_layers):
    cache_kvs.append(
        paddle.cast(
            paddle.to_tensor(
                np.zeros(
                    (2, args.batch_size, args.num_attention_heads // nranks,
                     args.max_seq_len + args.max_dec_len, args.hidden_size //
                     args.num_attention_heads),
                    dtype='float32')),
            args.dtype))

pre_ids = paddle.to_tensor(np.full((args.batch_size, 2048), -1, dtype='int64'))
tgt_generation_mask = paddle.zeros(
    shape=[args.batch_size, 1, 1, args.max_seq_len + args.max_dec_len],
    dtype=args.dtype)
if "chatglm" in args.architecture:
    #TODO JiangJiajun
    attention_mask = paddle.zeros(
        shape=(args.batch_size, 1, args.max_seq_len + args.max_dec_len,
               args.max_seq_len + args.max_dec_len),
        dtype=args.dtype)
    tgt_pos = paddle.ones(shape=(args.batch_size, 2, 1), dtype="int64")
    position_ids = paddle.full(
        shape=[args.batch_size, 2, args.max_seq_len],
        fill_value=0,
        dtype='int64')
else:
    attention_mask = paddle.zeros(
        shape=(args.batch_size, 1, args.max_seq_len + args.max_dec_len,
               args.max_seq_len + args.max_dec_len),
        dtype=args.dtype)
    position_ids = paddle.full(
        shape=[args.batch_size, args.max_seq_len], fill_value=0, dtype='int64')


#count = 0
class DygraphEngine(object):
    def __init__(self, model_dir, architecture, mp_degree):
        if mp_degree == 1:
            self.nranks = 1
            self.rank = 0
        else:
            self.nranks = fleet.worker_num()
            self.rank = fleet.worker_index()
        if "llama" in architecture:
            from paddlenlp.experimental.transformers import LlamaForCausalLMInferenceModel
            config = paddlenlp.transformers.AutoConfig.from_pretrained(
                model_dir)
            config.tensor_parallel_degree = self.nranks
            config.tensor_parallel_rank = self.rank
            config.quant_bits = quant_bits
            self.model = LlamaForCausalLMInferenceModel.from_pretrained(
                model_dir, dtype="float16", config=config)
            self.model.eval()
        else:
            raise Exception("The architecture {} is not support yet.".format(
                architecture))

    def predict(self, batch_data_dict):
        for k, v in batch_data_dict.items():
            if not isinstance(v, paddle.Tensor):
                batch_data_dict[k] = paddle.to_tensor(v)
        finished_ids, step_idx, stop_flags, sequence_lengths, tgt_pos = self.model.generate(
            **batch_data_dict,
            cache_kvs=cache_kvs,
            pre_ids=pre_ids,
            decode_strategy=decode_strategy)
        return finished_ids.numpy(), step_idx.numpy(), stop_flags.numpy(
        ), sequence_lengths.numpy(), tgt_pos.numpy()


class InferenceEngine(object):
    """
    Model Parallel Inference Engine

    Args:
        model_dir (string): root directory of inference model
        mp_degree (int): model parallel size
    """

    def __init__(self, model_dir, mp_degree=1):
        self.model_dir = model_dir
        self.mp_degree = mp_degree

        if mp_degree == 1:
            self.nranks = 1
            self.rank = 0
        else:
            self.nranks = fleet.worker_num()
            self.rank = fleet.worker_index()

        self._init_predictor()

    def _init_predictor(self):
        """predictor init"""
        device_id = self.rank % 8
        self.model_file = os.path.join(self.model_dir, f"model.pdmodel")
        self.param_file = os.path.join(self.model_dir, f"model.pdiparams")
        config = paddle.inference.Config(self.model_file, self.param_file)

        config.switch_ir_optim(False)
        gpu_mem = paddle.device.cuda.get_device_properties(
            device_id).total_memory / 1024 / 1024 * 0.92
        config.enable_use_gpu(int(gpu_mem), device_id)

        # distributed config
        if self.mp_degree >= 1:
            trainer_endpoints = fleet.worker_endpoints()
            current_endpoint = trainer_endpoints[self.rank]
            dist_config = config.dist_config()
            dist_config.set_ranks(self.nranks, self.rank)
            dist_config.set_endpoints(trainer_endpoints, current_endpoint)
            dist_config.enable_dist_model(True)
            dist_config.set_comm_init_config(
                os.path.join(args.model_dir, "rank_mapping.csv"))
            config.set_dist_config(dist_config)
            print(
                f'Init distributed config with {dist_config}, mp_degree: {self.mp_degree}',
                flush=True)
        self.predictor = paddle.inference.create_predictor(config)
        self.output_names = self.predictor.get_output_names()

    def predict(self, batch_data_dict):
        """
        predict
        """
        #        dump_data = dict()

        for k, v in batch_data_dict.items():
            input_tensor = self.predictor.get_input_handle(k)
            if isinstance(v, paddle.Tensor):
                input_tensor.share_external_data(v)
#                dump_data[k] = v.numpy()
            else:
                input_tensor.copy_from_cpu(v)
#                dump_data[k] = v

#        cache_dumps = list()
        for i in range(args.num_layers):
            input_tensor = self.predictor.get_input_handle('cache_kvs_' + str(
                i))
            input_tensor.share_external_data(cache_kvs[i])

#            cache_dumps.append(cache_kvs[i].numpy())

        input_tensor = self.predictor.get_input_handle('pre_ids')
        input_tensor.share_external_data(pre_ids)
        #        dump_data["pre_ids"] = pre_ids.numpy()
        #        import pickle
        #        with open("dump_data.pkl", "wb") as f:
        #            pickle.dump([dump_data, cache_dumps], f)

        self.predictor.run()
        # NOTE: The order of return values is refered from:
        # PaddleNLP/paddlenlp/experimental/transformers/generation_utils.py
        # step_idx: step of decoder, shape is [bsz, 1]
        # stop_flogs: the flags to denote whether the generation has done, [bsz, 1]
        finished_ids, step_idx, stop_flags, sequence_lengths, tgt_pos = [
            self.predictor.get_output_handle(k)
            for k in self.predictor.get_output_names()[:6]
        ]
        return finished_ids.copy_to_cpu(), step_idx.copy_to_cpu(
        ), stop_flags.copy_to_cpu(), sequence_lengths.copy_to_cpu(
        ), tgt_pos.copy_to_cpu()


def dy_input_preprocess(inputs):
    """
    prepare input for dybatch inference
    """
    stop_flags = inputs["dyinput_flags"]
    dec_length = inputs["seq_len_decoder"]
    bsz = len(stop_flags)

    tmp = np.zeros(shape=[args.batch_size, 2, args.max_seq_len], dtype="int64")

    for i in range(bsz):
        if stop_flags[i] == 1:
            length = int(dec_length[i, 0])
            if args.is_ptuning:
                model_id = inputs['model_id'][i]
                if not model_id:
                    attention_mask[i, 0, :length, :
                                   max_prefix_len] = paddle.zeros(
                                       [1, length, max_prefix_len],
                                       dtype=args.dtype)
                else:
                    attention_mask[i, 0, :length, :
                                   max_prefix_len] = paddle.ones(
                                       [1, length, max_prefix_len],
                                       dtype=args.dtype)
                attention_mask[i, 0, :length, max_prefix_len:max_prefix_len +
                               length] = paddle.tril(
                                   paddle.ones(
                                       shape=[length, length],
                                       dtype=args.dtype))
                position_ids[i, :max_prefix_len] = 0
                position_ids[i, max_prefix_len:max_prefix_len + inputs[
                    "input_ids"].shape[1]] = paddle.arange(inputs["input_ids"]
                                                           .shape[1])
                tgt_generation_mask[i, 0, 0, :max_prefix_len +
                                    length] = paddle.ones(
                                        shape=[1, max_prefix_len + length],
                                        dtype=args.dtype)
            else:
                if "chatglm" in args.architecture:
                    attention_mask[i, 0, :length, :length] = 0
                    attention_mask[i, 0, :length - 1, length - 1] = 1
                    tgt_pos[i, 0, 0] = paddle.to_tensor(
                        [length], dtype="int64")
                else:
                    position_ids[i, :length] = paddle.arange(length)
                    attention_mask[i, 0, :length, :length] = paddle.tril(
                        paddle.ones(
                            shape=[length, length], dtype=args.dtype))
                tgt_generation_mask[i, 0, 0, :length] = paddle.ones(
                    shape=[1, length], dtype=args.dtype)
            pre_ids[i:i + 1] = -1
    del inputs["dyinput_flags"]
    #TODO jiangjiajun
    if "chatglm" not in args.architecture:
        inputs["position_ids"] = position_ids
    inputs["tgt_generation_mask"] = tgt_generation_mask
    inputs["tgt_pos"] = tgt_pos
    if args.is_ptuning:
        prefix_caches = []
        for model_id in inputs['model_id']:
            prefix_caches.append(
                get_prefix_caches(args.model_prompt_dir_path, model_id, args.
                                  num_layers, args.num_attention_heads, args.
                                  hidden_size, prompt_num, model_prompt_dict))
        new_prefix_caches = pad_prefix_caches(
            prefix_caches, max_prefix_len, pad_style="left")
        new_prefix_caches = np.stack(new_prefix_caches, axis=2)
        new_prefix_caches = paddle.to_tensor(new_prefix_caches).astype(
            args.dtype)
        for index in range(len(new_prefix_caches)):
            inputs[f"pre_caches_{index}"] = new_prefix_caches[index]
        del inputs['model_id']
    inputs["attention_mask"] = attention_mask
    return inputs


def run(infer_engine):
    """
    infer_engine: InferenceEngine
    """
    # The setting of shared memory
    flag_array = np.zeros([nranks], dtype=np.int32)
    shm_flag_begin = shared_memory.SharedMemory(name="shm_pd_infer_flag_begin")
    flag_begin_array = np.ndarray(
        flag_array.shape, dtype=flag_array.dtype, buffer=shm_flag_begin.buf)
    shm_flag_end = shared_memory.SharedMemory(name="shm_pd_infer_flag_end")
    flag_end_array = np.ndarray(
        flag_array.shape, dtype=flag_array.dtype, buffer=shm_flag_end.buf)

    if rank == 0:
        shm_output_data = shared_memory.SharedMemory(
            name='shm_infer_output_data')

    shm_flag_ready = shared_memory.SharedMemory(name="shm_flag_infer_ready")
    flag_ready_array = np.ndarray(
        flag_array.shape, dtype=flag_array.dtype, buffer=shm_flag_ready.buf)
    flag_ready_array[rank] = 1  # init done

    while 1:
        if serving_pid > 0 and (not is_process_running(serving_pid)):
            print(
                "[IMPORTANT] The serving process {} is not running, will terminate engine now.".
                format(serving_pid))
            break
        if flag_begin_array[rank] != 1:
            continue

        # Load data from shared memory
        shm_input_data = shared_memory.SharedMemory(
            name="shm_infer_input_data")
        shm_input_data_str = bytes(shm_input_data.buf)
        shm_input_data.close()

        inputs = pickle.loads(shm_input_data_str)
        inputs = dy_input_preprocess(inputs)

        finished_ids, step_idx, stop_flags, seq_lens_decoder, tgt_pos = infer_engine.predict(
            inputs)

        if rank == 0:
            output = {
                "finished_ids": finished_ids,
                "step_idx": step_idx,
                "stop_flags": stop_flags,
                "seq_lens_decoder": seq_lens_decoder,
                "tgt_pos": tgt_pos
            }
            output_data_str = pickle.dumps(output)
            output_data_str_size = len(output_data_str)
            output_data_str = (output_data_str_size).to_bytes(
                4, 'big') + output_data_str
            shm_output_data.buf[:len(output_data_str)] = output_data_str
        # Update flag, and start new generation
        flag_end_array[rank] = 1
        flag_begin_array[rank] = 0

        # TODO ========
        break


def main():
    model_dir = os.path.join(args.model_dir, f"rank_{rank}")
    if not os.path.exists(model_dir):
        model_dir = args.model_dir
    if args.is_static_model:
        infer_engine = InferenceEngine(model_dir, nranks)
    else:
        infer_engine = DygraphEngine(model_dir)
    run(infer_engine)


if __name__ == "__main__":
    model_dir = os.path.join(args.model_dir, f"rank_{rank}")
    if not os.path.exists(model_dir):
        model_dir = args.model_dir
    print("Start to initialize engine in rank:{}.".format(rank))
    if args.is_static_model != 0:
        infer_engine = InferenceEngine(model_dir, nranks)
    else:
        infer_engine = DygraphEngine(model_dir, args.architecture, nranks)
    print("Engine intialized in rank:{}.".format(rank))
    run(infer_engine)
