import paddlenlp
import numpy as np
import fastdeploy as fd
from paddlenlp.transformers import AutoTokenizer
from paddlenlp.datasets import load_dataset
import os
import time
import distutils.util
import sys


def parse_arguments():
    import argparse
    import ast
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        required=True,
        help="The directory of model and tokenizer.")
    parser.add_argument(
        "--device",
        type=str,
        default='gpu',
        choices=['gpu', 'cpu'],
        help="Type of inference device, support 'cpu' or 'gpu'.")
    parser.add_argument(
        "--backend",
        type=str,
        default='pp',
        choices=['ort', 'pp', 'trt', 'pp-trt'],
        help="The inference runtime backend.")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="The batch size of data.")
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="The max length of sequence.")
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="The interval of logging.")
    parser.add_argument(
        "--cpu_num_threads",
        type=int,
        default=1,
        help="The number of threads when inferring on cpu.")
    parser.add_argument(
        "--use_fp16",
        type=distutils.util.strtobool,
        default=False,
        help="Use FP16 mode")
    return parser.parse_args()


def create_fd_runtime(args):
    option = fd.RuntimeOption()
    model_path = os.path.join(args.model_dir, "infer.pdmodel")
    params_path = os.path.join(args.model_dir, "infer.pdiparams")
    option.set_model_path(model_path, params_path)
    if args.device == 'cpu':
        option.use_cpu()
        option.set_cpu_thread_num(args.cpu_num_threads)
    else:
        option.use_gpu()
    if args.backend == 'pp':
        option.use_paddle_backend()
    elif args.backend == 'ort':
        option.use_ort_backend()
    else:
        option.use_trt_backend()
        if args.backend == 'pp-trt':
            option.enable_paddle_to_trt()
            option.enable_paddle_trt_collect_shape()
        trt_file = os.path.join(args.model_dir, "infer.trt")
        option.set_trt_input_shape(
            'input_ids',
            min_shape=[1, args.max_length],
            opt_shape=[args.batch_size, args.max_length],
            max_shape=[args.batch_size, args.max_length])
        option.set_trt_input_shape(
            'token_type_ids',
            min_shape=[1, args.max_length],
            opt_shape=[args.batch_size, args.max_length],
            max_shape=[args.batch_size, args.max_length])
        if args.use_fp16:
            option.enable_trt_fp16()
            trt_file = trt_file + ".fp16"
        option.set_trt_cache_file(trt_file)
    return fd.Runtime(option)


def convert_examples_to_data(dataset, batch_size):
    texts, text_pairs, labels = [], [], []
    batch_text, batch_text_pair, batch_label = [], [], []

    for i, item in enumerate(dataset):
        batch_text.append(item['sentence1'])
        batch_text_pair.append(item['sentence2'])
        batch_label.append(item['label'])
        if (i + 1) % batch_size == 0:
            texts.append(batch_text)
            text_pairs.append(batch_text_pair)
            labels.append(batch_label)
            batch_text, batch_text_pair, batch_label = [], [], []
    return texts, text_pairs, labels


if __name__ == "__main__":
    args = parse_arguments()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    runtime = create_fd_runtime(args)
    input_ids_name = runtime.get_input_info(0).name
    token_type_ids_name = runtime.get_input_info(1).name

    #test_ds = load_dataset("clue", "afqmc", splits=['test'])
    test_ds = load_dataset("clue", "afqmc", splits=['dev'])
    texts, text_pairs, labels = convert_examples_to_data(test_ds,
                                                         args.batch_size)

    def run_inference(warmup_steps=None):
        time_costs = []
        total_num = 0
        correct_num = 0
        for i, (text, text_pair,
                label) in enumerate(zip(texts, text_pairs, labels)):
            encoded_inputs = tokenizer(
                text=text,
                text_pair=text_pair,
                max_length=args.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='np')
            input_ids = encoded_inputs["input_ids"]
            token_type_ids = encoded_inputs["token_type_ids"]
            start = time.time()
            results = runtime.infer({
                input_ids_name: input_ids.astype('int64'),
                token_type_ids_name: token_type_ids.astype('int64'),
            })
            time_costs += [(time.time() - start) * 1000]

            total_num += len(label)
            logits = results[0]
            argmax_idx = np.argmax(logits, 1)
            correct_num += (label == argmax_idx).sum()
            if warmup_steps is not None and i >= warmup_steps:
                break
            if (i + 1) % args.log_interval == 0:
                print(
                    f"Step {i + 1: 6d}. Mean latency: {np.mean(time_costs):.4f} ms, p50 latency: {np.percentile(time_costs, 50):.4f} ms, "
                    f"p90 latency: {np.percentile(time_costs, 90):.4f} ms, p95 latency: {np.percentile(time_costs, 95):.4f} ms, "
                    f"acc = {correct_num/total_num*100:.2f}.")

        return time_costs, correct_num, total_num

    # Warm up
    run_inference(10)
    time_costs, correct_num, total_num = run_inference()
    print(
        f"Mean latency: {np.mean(time_costs):.4f} ms, p50 latency: {np.percentile(time_costs, 50):.4f} ms, "
        f"p90 latency: {np.percentile(time_costs, 90):.4f} ms, p95 latency: {np.percentile(time_costs, 95):.4f} ms, "
        f"acc = {correct_num/total_num*100:.2f}.")
