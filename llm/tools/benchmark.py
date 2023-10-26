import queue
import numpy as np
import time
import fire
import os
import json
import sys
from functools import partial
import multiprocessing as mp

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import *
from paddlenlp.transformers import AutoTokenizer

from sentencepiece import SentencePieceProcessor

class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()

def callback(user_data, result, error):
    if error:
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)


import random
def read_dataset(tokenizer_path, dataset_path, samples=-1, truncated=1024):
    with open(dataset_path) as f:
        dataset = json.load(f)
        dataset = [data for data in dataset if len(data['conversations']) >= 2]
        dataset = [(data['conversations'][0]['value'],
                    data['conversations'][1]['value']) for data in dataset]
        prompts = [prompt for prompt, _ in dataset]
        completions = [completion for _, completion in dataset]

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    result = list()
    for i, prompt in enumerate(prompts):
        token_ids = tokenizer(prompt, return_tensors="np", padding=False, return_attention_mas=False, return_token_type_ids=False)
        prompt_ids = token_ids["input_ids"][0].tolist()[:truncated]

        completion = completions[i]
        token_ids = tokenizer(completion, return_tensors="np", padding=False, return_attention_mas=False, return_token_type_ids=False)
        completion_ids = token_ids["input_ids"][0].tolist()[:truncated]
        result.append({"src": prompt, "tgt": completions[i], "src_ids": prompt_ids, "tgt_ids": completion_ids})

    if samples > 0 and samples < len(result):
        import random
        random.shuffle(result)
        result = result[:samples]

    if samples > len(result):
        times = (samples // len(result)) + 1
        result = result * times
        import random
        random.shuffle(result)
        result = result[:samples]
    return result

def infer(grpc_addr, model_name, i, request_queue, result_queue, finished_flags):
    counter = 0
    while not request_queue.empty():
        try:
            request = request_queue.get(timeout=0.1)
        except:
            continue
        counter += 1
        in_value = {
           "req_id": i * 10000 + counter,
           "text": request["src"],
           "topp": 0.0,
           "temperature": 1.0,
           "max_dec_len": 1024,
           "min_dec_len": 2,
           "penalty_score": 1.0,
           "eos_token_id": 2,
           "frequency_score": 0.0,
           "presence_score": 0.0
        }

        send_request_time = time.perf_counter()
        inputs = [grpcclient.InferInput("IN", [1], np_to_triton_dtype(np.object_))]
        outputs = [grpcclient.InferRequestedOutput("OUT")]
        user_data = UserData()
        completion = list()
        completion_num = 0

        is_error_request = False
        with grpcclient.InferenceServerClient(url=grpc_addr, verbose=False) as triton_client:
            triton_client.start_stream(callback=partial(callback, user_data))
            in_data = np.array([json.dumps(in_value)], dtype=np.object_)
            inputs[0].set_data_from_numpy(in_data)
            triton_client.async_stream_infer(model_name=model_name, inputs=inputs, request_id="0", outputs=outputs)
            while True:
                data_item = user_data._completed_requests.get(timeout=300)
                if type(data_item) == InferenceServerException:
                    print('Exception:', 'status', data_item.status(), 'msg', data_item.message())
                    is_error_request = True
                    break
                else:
                    results = data_item.as_numpy("OUT")[0]
                    data = json.loads(results)
                    completion.append(data["token_ids"])
                    completion_num += 1
                    if completion_num == 1:
                        first_token_time = time.perf_counter()
                    if data.get("is_end", 0) == 1:
                        break
        total_token_time = time.perf_counter()
        if not is_error_request:
            result_queue.put([len(request["src_ids"]), completion_num, first_token_time-send_request_time, total_token_time-send_request_time])
    finished_flags.put(1)


def main(grpc_addr, model_name, tokenizer_path, dataset_path, samples=-1, concurrency=1, test_round=1):
    dataset = read_dataset(tokenizer_path, dataset_path, samples)
    request_queue = mp.Queue()
    finished_flags = mp.Queue()
    result_queue = mp.Queue()
    for d in dataset:
        request_queue.put(d)

    start = time.perf_counter()
    procs = list()
    for i in range(concurrency):
       procs.append(mp.Process(target=infer, args=(grpc_addr, model_name, i, request_queue, result_queue, finished_flags)))
       procs[-1].start()

    while finished_flags.qsize() < concurrency:
        time.sleep(0.1)
        continue
    end = time.perf_counter()

    results = list()
    while not result_queue.empty():
        results.append(result_queue.get(timeout=0.1))

    qps = len(results) / (end - start)

    results = np.array(results).astype("float32")
    mean_stat = results.mean(axis=0).tolist()
    min_stat = results.min(axis=0).tolist()
    max_stat = results.max(axis=0).tolist()
    print("Send {} requests and get {} responses.".format(len(dataset), len(results)))
    print("Input Length(max, min, mean): ", max_stat[0], min_stat[0], mean_stat[0])
    print("Output Length(max, min, mean): ", max_stat[1], min_stat[1], mean_stat[1])
    print("First Token Latency(max, min, mean): ", max_stat[2], min_stat[2], mean_stat[2])
    print("Total Token Latency(max, min, mean): ",  max_stat[3], min_stat[3], mean_stat[3])
    print("QPS: ", qps)

if __name__ == '__main__':
    fire.Fire(main)
