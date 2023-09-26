import queue
import json
import sys
from functools import partial

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import *

class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()


def callback(user_data, result, error):
    if error:
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)


def get_completion(text, model_name, grpc_url):
    model_name = model_name
    in_value = {
    "text": text,
    "topp": 0.0,
    "temperature": 1.0,
    "max_dec_len": 1024,
    "min_dec_len": 2,
    "penalty_score": 1.0,
    "frequency_score": 0.99,
    "eos_token_id": 2,
    "model_test": "test",
    "presence_score": 0.0
    }
    inputs = [grpcclient.InferInput("IN", [1], np_to_triton_dtype(np.object_))]
    outputs = [grpcclient.InferRequestedOutput("OUT")]
    user_data = UserData()
    completion = ""
    with grpcclient.InferenceServerClient(url=grpc_url, verbose=False) as triton_client:
        triton_client.start_stream(callback=partial(callback, user_data))
        in_data = np.array([json.dumps(in_value)], dtype=np.object_)
        inputs[0].set_data_from_numpy(in_data)
        triton_client.async_stream_infer(model_name=model_name, inputs=inputs, request_id="0", outputs=outputs)
        while True:
            data_item = user_data._completed_requests.get(timeout=300)
            if type(data_item) == InferenceServerException:
                print('Exception:', 'status', data_item.status(), 'msg', data_item.message())
            else:
                results = data_item.as_numpy("OUT")[0]
                data = json.loads(results)

                completion += data["result"]
                if data.get("is_end", False):
                    break
        return completion

grpc_url = "0.0.0.0:8135"
model_name = "llama-ptuning"
result = get_completion("Hello, how are you", model_name, grpc_url)
