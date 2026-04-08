# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

"""Fastdeploy + ERNIE-4.5-Turbo 的指标评估"""
# adapted from https://github.com/sgl-project/sglang/blob/main/benchmark/gsm8k/bench_other.py
import argparse
import ast
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from tqdm import tqdm

INVALID = -9999999


def call_generate(prompt, **kwargs):
    """
    Generates response based on the input prompt.

    Args:
        prompt (str): The input prompt text.
        **kwargs: Keyword arguments, including server IP address and port number.

    Returns:
        str: The response generated based on the prompt.

    """
    url = f"http://{kwargs['ip']}:{kwargs['port']}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "temperature": 0.6,
        "max_tokens": 2047,
        "top_p": 0.95,
        "do_sample": True,
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    out = response.json()
    return out["choices"][0]["message"]["content"]


def get_one_example(lines, i, include_answer):
    """
    Retrieves a question-answer example from the given list of text lines.

    Args:
        lines (list of dict): A list of question-answer pairs.
        i (int): The index of the question-answer pair to retrieve from lines.
        include_answer (bool): Whether to include the answer in the returned string.

    Returns:
        str: A formatted question-answer string in the format "Question: <question>\nAnswer: <answer>".

    """
    ret = "Question: " + lines[i]["question"] + "\nAnswer:"
    if include_answer:
        ret += " " + lines[i]["answer"]
    return ret


def get_few_shot_examples(lines, k):
    """
    Selects k examples from the given list of text lines and concatenates them into a single string.

    Args:
        lines (list): A list containing text lines.
        k (int): The number of examples to select.

    Returns:
        str: A string composed of k examples, separated by two newline characters.
    """
    ret = ""
    for i in range(k):
        ret += get_one_example(lines, i, True) + "\n\n"
    return ret


def get_answer_value(answer_str):
    """
    Extracts numerical values from an answer string and returns them.

    Args:
        answer_str (str): The string containing the answer.

    Returns:
        The extracted numerical value; returns "INVALID" if extraction fails.
    """
    answer_str = answer_str.replace(",", "")
    numbers = re.findall(r"\d+", answer_str)
    if len(numbers) < 1:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except SyntaxError:
        return INVALID


def read_jsonl(filename: str):
    """
    Reads a JSONL file.

    Args:
        filename (str): Path to the JSONL file.

    Yields:
        dict: A dictionary object corresponding to each line in the JSONL file.
    """
    with open(filename) as fin:
        for line in fin:
            if line.startswith("#"):
                continue
            yield json.loads(line)


def main(args):
    """
    Process inputs and generate answers by calling the model in parallel using a thread pool.

    Args:
        args (argparse.Namespace):
            - num_questions (int): Number of questions to process.
            - num_shots (int): Number of few-shot learning examples.
            - ip (str): IP address of the model service.
            - port (int): Port number of the model service.
            - parallel (int): Number of questions to process in parallel.
            - result_file (str): File path to store the results.

    Returns:
        None

    """
    # Read data
    filename = "test.jsonl"

    lines = list(read_jsonl(filename))

    # Construct prompts
    num_questions = args.num_questions
    num_shots = args.num_shots
    few_shot_examples = get_few_shot_examples(lines, num_shots)

    questions = []
    labels = []
    for i in range(len(lines[:num_questions])):
        questions.append(get_one_example(lines, i, False))
        labels.append(get_answer_value(lines[i]["answer"]))
    assert all(l != INVALID for l in labels)

    states = [None] * len(labels)

    # Use thread pool
    def get_one_answer(i):
        answer = call_generate(
            prompt=few_shot_examples + questions[i],
            # stop=["Question", "Assistant:", "<|separator|>"],
            ip=args.ip,
            port=args.port,
        )
        states[i] = answer

    tic = time.time()
    if args.parallel == 1:
        for i in tqdm(range(len(questions))):
            get_one_answer(i)
    else:
        with ThreadPoolExecutor(args.parallel) as executor:
            list(
                tqdm(
                    executor.map(get_one_answer, list(range(len(questions)))),
                    total=len(questions),
                )
            )

    latency = time.time() - tic
    preds = []
    for i in range(len(states)):
        preds.append(get_answer_value(states[i]))

    # Compute accuracy
    acc = np.mean(np.array(preds) == np.array(labels))
    invalid = np.mean(np.array(preds) == INVALID)

    # Print results
    print(f"Accuracy: {acc:.3f}")
    print(f"Invalid: {invalid:.3f}")
    print(f"Latency: {latency:.3f} s")

    with open(args.result_file, "a") as fout:
        value = {
            "task": "gsm8k",
            "backend": "paddlepaddle",
            "num_gpus": 1,
            "latency": round(latency, 3),
            "accuracy": round(acc, 3),
            "num_requests": args.num_questions,
            "other": {
                "num_questions": args.num_questions,
                "parallel": args.parallel,
            },
        }
        fout.write(json.dumps(value) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=str, default="8188")
    parser.add_argument("--num-shots", type=int, default=10)
    parser.add_argument("--data-path", type=str, default="test.jsonl")
    parser.add_argument("--num-questions", type=int, default=1319)
    parser.add_argument("--result-file", type=str, default="result.jsonl")
    parser.add_argument("--parallel", type=int, default=1)
    args = parser.parse_args()
    main(args)
