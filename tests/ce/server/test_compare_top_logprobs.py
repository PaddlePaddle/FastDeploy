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

from core import TEMPLATE, URL, build_request_payload, send_request


def get_response(data):
    """
    Get the response from the API using the given data.
    Args:
        data (dict): The input data to be sent to the API.

        Returns:
            dict: The JSON response from the API.
    """
    payload = build_request_payload(TEMPLATE, data)
    resp = send_request(URL, payload)
    return resp.json()


def assert_top_logprobs_prefix_match(small_top, large_top, token_index):
    """
    Assert that all entries in small_top are a prefix of large_top,
    comparing token, logprob, and bytes values.
    """
    for j, (s, l) in enumerate(zip(small_top, large_top)):
        for field in ["token", "logprob", "bytes"]:
            s_val = s[field]
            l_val = l[field]
            assert s_val == l_val, "{} mismatch at token {} pos {}: {} != {}".format(
                field.capitalize(), token_index + 1, j + 1, repr(s_val), repr(l_val)
            )


def compare_top_logprobs(base_data, top_logprobs_values=[5, 10]):
    """
    Compare the top logprobs of two different values and check if they match.

    Args:
        base_data (dict): The base data used for generating the responses.
        top_logprobs_values (list): A list of integers representing the top logprobs values to compare.

    Raises:
        AssertionError: If any mismatches are found between the top logprobs values.
    """
    responses = {}

    for val in top_logprobs_values:
        data = base_data.copy()
        data.update(
            {
                "top_logprobs": val,
                "logprobs": True,
                "stream": False,
                "temperature": 0,
                "top_p": 0,
                "max_tokens": 10,
            }
        )

        response = get_response(data)
        responses[val] = response

    # Assertion for prefix consistency
    if len(top_logprobs_values) >= 2:
        small = top_logprobs_values[0]
        large = top_logprobs_values[1]

        small_contents = responses[small]["choices"][0]["logprobs"]["content"]
        large_contents = responses[large]["choices"][0]["logprobs"]["content"]
        min_len = min(len(small_contents), len(large_contents))

        for i in range(min_len):
            small_top = small_contents[i]["top_logprobs"]
            large_top = large_contents[i]["top_logprobs"]
            assert_top_logprobs_prefix_match(small_top, large_top, i)


def test_compare_top_logprobs():
    """
    Test the compare_top_logprobs function with a sample input data.
    Returns:
        None
        AssertionError: If there is a mismatch between the top logprobs values.

    """
    data = {
        "model": "default",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "牛顿的三大运动定律是什么？"},
        ],
    }

    compare_top_logprobs(data, top_logprobs_values=[5, 10])


if __name__ == "__main__":
    """
    Test the compare_top_logprobs function with a sample input data.
    Returns:
        None
        AssertionError: If there is a mismatch between the top logprobs values.

    """
    test_compare_top_logprobs()
