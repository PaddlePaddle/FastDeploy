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

"""Tests for process_stop_token_ids in fastdeploy.input.utils.common."""

from fastdeploy.input.utils.common import process_stop_token_ids


def _mock_update_stop_seq_fn(stop_sequences):
    """Mock update_stop_seq that simulates tokenization and padding.

    Simulates: "```" -> [101], "end" -> [201, 202]
    Returns padded sequences and actual lengths.
    """
    token_map = {
        "```": [101],
        "end": [201, 202],
        "\n\n": [301, 302, 303],
        "stop": [401],
    }
    seqs = [token_map.get(s, [999]) for s in stop_sequences]
    actual_lens = [len(s) for s in seqs]
    # Simulate pad_batch_data: pad to max length with -1
    max_len = max(len(s) for s in seqs) if seqs else 0
    padded = [s + [-1] * (max_len - len(s)) for s in seqs]
    return padded, actual_lens


def test_stop_token_ids_list_int():
    """stop_token_ids as List[int] should produce length-1 sequences."""
    request = {"stop_token_ids": [100, 200, 300]}
    process_stop_token_ids(request, _mock_update_stop_seq_fn)

    assert request["stop_token_ids"] == [[100], [200], [300]]
    assert request["stop_seqs_len"] == [1, 1, 1]


def test_stop_token_ids_list_list_int():
    """stop_token_ids as List[List[int]] should preserve actual lengths."""
    request = {"stop_token_ids": [[10, 20], [30]]}
    process_stop_token_ids(request, _mock_update_stop_seq_fn)

    assert request["stop_token_ids"] == [[10, 20], [30]]
    assert request["stop_seqs_len"] == [2, 1]


def test_stop_strings_uses_actual_lengths():
    """stop strings with different tokenized lengths should use actual lengths, not padded."""
    request = {"stop": ["```", "end"]}
    process_stop_token_ids(request, _mock_update_stop_seq_fn)

    # "```" -> [101, -1] (padded), actual len 1
    # "end" -> [201, 202], actual len 2
    assert request["stop_token_ids"] == [[101, -1], [201, 202]]
    assert request["stop_seqs_len"] == [1, 2]


def test_mixed_stop_token_ids_and_stop_strings():
    """Both stop_token_ids and stop strings should have correct lengths."""
    request = {
        "stop_token_ids": [100],
        "stop": ["```", "\n\n"],
    }
    process_stop_token_ids(request, _mock_update_stop_seq_fn)

    # stop_token_ids: [100] -> [[100]], len [1]
    # "```" -> [101, -1, -1] (padded to 3), actual len 1
    # "\n\n" -> [301, 302, 303], actual len 3
    assert request["stop_token_ids"] == [[100], [101, -1, -1], [301, 302, 303]]
    assert request["stop_seqs_len"] == [1, 1, 3]


def test_empty_request():
    """No stop tokens or strings should leave request unchanged."""
    request = {}
    process_stop_token_ids(request, _mock_update_stop_seq_fn)

    assert "stop_token_ids" not in request
    assert "stop_seqs_len" not in request


def test_stop_token_ids_none():
    """stop_token_ids=None should be treated as absent."""
    request = {"stop_token_ids": None, "stop": ["stop"]}
    process_stop_token_ids(request, _mock_update_stop_seq_fn)

    assert request["stop_token_ids"] == [[401]]
    assert request["stop_seqs_len"] == [1]


def test_stop_token_ids_empty_list():
    """stop_token_ids=[] should be treated as absent."""
    request = {"stop_token_ids": []}
    process_stop_token_ids(request, _mock_update_stop_seq_fn)

    assert "stop_seqs_len" not in request


if __name__ == "__main__":
    test_stop_token_ids_list_int()
    test_stop_token_ids_list_list_int()
    test_stop_strings_uses_actual_lengths()
    test_mixed_stop_token_ids_and_stop_strings()
    test_empty_request()
    test_stop_token_ids_none()
    test_stop_token_ids_empty_list()
    print("All tests passed.")
