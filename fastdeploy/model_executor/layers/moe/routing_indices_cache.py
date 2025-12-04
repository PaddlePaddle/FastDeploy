"""
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
"""


def _record_routing_native(
    routing_replay_table,
    seq_lens_decoder,
    seq_lens_this_time,
    cu_seqlens_q,
    layer_id,
    topk_ids,
):
    """record routing native"""
    # for i in range(hidden_states.shape[0]):
    # batch_id, layerid, tokenid,
    print("topk_ids", topk_ids)
    print("before update", routing_replay_table)
    history_length = seq_lens_decoder
    print("history_length", history_length)
    for batch_id in range(seq_lens_this_time.shape[0]):
        print("batch_id", batch_id)
        for token_id_one_dim in range(cu_seqlens_q[batch_id + 1]):
            token_id_query = history_length[batch_id] + (token_id_one_dim % cu_seqlens_q[batch_id + 1])
            print("token_id_one_dim", token_id_one_dim)
            print("token_id_query", token_id_query)
            routing_replay_table[batch_id][layer_id][token_id_query] = topk_ids[token_id_one_dim]
    print("after update", routing_replay_table)
