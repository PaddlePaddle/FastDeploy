# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
"""Sentinel-guard contract oracle for head-wise SWA (T53 PR1).

Pure-Python shadow oracle that verifies the invariant relied on by the c16
attention kernel's SWA sentinel guard:

    The kernel only ever reads block_table positions whose block_idx is
    >= window_start_block. recycle_request_swa_head_cache writes the -1
    sentinel ONLY at positions < window_start_block. Therefore the kernel
    never observes the sentinel inside the attended window, and the
    `if (block_id < 0) block_id = 0;` clamp is a safety net only.

This holds for BOTH sink configurations:
  - sink_size > 0: sink blocks [0, sink_blocks) are kept; -1 occupies
    [sink_blocks, window_start_block) which the kernel does not read.
  - sink_size == 0: -1 occupies [0, window_start_block); kernel still
    starts at chunk_start >= window_start so does not read it.

No GPU and no fastdeploy import required: this is a contract test on the
allocator/recycle/kernel triple's index arithmetic.
"""
import unittest

BLOCK_SIZE = 64
NUM_LOGICAL_BLOCKS = 16


def _build_block_table_with_recycle(window_start_block, sink_blocks):
    """Mimic the post-recycle state of one head's block_table row.

    Positions in [sink_blocks, window_start_block) are -1 (recycled gap).
    All other positions hold a sequential physical block id.
    """
    table = []
    next_phys = 0
    for idx in range(NUM_LOGICAL_BLOCKS):
        if sink_blocks <= idx < window_start_block:
            table.append(-1)
        else:
            table.append(next_phys)
            next_phys += 1
    return table


def _kernel_read_positions(chunk_start, chunk_end):
    """Return the set of block_table indices the kernel would dereference
    while iterating chunk_start .. chunk_end with BLOCK_SIZE stride."""
    positions = set()
    pos = chunk_start
    while pos < chunk_end:
        positions.add(pos // BLOCK_SIZE)
        pos += BLOCK_SIZE
    return positions


class HeadWiseSWASentinelGuardTest(unittest.TestCase):
    def test_sink_size_positive_no_sentinel_in_attended_window(self):
        sink_blocks = 2
        window_start_block = 5
        table = _build_block_table_with_recycle(window_start_block, sink_blocks)

        # kernel attends sink + window; window_start = window_start_block * BLOCK_SIZE
        chunk_start = window_start_block * BLOCK_SIZE
        chunk_end = NUM_LOGICAL_BLOCKS * BLOCK_SIZE
        for idx in _kernel_read_positions(chunk_start, chunk_end):
            self.assertGreaterEqual(
                table[idx],
                0,
                f"sentinel observed at attended block_idx={idx} "
                f"(window_start_block={window_start_block}, sink_blocks={sink_blocks})",
            )
        # And the sink window itself
        for idx in range(sink_blocks):
            self.assertGreaterEqual(table[idx], 0, f"sink block {idx} should not be -1")

    def test_sink_size_zero_no_sentinel_at_chunk_start(self):
        sink_blocks = 0
        window_start_block = 4
        table = _build_block_table_with_recycle(window_start_block, sink_blocks)

        chunk_start = window_start_block * BLOCK_SIZE
        chunk_end = NUM_LOGICAL_BLOCKS * BLOCK_SIZE
        for idx in _kernel_read_positions(chunk_start, chunk_end):
            self.assertGreaterEqual(
                table[idx],
                0,
                f"sentinel observed at attended block_idx={idx} "
                f"(window_start_block={window_start_block}, sink_size==0)",
            )

    def test_recycled_gap_does_not_overlap_kernel_reads(self):
        for sink_blocks, window_start_block in [(0, 3), (1, 4), (2, 6), (3, 7)]:
            table = _build_block_table_with_recycle(window_start_block, sink_blocks)
            recycled_positions = {i for i, v in enumerate(table) if v == -1}
            # kernel never starts before window_start
            chunk_start = window_start_block * BLOCK_SIZE
            chunk_end = NUM_LOGICAL_BLOCKS * BLOCK_SIZE
            kernel_positions = _kernel_read_positions(chunk_start, chunk_end)
            overlap = recycled_positions & kernel_positions
            self.assertEqual(
                overlap,
                set(),
                f"recycled positions {recycled_positions} overlap kernel reads "
                f"{kernel_positions} for sink_blocks={sink_blocks}, "
                f"window_start_block={window_start_block}",
            )


if __name__ == "__main__":
    unittest.main()
