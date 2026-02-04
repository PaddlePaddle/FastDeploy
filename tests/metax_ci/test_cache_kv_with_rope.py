import random
import unittest
from dataclasses import dataclass, field  # 1. 导入 dataclass 和 field

import paddle
import triton

from fastdeploy.model_executor.ops.gpu import cache_kv_with_rope

paddle.set_default_dtype("bfloat16")
paddle.seed(0)
random.seed(0)


def assert_value(x, y):
    return paddle.allclose(x.to("float32"), y.to("float32"), 1e-2, 1e-2).item()


@dataclass
class TestCacheKVWithRopeParams:
    max_model_len: int = 32768
    min_sampe_model_len: int = 1900
    max_sampe_model_len: int = 2200
    q_head_num: int = 20
    kv_head_num: int = 4
    head_dim: int = 128
    max_num_seqs: int = 8
    batch_size: int = 4
    block_size: int = 64
    enable_mm: bool = True
    apply_rope_method: str = "pd"

    block_num: int = field(init=False)
    max_block_num: int = field(init=False)

    def __post_init__(self):
        self.block_num = (self.max_model_len + self.block_size - 1) // self.block_size
        self.max_block_num = (self.block_num + 100) * self.max_num_seqs


class TestCacheKVWithRope(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Class-level setup that runs once before all tests."""
        cls.params = TestCacheKVWithRopeParams()
        cls._init_metadata()

    @classmethod
    def _init_metadata(cls):
        cls.all_head_num = cls.params.q_head_num + 2 * cls.params.kv_head_num
        cls.block_tables = paddle.full([cls.params.max_num_seqs, cls.params.block_num], fill_value=-1, dtype="int32")

        cls.batch_ids_prefill = sorted(random.sample(range(cls.params.max_num_seqs), cls.params.batch_size))
        cls.batch_ids_prefill = paddle.to_tensor(cls.batch_ids_prefill, dtype="int32")
        cls.token_num = 0
        cls.cu_seqlens_q = [0]
        cls.seq_lens_this_time = [0] * cls.params.max_num_seqs
        global_block_start = 0
        for i in cls.batch_ids_prefill.tolist():
            rand_token_num = random.randint(cls.params.min_sampe_model_len, cls.params.max_sampe_model_len)
            cls.token_num += rand_token_num
            cls.cu_seqlens_q.append(cls.token_num)
            cls.seq_lens_this_time[i] = rand_token_num
            used_block_num = (rand_token_num + cls.params.block_size - 1) // cls.params.block_size
            cls.block_tables[i][:used_block_num] = paddle.arange(
                global_block_start, global_block_start + used_block_num, dtype="int32"
            )
            global_block_start += used_block_num
        cls.cu_seqlens_q = paddle.to_tensor(cls.cu_seqlens_q, dtype="int32")
        cls.seq_lens_this_time = paddle.to_tensor(cls.seq_lens_this_time, dtype="int32")
        cls.seq_lens = paddle.zeros(shape=[cls.params.batch_size], dtype="int32")

        cls.qkv = paddle.randn([cls.token_num, cls.all_head_num * cls.params.head_dim])
        cls._init_rotary_embs()

    @classmethod
    def _init_rotary_embs(cls):
        if not cls.params.enable_mm:
            cls.rotary_embs = paddle.randn(
                [2, 1, cls.params.max_model_len, 1, cls.params.head_dim // 2], dtype="float32"
            )
        else:
            cls.rotary_embs = paddle.randn(
                [cls.params.max_num_seqs, 2, 1, cls.params.max_model_len, 1, cls.params.head_dim // 2], dtype="float32"
            )

        rot_cos, rot_sin = cls._update_rotary_embs_prefill()
        # non-neox style
        cls.rotary_cos_non_neox = paddle.repeat_interleave(rot_cos, repeats=2, axis=-1)
        cls.rotary_sin_non_neox = paddle.repeat_interleave(rot_sin, repeats=2, axis=-1)
        # neox style
        tile_time = [1] * rot_cos.dim()
        tile_time[-1] = 2
        cls.rotary_cos_neox = paddle.tile(rot_cos, tile_time)
        cls.rotary_sin_neox = paddle.tile(rot_sin, tile_time)
        # print(self.rotary_cos_neox)
        cls.rotary_embs_neox = paddle.concat([cls.rotary_embs, cls.rotary_embs], axis=-1)

    @classmethod
    def _update_rotary_embs_prefill(cls):
        batch_ids = cls.batch_ids_prefill
        seq_lens_this_time = cls.seq_lens_this_time[cls.batch_ids_prefill]

        # mapping token idx to batch idx
        cls.batch_ids_q = paddle.repeat_interleave(
            paddle.arange(0, batch_ids.shape[0], dtype="int32"), repeats=seq_lens_this_time, axis=0
        )

        all_indices = []
        for i in range(len(batch_ids)):
            start_pos = 0
            seq_len_i = seq_lens_this_time[i]
            if seq_len_i > 0:
                indices_i = paddle.arange(start_pos, start_pos + seq_len_i, dtype="int32")
                all_indices.append(indices_i)
        if not all_indices:
            return

        all_indices = paddle.concat(all_indices)  # [token_num]
        if cls.params.enable_mm:
            gather_nd_indices = paddle.stack(
                [  # [token_num, 2]
                    paddle.repeat_interleave(batch_ids, repeats=seq_lens_this_time, axis=0),
                    all_indices,
                ],
                axis=1,
            )
            # print(f"{gather_nd_indices=}")
            gathered_embs = paddle.gather_nd(
                cls.rotary_embs.squeeze([2]).transpose(
                    [0, 2, 1, 3, 4]
                ),  # [B, 2, 1, S, 1, D // 2] -> [B, S, 2, 1, D // 2]
                gather_nd_indices,
            )  # [token_num, 2, 1, D // 2]
            rot_cos = gathered_embs[:, 0, :, :]  # [token_num, 1, D // 2]
            rot_sin = gathered_embs[:, 1, :, :]
        else:
            gathered_embs = paddle.gather(
                cls.rotary_embs.squeeze([1]), all_indices, axis=1  # [2, 1, S, 1, D // 2] -> [2, S, 1, D // 2]
            )  # [2, token_num, 1, D // 2]
            rot_cos = gathered_embs[0, :, :, :]  # [token_num, 1, D // 2]
            rot_sin = gathered_embs[1, :, :, :]

        return rot_cos, rot_sin

    def _update_kv_cache(self, k, v, caches_k: paddle.Tensor, caches_v: paddle.Tensor):
        tensor_start = 0
        specific_batch_ids = self.batch_ids_prefill.tolist()
        for batch_idx in range(self.block_tables.shape[0]):
            if specific_batch_ids is not None and batch_idx not in specific_batch_ids:
                continue
            seq_len = self.seq_lens_this_time[batch_idx]
            if seq_len == 0:
                continue
            tensor_end = tensor_start + seq_len
            slice_trans_k = k[tensor_start:tensor_end, :, :]
            slice_trans_v = v[tensor_start:tensor_end, :, :]

            cur_block_tables = self.block_tables[batch_idx]
            cur_used_block_tables = cur_block_tables[cur_block_tables != -1]

            # encoder prefil
            if seq_len > 1:
                cache_start = 0
                cur_used_num_blocks = cur_used_block_tables.shape[0]

                for i, block_id in enumerate(cur_used_block_tables):

                    # last block: seq_len - cache_start <= block_size
                    if i == cur_used_num_blocks - 1:
                        cache_end = seq_len - cache_start
                        assert cache_end <= self.params.block_size

                        caches_k[block_id, 0:cache_end, :, :] = slice_trans_k[cache_start:seq_len, :, :]
                        caches_v[block_id, 0:cache_end, :, :] = slice_trans_v[cache_start:seq_len, :, :]
                        # if layer_id == self.num_layers - 1:
                        #     self.record_block_table_metadata[batch_idx] = {
                        #         "block_id": block_id.item(),
                        #         "cache_end": cache_end,
                        #     }
                    # non last block: seq_lens_this_time > block_size
                    else:
                        assert seq_len > self.params.block_size
                        cache_end = cache_start + self.params.block_size
                        caches_k[block_id] = slice_trans_k[cache_start:cache_end, :, :]
                        caches_v[block_id] = slice_trans_v[cache_start:cache_end, :, :]
                        cache_start += self.params.block_size
            tensor_start = tensor_end

        return caches_k, caches_v

    # ====================== Non-Neox ======================

    def test_cache_kv_with_rope_paddle(self):
        if self.params.apply_rope_method == "pd":

            def rotate_half(x):
                x1 = x[..., 0::2]
                x2 = x[..., 1::2]
                return paddle.stack([-x2, x1], axis=-1).reshape(x.shape)

            def apply_rotary_pos_emb_vision(x, cos, sin):
                orig_dtype = x.dtype
                x = x.astype("float32")
                x_embed = (x * cos) + (rotate_half(x) * sin)
                return x_embed.astype(orig_dtype)

            qkv = self.qkv.view([-1, self.all_head_num, self.params.head_dim])
            q, k, v = paddle.split(
                qkv,
                num_or_sections=[self.params.q_head_num, self.params.kv_head_num, self.params.kv_head_num],
                axis=-2,
            )

            q = apply_rotary_pos_emb_vision(q, self.rotary_cos_non_neox, self.rotary_sin_non_neox)
            k = apply_rotary_pos_emb_vision(k, self.rotary_cos_non_neox, self.rotary_sin_non_neox)
        else:
            batch_ids_q = paddle.repeat_interleave(
                paddle.arange(0, self.batch_ids_prefill.shape[0], dtype="int32"),
                repeats=self.seq_lens_this_time[self.batch_ids_prefill],
                axis=0,
            )
            q, k, v = cache_kv_with_rope(
                self.qkv,
                self.rotary_embs,
                batch_ids_q,
                self.batch_ids_prefill,
                self.cu_seqlens_q,
                self.seq_lens,
                None,
                None,
                None,
                self.params.q_head_num,
                self.params.kv_head_num,
                self.params.head_dim,
                -1,
                3,
                False,
            )

        caches_k = paddle.zeros(
            [self.params.max_block_num, self.params.block_size, self.params.kv_head_num, self.params.head_dim]
        )
        caches_v = paddle.zeros_like(caches_k)

        caches_k, caches_v = self._update_kv_cache(k, v, caches_k, caches_v)

        return q, k, v, caches_k, caches_v

    def test_cache_kv_with_rope_cuda(self):
        caches_k = paddle.zeros(
            [self.params.max_block_num, self.params.block_size, self.params.kv_head_num, self.params.head_dim]
        )
        caches_v = paddle.zeros_like(caches_k)

        batch_ids_q = paddle.repeat_interleave(
            paddle.arange(0, self.batch_ids_prefill.shape[0], dtype="int32"),
            repeats=self.seq_lens_this_time[self.batch_ids_prefill],
            axis=0,
        )

        # print(f"{self.rotary_embs=}")
        # print(f"{batch_ids_q=}")
        # print(f"{self.cu_seqlens_q=}")
        # print(f"{self.batch_ids_prefill=}")
        # print(f"{self.block_tables[self.batch_ids_prefill]=}")

        # print(batch_ids_q.tolist())
        q, k, v = cache_kv_with_rope(
            self.qkv,
            self.rotary_embs,
            batch_ids_q,
            self.batch_ids_prefill,
            self.cu_seqlens_q,
            self.seq_lens,
            caches_k,
            caches_v,
            self.block_tables,
            self.params.q_head_num,
            self.params.kv_head_num,
            self.params.head_dim,
            self.params.block_size,
            3,
            False,
        )

        # print(f'{q[:, 0, :]=}')

        return q, k, v, caches_k, caches_v

    def bench_precision(self):
        value_names = ["q", "k", "v", "caches_k", "caches_v"]  # 'q', 'k', 'v', 'caches_k', 'caches_v'

        out_paddle = self.test_cache_kv_with_rope_paddle()
        out_cuda = self.test_cache_kv_with_rope_cuda()

        for i in range(len(value_names)):
            is_equal = assert_value(out_paddle[i], out_cuda[i])
            print(f"{value_names[i]}: {is_equal}")
            if not is_equal:
                print(out_paddle[i])
                print(out_cuda[i])
                break

    def bench_time_cost(self):
        for bs in range(1, 9):
            self.params.batch_size = bs
            self._init_metadata()
            paddle_cost = triton.testing.do_bench(self.test_cache_kv_with_rope_paddle)
            cuda_cost = triton.testing.do_bench(self.test_cache_kv_with_rope_cuda)

            print(f"{bs=} {self.token_num=} {paddle_cost=} {cuda_cost=} {(cuda_cost / paddle_cost)=}")

    # ======================== Neox ========================

    def test_cache_kv_with_neox_rope_paddle(self):
        if self.params.apply_rope_method == "pd":

            def rotate_half(x):
                Dh = x.shape[-1]
                x1 = x[..., : Dh // 2]
                x2 = x[..., Dh // 2 :]
                return paddle.concat([-x2, x1], axis=-1)

            def apply_rotary_pos_emb_vision(x, cos, sin):
                orig_dtype = x.dtype
                x = x.astype("float32")
                x_embed = (x * cos) + (rotate_half(x) * sin)
                return x_embed.astype(orig_dtype)

            qkv = self.qkv.view([-1, self.all_head_num, self.params.head_dim])
            q, k, v = paddle.split(
                qkv,
                num_or_sections=[self.params.q_head_num, self.params.kv_head_num, self.params.kv_head_num],
                axis=-2,
            )

            q = apply_rotary_pos_emb_vision(q, self.rotary_cos_neox, self.rotary_sin_neox)
            k = apply_rotary_pos_emb_vision(k, self.rotary_cos_neox, self.rotary_sin_neox)
        else:
            batch_ids_q = paddle.repeat_interleave(
                paddle.arange(0, self.batch_ids_prefill.shape[0], dtype="int32"),
                repeats=self.seq_lens_this_time[self.batch_ids_prefill],
                axis=0,
            )
            q, k, v = cache_kv_with_rope(
                self.qkv,
                self.rotary_embs_neox,
                batch_ids_q,
                self.batch_ids_prefill,
                self.cu_seqlens_q,
                self.seq_lens,
                None,
                None,
                None,
                self.params.q_head_num,
                self.params.kv_head_num,
                self.params.head_dim,
                -1,
                3,
                True,
            )

        caches_k = paddle.zeros(
            [self.params.max_block_num, self.params.block_size, self.params.kv_head_num, self.params.head_dim]
        )
        caches_v = paddle.zeros_like(caches_k)

        caches_k, caches_v = self._update_kv_cache(k, v, caches_k, caches_v)

        return q, k, v, caches_k, caches_v

    def test_cache_kv_with_neox_rope_cuda(self):
        caches_k = paddle.zeros(
            [self.params.max_block_num, self.params.block_size, self.params.kv_head_num, self.params.head_dim]
        )
        caches_v = paddle.zeros_like(caches_k)

        batch_ids_q = paddle.repeat_interleave(
            paddle.arange(0, self.batch_ids_prefill.shape[0], dtype="int32"),
            repeats=self.seq_lens_this_time[self.batch_ids_prefill],
            axis=0,
        )
        # print(batch_ids_q.tolist())
        q, k, v = cache_kv_with_rope(
            self.qkv,
            self.rotary_embs_neox,
            batch_ids_q,
            self.batch_ids_prefill,
            self.cu_seqlens_q,
            self.seq_lens,
            caches_k,
            caches_v,
            self.block_tables,
            self.params.q_head_num,
            self.params.kv_head_num,
            self.params.head_dim,
            self.params.block_size,
            3,
            True,
        )

        return q, k, v, caches_k, caches_v

    def bench_precision_neox(self):
        value_names = ["q", "k", "v", "caches_k", "caches_v"]  # 'q', 'k', 'v', 'caches_k', 'caches_v'

        out_paddle = self.test_cache_kv_with_neox_rope_paddle()
        out_cuda = self.test_cache_kv_with_neox_rope_cuda()

        for i in range(len(value_names)):
            is_equal = assert_value(out_paddle[i], out_cuda[i])
            print(f"{value_names[i]}: {is_equal}")
            if not is_equal:
                print(out_paddle[i])
                print(out_cuda[i])
                break

    def bench_time_cost_neox(self):
        for bs in range(1, 9):
            self.params.batch_size = bs
            self._init_metadata()
            paddle_cost = triton.testing.do_bench(self.test_cache_kv_with_neox_rope_paddle)
            cuda_cost = triton.testing.do_bench(self.test_cache_kv_with_neox_rope_cuda)

            print(f"{bs=} {self.token_num=} {paddle_cost=} {cuda_cost=} {(cuda_cost / paddle_cost)=}")


if __name__ == "__main__":
    unittest.main()

    # # cases.test_cache_kv_with_rope_paddle()
    # # cases.test_cache_kv_with_rope_cuda()
    # # cases.test_cache_kv_with_rope_neox_paddle()
    # # cases.test_cache_kv_with_rope_neox_cuda()

    # print("=" * 40 + " Non-Neox " + "=" * 40)
    # cases.bench_precision()
    # print("-" * 90)
    # cases.bench_time_cost()
    # print("=" * 90)
    # print()

    # print("=" * 42 + " Neox " + "=" * 42)
    # cases.bench_precision_neox()
    # print("-" * 90)
    # cases.bench_time_cost_neox()
    # print("=" * 90)
    # print()

    # # import paddle.profiler as profiler
    # # prof = profiler.Profiler(
    # #     targets=[profiler.ProfilerTarget.CPU, profiler.ProfilerTarget.CUSTOM_DEVICE],
    # #     scheduler = (10, 40))
    # # prof.start()
    # # for iter in range(50):
    # #     cases.test_cache_kv_with_neox_rope_cuda()
    # #     prof.step()
    # # prof.stop()
    # # prof.summary(sorted_by=profiler.SortedKeys.CPUTotal, op_detail=True, thread_sep=False, time_unit='ms')
