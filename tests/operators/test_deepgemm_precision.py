"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

import unittest

import paddle

paddle.enable_compat(scope={"deep_gemm"})

paddle.set_default_dtype("bfloat16")

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import cpasync, tcgen05


class DenseGemmKernel:
    def __init__(self):
        self.num_warps = 4
        self.num_tmem_alloc_cols = 512
        self.threads_per_cta = 128
        self.a_dtype = cutlass.BFloat16
        self.b_dtype = cutlass.BFloat16
        self.acc_dtype = cutlass.Float32

        self.num_acc_stage = 1
        self.use_2cta_instrs = False
        self.cluster_shape_mnk = (2, 1, 1) if self.use_2cta_instrs else (1, 1, 1)
        self.cluster_shape_mn = (2, 1) if self.use_2cta_instrs else (1, 1)
        self.cta_group = tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE

        self.mma_tiler = (128, 128, 64)

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
    ):
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            cutlass.BFloat16,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
        )
        self.atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (tiled_mma.thr_id.shape,),
        )
        # ((2),1,1,1):((1),0,0,0)

        a_smem_layout_staged = sm100_utils.make_smem_layout_a(tiled_mma, self.mma_tiler, cutlass.BFloat16, 1)
        b_smem_layout_staged = sm100_utils.make_smem_layout_b(tiled_mma, self.mma_tiler, cutlass.BFloat16, 1)

        a_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=(cutlass.TFloat32 if a.element_type is cutlass.Float32 else None),
        )

        # Setup TMA load for B
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=(cutlass.TFloat32 if b.element_type is cutlass.Float32 else None),
        )

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        self.num_tma_load_bytes = (a_copy_size + b_copy_size) * self.atom_thr_size

        self.kernel(
            tiled_mma,
            a,
            b,
            c,
            a_smem_layout_staged,
            b_smem_layout_staged,
            tma_atom_a,
            tma_atom_b,
            self.cluster_layout_vmnk,
        ).launch(
            grid=self.cluster_shape_mnk,
            block=[128, 1, 1],
            cluster=self.cluster_shape_mnk,
        )

    # GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tiled_mma,
        a,
        b,
        c,
        a_smem_layout_staged,
        b_smem_layout_staged,
        tma_atom_a,
        tma_atom_b,
        cluster_layout_vmnk: cute.Layout,
    ):
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        tidx = cute.arch.thread_idx()[0]

        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)

        @cute.struct
        class SharedStorage:
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        sA = smem.allocate_tensor(
            element_type=cutlass.BFloat16,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )

        sB = smem.allocate_tensor(
            element_type=cutlass.BFloat16,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )

        tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=0, num_threads=self.threads_per_cta)

        # Tensor memory dealloc barrier init
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_barrier,
            is_two_cta=self.use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar,
        )

        # Alloc tensor memory buffer
        tmem.allocate(self.num_tmem_alloc_cols)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)

        # Initialize acc_pipeline (barrier) and states
        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.threads_per_cta)
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        acc_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_acc_stage)
        acc_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_acc_stage)

        for i in cutlass.range(tidx, cute.cosize(sA), self.threads_per_cta):
            if self.use_2cta_instrs:
                sA[i] = a[bidx * 64 + i % 64, i // 64]
                sB[i] = b[bidx * 64 + i % 64, i // 64]
            else:
                sA[i] = a[i]
                sB[i] = b[i]

        pipeline.sync(barrier_id=1)

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)
        tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

        if warp_idx == 0 and is_leader_cta:
            blk_count = tCrA.shape[2]
            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for i in cutlass.range_constexpr(blk_count):
                cute.gemm(tiled_mma, tCtAcc, tCrA[None, None, i, 0], tCrB[None, None, i, 0], tCtAcc)
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            acc_pipeline.producer_commit(acc_producer_state)

        acc_pipeline.consumer_wait(acc_consumer_state)

        tmem_load_atom = cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), self.acc_dtype)

        tCtAcc = tCtAcc[(None, None), 0, 0]

        tmem_tiled_copy = tcgen05.make_tmem_copy(tmem_load_atom, tCtAcc)

        tmem_thr_copy = tmem_tiled_copy.get_slice(tidx)
        tTR_tAcc = tmem_thr_copy.partition_S(tCtAcc)

        mma_tiler = (self.mma_tiler[0] // tiled_mma.thr_id.shape, self.mma_tiler[1], self.mma_tiler[2])

        cS = cute.make_identity_tensor(cute.select(mma_tiler, mode=[0, 1]))

        tTR_tS = tmem_thr_copy.partition_D(cS)

        tTR_rAcc = cute.make_fragment_like(tTR_tS, self.acc_dtype)
        cute.copy(tmem_tiled_copy, tTR_tAcc, tTR_rAcc)

        if self.use_2cta_instrs:
            for i in cutlass.range_constexpr(64):
                c[tidx % 64 + 64 * bidx, i + tidx // 64 * 64] = (cutlass.BFloat16)(tTR_rAcc[i])
        else:
            for i in cutlass.range_constexpr(128):
                c[tidx, i] = (cutlass.BFloat16)(tTR_rAcc[i])

        pipeline.sync(barrier_id=2)
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)


class TestDeepDenseGemm(unittest.TestCase):
    def setUp(self):
        pass

    def two_invoke(self, M, N, K):

        a = paddle.randn([M, K])
        b = paddle.randn([N, K])
        baseline_out = paddle.matmul(a, b, False, True)

        my_tensor = paddle.empty_like(baseline_out)

        mm = DenseGemmKernel()
        from cutlass.cute.runtime import from_dlpack

        my_a = from_dlpack(a, assumed_align=16)
        my_b = from_dlpack(b, assumed_align=16)
        my_res = from_dlpack(my_tensor, assumed_align=16)

        compiled_mm = cute.compile(
            mm,
            my_a,
            my_b,
            my_res,
            options="--opt-level 2",
        )
        compiled_mm(my_a, my_b, my_res)

        print(my_tensor)

        print(my_tensor - baseline_out)
        assert (my_tensor - baseline_out).abs().max().item() == 0.0

    def one_invoke(self, M, N, K):
        try:
            import deep_gemm
        except:
            return

        block_size = 128

        raw_x = paddle.randn([M, K], dtype="bfloat16").cast(paddle.float8_e4m3fn)
        raw_x_scale = paddle.randn([M, K // block_size], dtype="float32")

        raw_x_scale = paddle.randn([M, K // block_size], dtype="float32") * 10 + 127
        raw_x_scale = paddle.clip(raw_x_scale, 0, 127)
        raw_x_scale = raw_x_scale.cast("int32")
        raw_x_scale = raw_x_scale.cast("uint8").view("int32")

        float32_x_scale = raw_x_scale.view("uint8").cast("int32").flatten().numpy().tolist()
        for i in range(len(float32_x_scale)):
            float32_x_scale[i] = 2.0 ** (float32_x_scale[i] - 127)
        float32_x_scale = (
            paddle.to_tensor(float32_x_scale, dtype="float32")
            .reshape([M, K // block_size, 1])
            .tile([1, 1, block_size])
            .reshape([M, K])
        )

        raw_w = paddle.randn([N, K], dtype="bfloat16").cast(paddle.float8_e4m3fn)
        raw_w_scale = paddle.randn([N // block_size, K // block_size], dtype="float32")

        raw_w_scale = paddle.zeros([N, K // block_size], dtype="int32") + 128
        raw_w_scale = raw_w_scale.cast("uint8").view("int32")

        baseline_out = paddle.empty([M, N], dtype="bfloat16")
        tmp0 = raw_x.cast("float32") * float32_x_scale

        tmp1 = raw_w.cast("float32") * 2

        baseline_out = paddle.matmul(tmp0, tmp1, False, True)

        deepgemm_output = paddle.zeros_like(baseline_out)
        for i in range(10):
            a = paddle.zeros([1024, 1024, 1024]) + 1
            del a

            a = raw_x_scale.transpose([1, 0]).contiguous().transpose([1, 0])
            b = raw_w_scale.transpose([1, 0]).contiguous().transpose([1, 0])

            deep_gemm.fp8_gemm_nt(
                (raw_x, a),
                (raw_w, b),
                deepgemm_output,
            )

        print(baseline_out - deepgemm_output)
        # assert (baseline_out - deepgemm_output).abs().max().item() < 0.1

    def test_main(self):
        prop = paddle.device.cuda.get_device_properties()
        if prop.major != 10:
            return
        # import paddle.profiler as profiler
        # p = profiler.Profiler(
        #     targets=[profiler.ProfilerTarget.CPU, profiler.ProfilerTarget.GPU],
        #     on_trace_ready=profiler.export_chrome_tracing("./profile_log"),
        # )
        # p.start()
        # p.step()

        self.one_invoke(128 * 20, 2048, 4096)
        self.one_invoke(128 * 20, 2048, 2048)

        self.two_invoke(128, 128, 64)

        # p.stop()


if __name__ == "__main__":
    unittest.main()
