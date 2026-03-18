# import paddle

# A = paddle.randn([128, 7168])
# B = paddle.randn([7168, 384])

# C_baseline = paddle.matmul(A, B)

# from fastdeploy.model_executor.ops.gpu import cute_gemm

# B_kmajor = B.transpose([1, 0]).contiguous().transpose([1, 0])

# C_fastdeploy = cute_gemm(A, B_kmajor)

# print(C_fastdeploy - C_baseline)
# print((C_fastdeploy - C_baseline).abs().max())

# --------------------------------------------------

import paddle
paddle.set_default_dtype("float16")

A = paddle.randn([5120, 4096])
B = paddle.randn([4096, 5120])

C_baseline = paddle.matmul(A, B)

from fastdeploy.model_executor.ops.gpu import cute_gemm

B_kmajor = B.transpose([1, 0]).contiguous().transpose([1, 0])

C_fastdeploy = cute_gemm(A, B_kmajor)


import numpy as np
num_tests = 100
start_events = [paddle.device.cuda.Event(enable_timing=True) for _ in range(num_tests)]
end_events = [paddle.device.cuda.Event(enable_timing=True) for _ in range(num_tests)]
for i in range(num_tests):
    start_events[i].record()

    C_fastdeploy = cute_gemm(A, B_kmajor)

    end_events[i].record()
paddle.device.cuda.synchronize()

times = np.array([round(s.elapsed_time(e),1) for s, e in zip(start_events, end_events)])[1:]
print(times[-5:])


print(C_fastdeploy)
print(C_baseline)
print(C_baseline - C_fastdeploy)
print((C_fastdeploy - C_baseline).abs().max())

