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

print(C_fastdeploy)
print(C_baseline)
print((C_fastdeploy - C_baseline).abs().max())

