







import paddle



A = paddle.randn([128,7168])
B = paddle.randn([7168, 384])

C_baseline = paddle.matmul(A, B)


from fastdeploy.model_executor.ops.gpu import cute_gemm

C_fastdeploy = cute_gemm(A, B.transpose([1,0]).contiguous().transpose([1,0]))

print(C_fastdeploy - C_baseline)
print((C_fastdeploy - C_baseline).abs().max())


