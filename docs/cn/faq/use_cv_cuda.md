# 使用CV-CUDA/CUDA加速GPU端到端推理性能

FastDeploy集成了CV-CUDA来加速预/后处理，个别CV-CUDA不支持的算子使用了CUDA kernel的方式实现。

FastDeploy的Vision Processor模块对CV-CUDA的算子做了进一步的封装，用户不需要自己去调用CV-CUDA，
使用FastDeploy的模型推理接口即可利用CV-CUDA的加速能力。

FastDeploy的Vision Processor模块在集成CV-CUDA时，做了以下工作来方便用户的使用：
- GPU内存管理，缓存算子的输入、输出tensor，避免重复分配GPU内存
- CV-CUDA不支持的个别算子利用CUDA kernel实现
- CV-CUDA/CUDA不支持的算子可以fallback到OpenCV/FlyCV

## 使用方式
编译FastDeploy时，开启CV-CUDA编译选项
```bash
# 编译C++预测库时, 开启CV-CUDA编译选项.
-DENABLE_CVCUDA=ON \

# 在编译Python预测库时, 开启CV-CUDA编译选项
export ENABLE_CVCUDA=ON
```

只有继承了ProcessorManager类的模型预处理，才可以使用CV-CUDA，这里以PaddleClasPreprocessor为例
```bash
# C++
# 创建model之后，调用model preprocessor的UseCuda接口即可打开CV-CUDA/CUDA预处理
# 第一个参数enable_cv_cuda，true代表使用CV-CUDA，false代表只使用CUDA（支持的算子较少）
# 第二个参数是GPU id，-1代表不指定，使用当前GPU
model.GetPreprocessor().UseCuda(true, 0);

# Python
model.preprocessor.use_cuda(True, 0)
```

## 最佳实践

- 如果预处理第一个算子是resize，则要根据实际情况决定resize是否跑在GPU。因为当resize跑在GPU，
  且图片解码在CPU时，需要把原图copy到GPU内存，开销较大，而resize之后再copy到GPU内存，则往往只需要
  copy较少的数据。
