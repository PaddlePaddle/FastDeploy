## **项目：在 FastDeploy 中原生支持 MiniMax-M1**

**目标**: 实现一个高性能、功能完整的 MiniMax-M1 推理后端，支持其混合注意力、MoE、DeepNorm 及超长上下文特性。

**核心技术路径**:
1.  **复用**: 最大化复用 GLM-4.5 PR 中已有的 Partial RoPE 和标准 GQA Attention 组件。
2.  **翻译与开发**: 将 vLLM 的 `lightning_attn.py` (Triton) 翻译为高性能的 CUDA C++ 算子，以支持 MiniMax-M1 的线性注意力层。
3.  **集成**: 在 Python 层构建完整的 MiniMax-M1 模型结构，并正确调度新开发的算子。

**核心策略**: 并行推进 CUDA 开发与 Python 集成，以 5 层模型快速验证核心算子，然后无缝扩展到 80 层全量模型。
---

### **Phase 0: 项目设置与配置 (1-2 天)**

在写任何代码之前，先搭建好项目的骨架。

**任务 1: 创建目录结构**
在 FastDeploy 代码库中，创建以下新文件和目录的占位符：
```bash
# 1. 创建 Mamba 算子的 C++/CUDA 目录
mkdir -p custom_ops/gpu_ops/mamba_attn

# 2. 创建 Mamba 算子的 Python 包装器目录和文件
touch fastdeploy/model_executor/layers/attention/ops/mamba_attention.py

# 3. 创建 Mamba 算子的 Python 后端文件
touch fastdeploy/model_executor/layers/attention/mamba_backend.py

# 4. 创建 MiniMax-M1 的模型定义文件
touch fastdeploy/model_executor/models/minimax_m1.py
```

**任务 2: 更新 `FDConfig`**
编辑 `fastdeploy/config.py`，在 `ModelConfig` 类中添加 MiniMax-M1 特有的配置项。
```python
# fastdeploy/config.py -> class ModelConfig

class ModelConfig:
    def __init__(self, model: str, **args):
        # ...
        self.partial_rotary_factor: float = 1.0
        
        # === 在这里添加新配置 ===
        self.attn_type_list: list = []
        self.layernorm_full_attention_alpha: float = 1.0
        self.layernorm_full_attention_beta: float = 1.0
        self.layernorm_linear_attention_alpha: float = 1.0
        self.layernorm_linear_attention_beta: float = 1.0
        self.layernorm_mlp_alpha: float = 1.0
        self.layernorm_mlp_beta: float = 1.0
        self.postnorm: bool = False
        # =======================

        for key, value in args.items():
            # ...
```

---

### **Phase 1: [核心开发] 实现 Mamba/线性注意力 CUDA 算子 (2-4 周)**

这是整个项目中技术含量最高、最耗时的部分。

**任务 1.1: 翻译 Triton Kernels 为 CUDA C++ (`mamba_impl.cuh`)**
*   **目标**: 将 `vllm/model_executor/layers/lightning_attn.py` 中的所有 `@triton.jit` 函数翻译成 `__global__` CUDA kernel。
*   **创建文件**: `custom_ops/gpu_ops/mamba_attn/mamba_impl.cuh`
*   **翻译指南**:
    1.  将 `_fwd_diag_kernel` 翻译为 `MambaFwdDiagKernel`。
    2.  将 `_fwd_kv_parallel` 翻译为 `MambaFwdKVParallelKernel`。
    3.  将 `_fwd_kv_reduce` 翻译为 `MambaFwdKVReduceKernel`。
    4.  将 `_fwd_none_diag_kernel` 翻译为 `MambaFwdNonDiagKernel`。
    5.  将 `_linear_attn_decode_kernel` 翻译为 `MambaDecodeKernel`。
    *   **关键替换**: `tl.program_id` -> `blockIdx`, `tl.load/store` -> `if-guarded memory access`, `tl.dot` -> 手写寄存器级矩阵乘法，`tl.sum` -> Warp/Block 级归约。

**任务 1.2: 编写 C++ Host 启动器 (`mamba.cu`)**
*   **目标**: 编写一个 C++ Host 函数，模仿 `lightning_attn.py` 中的 `_attention.forward` 和 `linear_decode_forward_triton` 的调度逻辑。
*   **创建文件**: `custom_ops/gpu_ops/mamba_attn/mamba.cu`
*   **代码框架**:
    ```cpp
    #include "mamba_impl.cuh"
    #include "paddle/extension.h"

    void MambaAttentionForwardKernel(
        const paddle::Tensor& q, const paddle::Tensor& k, const paddle::Tensor& v,
        const paddle::Tensor& slope_rate, paddle::Tensor& mamba_state,
        paddle::Tensor& out, /* ... 其他 ForwardMeta 参数 ... */
    ) {
        // 1. 从 ForwardMeta 解析出 is_prefill, b, h, n, d 等信息
        bool is_prefill = ...; // 根据 seq_lens 判断

        // 2. 模仿 _attention.forward 和 linear_decode_forward_triton
        if (is_prefill) {
            // 计算 grid/block 维度
            // 依次启动 MambaFwd... 系列的 CUDA Kernel
        } else { // Decode
            // 计算 grid/block 维度
            // 启动 MambaDecodeKernel
        }
    }
    ```

**任务 1.3: 暴露自定义算子 (`mamba.cu` & `cpp_extensions.cc`)**
1.  在 `mamba.cu` 文件末尾添加算子注册代码：
    ```cpp
    PD_BUILD_STATIC_OP(mamba_attention_forward)
        .Inputs({ ... })
        .Outputs({ "Out", "MambaStateOut" })
        .SetKernelFn(PD_KERNEL(MambaAttentionForwardKernel));
    ```
2.  编辑 `custom_ops/gpu_ops/cpp_extensions.cc`，添加 `m.def("mamba_attention_forward", &MambaAttentionForward);`。
3.  更新 `custom_ops/setup_ops.py` 或 `CMakeLists.txt`，将 `mamba.cu` 加入编译列表。

**里程碑**: 完成并编译通过后，拥有了一个底层的、可被调用的 Mamba/线性注意力算子。

---

### **Phase 2: 构建 Python 接口与后端 (1 周)**

**任务 2.1: 创建底层 Python 包装器**
*   **文件**: `fastdeploy/model_executor/layers/attention/ops/mamba_attention.py`
*   **代码**:
    ```python
    from paddle.fluid import core
    
    def mamba_attention_forward(q, k, v, slope_rate, mamba_state, ...):
        # 实际调用 C++ 扩展
        out, mamba_state_out = core.eager._run_custom_op(
            "mamba_attention_forward", q, k, v, slope_rate, mamba_state, ...
        )
        return out, mamba_state_out
    ```

**任务 2.2: 创建高级 MambaBackend**
*   **文件**: `fastdeploy/model_executor/layers/attention/mamba_backend.py`
*   **代码框架**:
    ```python
    from .ops.mamba_attention import mamba_attention_forward
    from fastdeploy.model_executor.layers.attention.base_attention_backend import AttentionBackend

    class MambaBackend(AttentionBackend):
        def __init__(self, fd_config, ...):
            # 初始化 Mamba 需要的参数
            pass

        def init_attention_metadata(self, forward_meta: ForwardMeta):
            # Mamba 可能需要一些独特的元数据准备
            pass

        def forward(self, q, k, v, qkv, ..., layer: Attention, forward_meta: ForwardMeta):
            # 1. 这里是核心业务逻辑，模仿 vLLM 的 MiniMaxText01LinearAttention._forward
            #    它会从 forward_meta 中解析调度信息
            
            # 2. 调用底层算子
            out, ssm_states_out = mamba_attention_forward(...)
            
            # 3. 更新 mamba_state (可能通过 forward_meta.caches)
            forward_meta.caches[layer.layer_id].copy_(ssm_states_out, False)

            return out
    ```


---

### **Phase 3: 全量模型支持与性能验证 (1 周)**

**目标**: 将已在 5 层模型上验证通过的核心组件无缝扩展至 80 层全量模型。

修改 `minimax_m1.py` 中的 `load_weights` 函数，使其能够加载并正确映射**全部 80 层**的权重。在启动脚本中，将 `num_hidden_layers` 设置为 80，并且对齐vllm精度。

---

### **Phase 4: 模型集成与测试 (1 周)**

**任务 3.1: 编写 `minimax_m1.py`**
*   **文件**: `fastdeploy/model_executor/models/minimax_m1.py`
*   **核心逻辑**:
    ```python
    from fastdeploy.model_executor.layers.attention.mamba_backend import MambaBackend
    from fastdeploy.model_executor.layers.attention.mla_attention_backend import MLAAttentionBackend

    class MiniMaxM1DecoderLayer(nn.Layer):
        def __init__(self, fd_config, layer_id):
            attn_type = fd_config.model_config.attn_type_list[layer_id]
            if attn_type == 0: # 线性注意力
                self.self_attn = MiniMaxM1LinearAttention(fd_config, layer_id) # 这是一个新的 nn.Layer
            else: # 标准注意力
                self.self_attn = MiniMaxM1StandardAttention(fd_config, layer_id) # 复用/包装标准 Attention

            # DeepNorm 参数
            self.layernorm_attention_alpha = ...
        
        def forward(self, hidden_states, ...):
            # 实现 DeepNorm 和 MoE+SharedExpert 逻辑
            attn_output = self.self_attn(...)
            layernorm_input = residual * self.layernorm_attention_alpha + attn_output * self.layernorm_attention_beta
            ...
    
    # 建议为两种 Attention 创建独立的 nn.Layer 包装类
    class MiniMaxM1LinearAttention(nn.Layer):
        def __init__(...):
            self.backend = MambaBackend(...)
            self.qkv_proj = ...
            self.out_proj = ...
            self.norm = ...
            # ...
        def forward(...):
            qkv = self.qkv_proj(...)
            # ... 准备输入 ...
            attn_out = self.backend.forward(...)
            # ... 后处理 ...
            return final_out

    # MiniMaxM1StandardAttention 类似
    ```

**任务 3.2: 编写测试用例**
*   **单元测试**: 编写一个 Python 测试脚本，直接调用 `ops.mamba_attention_forward`，并与一个纯 Python/Paddle 实现的、功能等价的 Mamba 算法进行对比，验证数值精度。
*   **端到端测试**: 创建一个完整的 MiniMax-M1 模型实例，加载权重，并运行推理。将输出与 Hugging Face `transformers` 的参考输出进行对比。

**任务 3.3: 性能基准测试**
*   在与 vLLM 相同的硬件和环境下，测试 FastDeploy 实现的吞吐量和时延，确保性能达标。

