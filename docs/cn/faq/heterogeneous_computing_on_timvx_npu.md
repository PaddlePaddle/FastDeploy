[English](../../en/faq/heterogeneous_computing_on_timvx_npu.md) | 中文

# 在芯原系列 NPU 上实现异构计算
在芯原系列 NPU 上，例如 RV1126 或者 A311D 上部署全量化模型时，有可能会有精度下降的问题，那么就需要在 NPU 和 ARM CPU 上进行异构计算，FastDeploy 中的异构计算是通过 subgraph.txt 配置文件来完成的，如果在更换全量化模型后，发现精度有较大的下降，可以参考本文档来定义异构配置文件。

异构配置文件的更新步骤：
1. 确定模型量化后在 ARM CPU 上的精度。
- 如果在 ARM CPU 上，精度都无法满足，那量化本身就是失败的，此时可以考虑修改训练集或者更改量化方法。
- 只需要修改 demo 中的代码，将 NPU 推理的部分改为使用 ARM CPU int8 推理，便可实现使用ARM CPU进行计算
    ```
    # 如下接口表示使用 NPU 进行推理
    fastdeploy::RuntimeOption option;
    option.UseTimVX(); # 开启 TIMVX 进行 NPU 推理
    option.SetLiteSubgraphPartitionPath(subgraph_file); # 加载异构计算配置文件

    # 如下接口表示使用 ARM CPU int8 推理
    fastdeploy::RuntimeOption option;
    option.UseLiteBackend();
    option.EnableLiteInt8();
    ```
    如果 ARM CPU 计算结果精度达标，则继续下面的步骤。

2. 获取整网拓扑信息。
- 回退第一步中的修改，使用 NPU 进行推理的 API 接口，加载异构计算配置文件的开关保持关闭。
- 将所有的日志信息写入到 log.txt中，在 log.txt 中搜索关键字 "subgraph operators" 随后的一段便是整个模型的拓扑信息
- 它的格式如下：
    - 每行记录由 ”算子类型:输入张量名列表:输出张量名列表“ 组成（即以分号分隔算子类型、输入和输出张量名列表），以逗号分隔输入、输出张量名列表中的每个张量名；
    - 示例说明：
    ```
        op_type0:var_name0,var_name1:var_name2 # 表示将算子类型为 op_type0、输入张量为var_name0 和 var_name1、输出张量为 var_name2 的节点强制运行在 ARM CPU 上
    ```

3. 修改异构配置文件
- 将所有的 Subgraph operators 写到在 subgraph.txt 中，并打开加载异构计算配置文件的接口
- 逐行删除、成片删除、二分法，发挥开发人员的耐心，找到引入 NPU 精度异常的 layer，将其留在 subgraph.txt 中
- 在 txt 中的结点都是需要异构到 ARM CPU 上的 layer，不用特别担心性能问题，Paddle Lite 的 ARM kernel 性能也是非常卓越的
