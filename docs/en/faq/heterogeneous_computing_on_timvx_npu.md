English | [中文](../../cn/faq/heterogeneous_computing_on_timvx_npu.md)

# Heterogeneous Computing on VeriSilicon Series NPUs
When deploying a quantized model on a VeriSilicon series NPU, such as RV1126 or A311D, there may be a problem of decreased accuracy, so heterogeneous computing needs to be performed on the NPU and ARM CPU. The heterogeneous computing in FastDeploy is implemented by loading subgraph.txt configuration files. If you find that the accuracy has dropped significantly after replacing the quantized model, you can refer to this document to define the heterogeneous configuration file.

Update steps for heterogeneous configuration files:
1. Determine the accuracy of the quantized model on an ARM CPU.
- If the accuracy cannot be satisfied on the ARM CPU, then there is a problem with the quantized model. At this time, you can consider modifying the dataset or changing the quantization method.
- Only need to modify a few lines of code in the demo, change the part of NPU inference to use ARM CPU int8.
    ```
    # The following interface represents the use of NPU for inference
    fastdeploy::RuntimeOption option;
    option.UseTimVX(); # Turn on TIMVX for NPU inference
    option.SetLiteSubgraphPartitionPath(subgraph_file); # Load heterogeneous computing configuration files

    # The following interface indicates the use of ARM CPU int8 inference
    fastdeploy::RuntimeOption option;
    option.UseLiteBackend();
    option.EnableLiteInt8();
    ```
    If the ARM CPU accuracy is up to standard, continue with the next steps.

2. Obtain the topology information of the entire network.
- Roll back the modification in the first step, use the API interface of NPU for inference, and keep the switch of loading heterogeneous computing configuration files off.
- Write all the log information to log.txt, search for the keyword "subgraph operators" in log.txt and the following paragraph is the topology information of the entire model.
- It has the following format:
    - Each line of records consists of "operator type: list of input tensor names: list of output tensor names" (that is, the operator type, list of input and output tensor names are separated by semicolons), and the input and output tensor names are separated by commas each tensor name in the list;
    - Example:
    ```
        op_type0:var_name0,var_name1:var_name2 # Indicates that the node whose operator type is op_type0, input tensors are var_name0 and var_name1, and output tensor is var_name2 is forced to run on the ARM CPU
    ```

3. Modify heterogeneous configuration files
- Write all Subgraph operators in subgraph.txt, and open the interface for loading heterogeneous computing configuration files
- Delete line by line, delete in pieces, dichotomy, use the patience of developers, find the layer that introduces NPU precision exception, and leave it in subgraph.txt
- The nodes in txt all need to be heterogeneous to the layer on the ARM CPU, so don’t worry about performance issues. The ARM kernel performance of Paddle Lite is also very good.
