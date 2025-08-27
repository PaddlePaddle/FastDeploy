import unittest

import paddle

from fastdeploy.config import (
    CacheConfig,
    FDConfig,
    GraphOptimizationConfig,
    ParallelConfig,
)
from fastdeploy.model_executor.forward_meta import ForwardMeta
from fastdeploy.model_executor.graph_optimization.decorator import (
    support_graph_optimization,
)
from fastdeploy.utils import print_gpu_memory_use


@support_graph_optimization
class TestCase1SubLayer1(paddle.nn.Layer):
    """Sub layer 1 of test case 1"""

    def __init__(self, fd_config: FDConfig, **kwargs):
        super().__init__()

    def forward(self, ids_remove_padding, forward_meta: ForwardMeta):
        """Sub layer1 forward pass"""

        output = paddle.add(forward_meta.input_ids, forward_meta.input_ids)
        return output

    def forward_correct(self, ids_remove_padding, forward_meta: ForwardMeta):
        """Sub layer1 Correct forward pass"""

        output = paddle.add(forward_meta.input_ids, forward_meta.input_ids)
        return output


class TestModel1(paddle.nn.Layer):
    """Tast Model"""

    def __init__(self, fd_config: FDConfig, **kwargs):
        super().__init__()
        self.fd_config = fd_config

        self.sublayer1 = TestCase1SubLayer1(self.fd_config)
        sublayer1_copy = TestCase1SubLayer1(self.fd_config)
        self.sublayer2 = sublayer1_copy

    def forward(self, ids_remove_padding, forward_meta: ForwardMeta):
        """Test model forward pass"""
        # sublayer1 use cuda graph
        sub_meta1 = forward_meta
        sublayer1_output = self.sublayer1(ids_remove_padding=ids_remove_padding, forward_meta=sub_meta1)

        # sublayer2 use cuda graph
        sub_meta2 = ForwardMeta(
            input_ids=sublayer1_output, ids_remove_padding=sublayer1_output, step_use_cudagraph=True
        )
        sublayer2_output = self.sublayer2(ids_remove_padding=sublayer1_output, forward_meta=sub_meta2)

        return sublayer2_output

    def forward_correct(self, ids_remove_padding, forward_meta: ForwardMeta):
        """Test model Correct forward pass"""
        # sublayer1 not use cuda graph
        sub_meta1 = forward_meta
        sublayer1_output = self.sublayer1.forward_correct(
            ids_remove_padding=ids_remove_padding, forward_meta=sub_meta1
        )

        # sublayer2 not use cuda graph
        sub_meta2 = ForwardMeta(input_ids=sublayer1_output, ids_remove_padding=sublayer1_output)
        sublayer2_output = self.sublayer2.forward_correct(ids_remove_padding=sublayer1_output, forward_meta=sub_meta2)

        return sublayer2_output

    def clear_grpah_opt_backend(self):
        """ """
        self.sublayer1.clear_grpah_opt_backend(fd_config=self.fd_config)
        self.sublayer2.clear_grpah_opt_backend(fd_config=self.fd_config)


class TestCUDAGrpahRecapture(unittest.TestCase):
    """
    Test CUDAGraph Memory change
    """

    def test_cuda_graph_recapture(self):
        """Run test case"""
        # Set FastDeploy config
        graph_opt_config = GraphOptimizationConfig(args={})
        graph_opt_config.use_cudagraph = True
        parallel_config = ParallelConfig(args={})
        cache_config = CacheConfig(args={})
        parallel_config.max_num_seqs = 1
        fd_config = FDConfig(
            graph_opt_config=graph_opt_config, parallel_config=parallel_config, cache_config=cache_config
        )

        # Run Test Case1
        self.test_model1 = TestModel1(fd_config=fd_config)
        input_tensor1 = paddle.ones([1, 32768])
        forward_meta1 = ForwardMeta(input_ids=input_tensor1, ids_remove_padding=input_tensor1, step_use_cudagraph=True)

        # Corrent output
        self.output_correct = self.test_model1.forward_correct(
            ids_remove_padding=input_tensor1, forward_meta=forward_meta1
        )

        # Capture and Destory
        self.capture_and_replay(input_tensor1, forward_meta1)
        self.recapture_and_replay(input_tensor1, forward_meta1)

    def capture_and_replay(self, input_tensor1, forward_meta1):
        """ """
        # Triger Capture
        print_gpu_memory_use(0, "before capture")
        output1 = self.test_model1(ids_remove_padding=input_tensor1, forward_meta=forward_meta1)
        print_gpu_memory_use(0, "after capture")

        # Reaplay
        output1 = self.test_model1(ids_remove_padding=input_tensor1, forward_meta=forward_meta1)
        assert (output1 == self.output_correct).all()

        # Destory
        print_gpu_memory_use(0, "before destory")
        self.test_model1.clear_grpah_opt_backend()
        print_gpu_memory_use(0, "after destory")

    def recapture_and_replay(self, input_tensor1, forward_meta1):
        """ """
        # Triger Capture
        print_gpu_memory_use(0, "before recapture")
        output2 = self.test_model1(ids_remove_padding=input_tensor1, forward_meta=forward_meta1)
        print_gpu_memory_use(0, "after recapture")

        # Reaplay
        output2 = self.test_model1(ids_remove_padding=input_tensor1, forward_meta=forward_meta1)
        assert (output2 == self.output_correct).all()

        # Destory
        print_gpu_memory_use(0, "before destory")
        self.test_model1.clear_grpah_opt_backend()
        print_gpu_memory_use(0, "after destory")


if __name__ == "__main__":
    unittest.main()
