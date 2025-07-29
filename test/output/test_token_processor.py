import unittest
import time
from unittest.mock import MagicMock, patch

from fastdeploy.engine.request import RequestMetrics, RequestOutput, CompletionOutput

class TestTokenProcessor(unittest.TestCase):
    def setUp(self):
        # 创建最小的模拟对象
        self.processor = MagicMock()
        self.processor.tokens_counter = {}
        # 模拟资源管理器及相关属性
        self.processor.resource_manager = MagicMock()
        self.processor.resource_manager.stop_flags = [False] * 512
        self.processor.resource_manager.tasks_list = [None] * 512
        self.processor.resource_manager.req_dict = {}
        self.processor.postprocess = MagicMock()

    def test_request_start_time_assignment(self):
        # 同时测试tokens_counter为0和1两种情况
        for counter in [0, 1]:
            with self.subTest(counter=counter):
                # 1. 模拟人物对象
                task = MagicMock()
                task_id = f"test_task_{counter}"
                task.request_id = task_id
                task.arrival_time = time.time()
                
                # 2. 模拟_process_batch_output方法
                with patch.object(self.processor, '_process_batch_output') as mock_process:
                    def mock_impl():
                        metrics = RequestMetrics(
                            arrival_time=task.arrival_time if counter == 0 else time.time(),
                            request_start_time=task.arrival_time,
                        )
                        result = RequestOutput(
                            request_id=task_id,
                            outputs=CompletionOutput(index=0, send_idx=0, token_ids=[42]),
                            finished=False,
                            metrics=metrics
                        )
                        self.processor.postprocess([result])
                    mock_process.side_effect = mock_impl
                    
                    self.processor.tokens_counter[task_id] = counter
                    self.processor.resource_manager.tasks_list[0] = task
                    self.processor.resource_manager.req_dict = {task_id: task}
                    
                    self.processor._process_batch_output()
                    
                    # 验证postprocess被调用
                    args, _ = self.processor.postprocess.call_args
                    result = args[0][0]
                    # 验证结果中的request_start_time是否正确
                    self.assertEqual(result.metrics.request_start_time, task.arrival_time,
                                    f"request_start_time should equal task.arrival_time (counter={counter})")

if __name__ == "__main__":
    unittest.main()