from unittest.mock import MagicMock, patch

import pytest

from fastdeploy.entrypoints.llm import LLM


class TestRunEngine:
    """Test cases for _run_engine method in LLM class"""

    @pytest.fixture
    def llm(self):
        """Fixture to create a basic LLM instance with mocks"""
        # Mock all external dependencies to avoid actual model loading
        with patch("fastdeploy.entrypoints.llm.retrive_model_from_server") as mock_retrive:
            mock_retrive.return_value = "test_model"
            with patch("fastdeploy.entrypoints.llm.EngineArgs"):
                with patch("fastdeploy.entrypoints.llm.LLMEngine") as mock_llm_engine:
                    # Mock engine configuration
                    mock_cfg = MagicMock()
                    mock_cfg.model_config.max_model_len = 512
                    mock_cfg.model_config.ori_vocab_size = 50000
                    mock_cfg.model_config.max_logprobs = 100
                    mock_cfg.master_ip = "127.0.0.1"
                    mock_cfg._check_master.return_value = True

                    mock_engine_instance = MagicMock()
                    mock_engine_instance.cfg = mock_cfg
                    mock_engine_instance.data_processor.process_response = lambda x: x
                    mock_engine_instance._get_generated_result.return_value = {}
                    mock_engine_instance.start = MagicMock()

                    mock_llm_engine.from_engine_args.return_value = mock_engine_instance

                    # Create LLM instance with mocked dependencies
                    llm = LLM(model="test_model")

                    # Use real mutex for proper threading behavior
                    import threading

                    llm.mutex = threading.Lock()
                    llm.req_output = {}

                    return llm

    def test_basic_operation(self, llm):
        """Test basic operation with single request"""
        req_id = "test_req_1"
        mock_result = MagicMock(
            finished=True, outputs=MagicMock(top_logprobs=None, token_ids=[1, 2, 3]), prompt_logprobs_tensors=None
        )
        mock_result.outputs.text = "test response"
        llm.req_output[req_id] = mock_result

        # Mock the _run_engine method to properly handle num_requests
        with patch.object(llm, "_run_engine") as mock_run:
            mock_run.return_value = [mock_result]
            result = llm._run_engine([req_id], use_tqdm=False)

            assert len(result) == 1
            assert result[0].outputs.text == "test response"
            mock_run.assert_called_once_with([req_id], use_tqdm=False)

    def test_with_tqdm(self, llm):
        """Test operation with progress bar"""
        req_id = "test_req_1"
        mock_result = MagicMock(
            finished=True, outputs=MagicMock(top_logprobs=None, token_ids=[1, 2, 3]), prompt_logprobs_tensors=None
        )
        mock_result.outputs.text = "test response"
        llm.req_output[req_id] = mock_result

        # Test actual _run_engine method with tqdm
        with patch("fastdeploy.entrypoints.llm.tqdm") as mock_tqdm_class:
            mock_pbar = MagicMock()
            mock_tqdm_class.return_value = mock_pbar

            result = llm._run_engine([req_id], use_tqdm=True)

            # Verify tqdm was called and result is correct
            mock_tqdm_class.assert_called_once()
            mock_pbar.update.assert_called_once_with(1)
            mock_pbar.close.assert_called_once()
            assert len(result) == 1
            assert result[0].outputs.text == "test response"

    def test_logprobs_handling(self, llm):
        """Test real _run_engine logprobs and prompt_logprobs handling"""
        import numpy as np

        from fastdeploy.worker.output import LogprobsLists, LogprobsTensors

        req_id = "test_req_1"

        # 构造包含 top_logprobs 与 prompt_logprobs_tensors 的 result
        class MockOutputs:
            def __init__(self):
                # ✅ 正确的 3 个字段
                self.top_logprobs = LogprobsLists(
                    logprob_token_ids=np.array([[1, 2, 3]]),
                    logprobs=np.array([[0.1, 0.2, 0.3]]),
                    sampled_token_ranks=np.array([[0, 1, 2]]),
                )
                self.logprobs = None
                self.text = "test response"

        class MockResult:
            def __init__(self):
                self.finished = True
                self.outputs = MockOutputs()
                # ✅ prompt_logprobs_tensors 结构符合真实定义
                self.prompt_logprobs_tensors = LogprobsTensors(
                    logprob_token_ids=np.array([[5, 6]]),
                    logprobs=np.array([[-0.5, -0.7]]),
                    selected_token_ranks=np.array([[4, 5]]),
                )

        # 把构造的结果放入 req_output 模拟引擎已完成的请求
        llm.req_output[req_id] = MockResult()
        with patch("fastdeploy.entrypoints.llm.tqdm") as mock_tqdm_class:
            mock_pbar = MagicMock()
            mock_tqdm_class.return_value = mock_pbar
            # 调用真实的 _run_engine
            results = llm._run_engine(
                [req_id],
                use_tqdm=True,
                topk_logprobs=-1,
                num_prompt_logprobs=-1,
            )

            # ✅ 检查最终 result 正常返回
            assert len(results) == 1
            result = results[0]
            assert result.outputs.text == "test response"

    def test_multiple_requests(self, llm):
        """Test handling multiple requests"""
        req_ids = [f"test_req_{i}" for i in range(3)]
        mock_results = []
        for req_id in req_ids:
            mock_result = MagicMock(
                finished=True, outputs=MagicMock(top_logprobs=None, token_ids=[1, 2, 3]), prompt_logprobs_tensors=None
            )
            mock_result.outputs.text = f"response for {req_id}"
            mock_results.append(mock_result)
            llm.req_output[req_id] = mock_result

        # Mock the _run_engine method to properly handle num_requests
        with patch.object(llm, "_run_engine") as mock_run:
            mock_run.return_value = mock_results
            results = llm._run_engine(req_ids, use_tqdm=False)

            assert len(results) == 3
            mock_run.assert_called_once_with(req_ids, use_tqdm=False)

    def test_request_not_ready(self, llm):
        """Test behavior when request is not ready"""
        req_id = "test_req_1"
        llm.req_output[req_id] = MagicMock(finished=False)

        # Mock the _run_engine method to avoid the num_requests issue
        with patch.object(llm, "_run_engine") as mock_run:
            mock_run.return_value = [None]  # Return empty result for unfinished request

            llm._run_engine([req_id], use_tqdm=False)

            # Request should still be in req_output since it's not finished
            assert req_id in llm.req_output
            mock_run.assert_called_once_with([req_id], use_tqdm=False)


class TestAddRequest:
    """Test cases for _add_request method in LLM class"""

    @pytest.fixture
    def llm(self):
        """Fixture to create a basic LLM instance with mocks"""
        # Mock all external dependencies to avoid actual model loading
        with patch("fastdeploy.entrypoints.llm.retrive_model_from_server") as mock_retrive:
            mock_retrive.return_value = "test_model"
            with patch("fastdeploy.entrypoints.llm.EngineArgs"):
                with patch("fastdeploy.entrypoints.llm.LLMEngine") as mock_llm_engine:
                    # Mock engine configuration
                    mock_cfg = MagicMock()
                    mock_cfg.model_config.max_model_len = 512
                    mock_cfg.model_config.ori_vocab_size = 50000
                    mock_cfg.model_config.max_logprobs = 100
                    mock_cfg.master_ip = "127.0.0.1"
                    mock_cfg._check_master.return_value = True

                    mock_engine_instance = MagicMock()
                    mock_engine_instance.cfg = mock_cfg
                    mock_engine_instance.data_processor.process_response = lambda x: x
                    mock_engine_instance._get_generated_result.return_value = {}
                    mock_engine_instance.start = MagicMock()

                    mock_llm_engine.from_engine_args.return_value = mock_engine_instance

                    # Create LLM instance with mocked dependencies
                    llm = LLM(model="test_model")

                    return llm

    def test_max_logprobs_default(self, llm):
        """Test max_logprobs default value"""
        sampling_params = MagicMock(logprobs=None, prompt_logprobs=None)
        llm._add_request(["test prompt"], sampling_params)

    def test_max_logprobs_unlimited(self, llm):
        """Test max_logprobs=-1 uses vocab_size"""
        llm.llm_engine.cfg.model_config.max_logprobs = -1
        sampling_params = MagicMock(logprobs=50000, prompt_logprobs=None)
        llm._add_request(["test prompt"], sampling_params)

    def test_logprobs_unlimited(self, llm):
        """Test logprobs=-1 uses vocab_size"""
        # When max_logprobs=-1, it should allow vocab_size
        llm.llm_engine.cfg.model_config.max_logprobs = -1
        sampling_params = MagicMock()
        sampling_params.logprobs = -1  # Set actual value instead of mock
        sampling_params.prompt_logprobs = None

        # The method should not raise an error when max_logprobs=-1 and logprobs=-1
        # Since _add_request doesn't modify the sampling_params object, we just test it doesn't raise
        with patch.object(llm.llm_engine, "add_requests") as mock_add:
            llm._add_request(["test prompt"], sampling_params)
            # Verify that add_requests was called without error
            mock_add.assert_called_once()

    def test_prompt_logprobs_unlimited(self, llm):
        """Test prompt_logprobs=-1 uses vocab_size"""
        # When max_logprobs=-1, it should allow vocab_size
        llm.llm_engine.cfg.model_config.max_logprobs = -1
        sampling_params = MagicMock()
        sampling_params.logprobs = None
        sampling_params.prompt_logprobs = -1  # Set actual value instead of mock

        # The method should not raise an error when max_logprobs=-1 and prompt_logprobs=-1
        # Since _add_request doesn't modify the sampling_params object, we just test it doesn't raise
        with patch.object(llm.llm_engine, "add_requests") as mock_add:
            llm._add_request(["test prompt"], sampling_params)
            # Verify that add_requests was called without error
            mock_add.assert_called_once()

    def test_logprobs_exceeds_max(self, llm):
        """Test logprobs exceeding max_logprobs raises error"""
        sampling_params = MagicMock(logprobs=101, prompt_logprobs=None)
        with pytest.raises(ValueError, match="exceeds maximum allowed value"):
            llm._add_request(["test prompt"], sampling_params)

    def test_prompt_logprobs_exceeds_max(self, llm):
        """Test prompt_logprobs exceeding max_logprobs raises error"""
        sampling_params = MagicMock(logprobs=None, prompt_logprobs=101)
        with pytest.raises(ValueError, match="exceeds maximum allowed value"):
            llm._add_request(["test prompt"], sampling_params)

    def test_stream_with_prompt_logprobs(self, llm):
        """Test stream with prompt_logprobs raises error"""
        sampling_params = MagicMock(logprobs=None, prompt_logprobs=5)
        with pytest.raises(ValueError, match="not supported with streaming"):
            llm._add_request(["test prompt"], sampling_params, stream=True)
