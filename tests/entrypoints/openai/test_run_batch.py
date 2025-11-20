import asyncio
import json
import os
import subprocess
import tempfile
import unittest
from http import HTTPStatus
from unittest.mock import AsyncMock, MagicMock, Mock, mock_open, patch

from tqdm import tqdm

from fastdeploy.entrypoints.openai.protocol import (
    BatchRequestOutput,
    BatchResponseData,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    ErrorResponse,
    UsageInfo,
)
from fastdeploy.entrypoints.openai.run_batch import (
    _BAR_FORMAT,
    BatchProgressTracker,
    ModelPath,
    cleanup_resources,
    create_model_paths,
    create_serving_handlers,
    determine_process_id,
    init_engine,
    initialize_engine_client,
    main,
    make_async_error_request_output,
    make_error_request_output,
    parse_args,
    random_uuid,
    read_file,
    run_batch,
    run_request,
    setup_engine_and_handlers,
    upload_data,
    write_file,
    write_local_file,
)

INPUT_BATCH = """
{"custom_id": "req-00001", "method": "POST", "url": "/v1/chat/completions", "body": {"messages": [{"role": "user", "content": "Can you write a short poem? (id=1)"}], "temperature": 0.7, "max_tokens": 200}}
{"custom_id": "req-00002", "method": "POST", "url": "/v1/chat/completions", "body": {"messages": [{"role": "user", "content": "What can you do? (id=2)"}], "temperature": 0.7, "max_tokens": 200}}
{"custom_id": "req-00003", "method": "POST", "url": "/v1/chat/completions", "body": {"messages": [{"role": "user", "content": "Hello, who are you? (id=3)"}], "temperature": 0.7, "max_tokens": 200}}
"""

INVALID_INPUT_BATCH = """
{"invalid_field": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"messages": [{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 1000}}
{"custom_id": "request-2", "method": "POST", "url": "/v1/chat/completions", "body": {"messages": [{"role": "system", "content": "You are an unhelpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 1000}}
"""

BATCH_RESPONSE = """
{"id":"fastdeploy-7fcc30e2e4334fca806c4d01ee7ac4ab","custom_id":"req-00001","response":{"status_code":200,"request_id":"fastdeploy-batch-5f4017beded84b15aa3a8b0f1fce154c","body":{"id":"chatcmpl-33b09ae5-a8f1-40ad-9110-efa2b381eac9","object":"chat.completion","created":1758698637,"model":"/root/paddlejob/zhaolei36/ernie-4_5-0_3b-bf16-paddle","choices":[{"index":0,"message":{"role":"assistant","content":"In a sunlit meadow where dreams bloom,\\nA gentle breeze carries the breeze,\\nThe leaves rustle like ancient letters,\\nAnd in the sky, a song of hope and love.","multimodal_content":null,"reasoning_content":null,"tool_calls":null,"prompt_token_ids":null,"completion_token_ids":null,"prompt_tokens":null,"completion_tokens":null},"logprobs":null,"finish_reason":"stop"}],"usage":{"prompt_tokens":19,"total_tokens":60,"completion_tokens":41,"prompt_tokens_details":{"cached_tokens":0}}}},"error":null}
{"id":"fastdeploy-bf549849df2145598ae1758ba260f784","custom_id":"req-00002","response":{"status_code":200,"request_id":"fastdeploy-batch-81223f12fdc345efbfe85114ced10a1d","body":{"id":"chatcmpl-9479e36c-1542-45ff-b364-1dc6d34be9e7","object":"chat.completion","created":1758698637,"model":"/root/paddlejob/zhaolei36/ernie-4_5-0_3b-bf16-paddle","choices":[{"index":0,"message":{"role":"assistant","content":"Based on the given text, here are some possible actions you can take:\\n\\n1. **Read the question**: To understand what you can do, you can read the question (id=2) and analyze its requirements or constraints.\\n2. **Identify the keywords**: Look for specific keywords or phrases that describe what you can do. For example, if the question mentions \\"coding,\\" you can focus on coding skills or platforms.\\n3. **Brainstorm ideas**: You can think creatively about different ways to perform the action. For example, you could brainstorm different methods of communication, data analysis, or problem-solving.\\n4. **Explain your action**: If you have knowledge or skills in a particular area, you can explain how you would use those skills to achieve the desired outcome.\\n5. **Ask for help**: If you need assistance, you can ask for help from a friend, teacher, or mentor.","multimodal_content":null,"reasoning_content":null,"tool_calls":null,"prompt_token_ids":null,"completion_token_ids":null,"prompt_tokens":null,"completion_tokens":null},"logprobs":null,"finish_reason":"stop"}],"usage":{"prompt_tokens":17,"total_tokens":211,"completion_tokens":194,"prompt_tokens_details":{"cached_tokens":0}}}},"error":null}
"""


class TestArgParser(unittest.TestCase):
    """测试参数解析相关函数"""

    @patch("fastdeploy.entrypoints.openai.run_batch.FlexibleArgumentParser")
    @patch("fastdeploy.entrypoints.openai.run_batch.EngineArgs")
    def test_make_arg_parser(self, mock_engine_args, mock_parser_class):
        """测试make_arg_parser函数"""
        from fastdeploy.entrypoints.openai.run_batch import make_arg_parser

        mock_parser = Mock()
        mock_parser_class.return_value = mock_parser

        # 让EngineArgs.add_cli_args返回parser本身
        mock_engine_args.add_cli_args.return_value = mock_parser

        result = make_arg_parser(mock_parser)

        # 验证参数被正确添加
        mock_parser.add_argument.assert_any_call("-i", "--input-file", required=True, type=str, help=unittest.mock.ANY)
        mock_parser.add_argument.assert_any_call(
            "-o", "--output-file", required=True, type=str, help=unittest.mock.ANY
        )
        mock_parser.add_argument.assert_any_call("--output-tmp-dir", type=str, default=None, help=unittest.mock.ANY)
        mock_engine_args.add_cli_args.assert_called_once_with(mock_parser)
        # 现在应该返回parser而不是EngineArgs.add_cli_args的返回值
        self.assertEqual(result, mock_parser)

    @patch("fastdeploy.entrypoints.openai.run_batch.FlexibleArgumentParser")
    @patch("fastdeploy.entrypoints.openai.run_batch.make_arg_parser")
    @patch("fastdeploy.entrypoints.openai.run_batch.console_logger")
    def test_parse_args(self, mock_logger, mock_make_parser, mock_parser_class):
        """测试parse_args函数"""
        mock_parser = Mock()
        mock_args = Mock()
        mock_parser_class.return_value = mock_parser
        mock_parser.parse_args.return_value = mock_args
        mock_make_parser.return_value = mock_parser

        result = parse_args()

        mock_parser_class.assert_called_once_with(description="FastDeploy OpenAI-Compatible batch runner.")
        mock_make_parser.assert_called_once_with(mock_parser)
        mock_parser.parse_args.assert_called_once()
        self.assertEqual(result, mock_args)


class TestEngineInitialization(unittest.TestCase):
    """测试引擎初始化相关函数"""

    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    @patch("fastdeploy.entrypoints.openai.run_batch.LLMEngine")
    @patch("fastdeploy.entrypoints.openai.run_batch.EngineArgs")
    @patch("fastdeploy.entrypoints.openai.run_batch.api_server_logger")
    @patch("fastdeploy.entrypoints.openai.run_batch.os")
    def test_init_engine_success(self, mock_os, mock_logger, mock_engine_args, mock_llm_engine):
        """测试init_engine成功初始化"""

        with patch("fastdeploy.entrypoints.openai.run_batch.llm_engine", None):
            mock_args = Mock()
            mock_engine_args.from_cli_args.return_value = Mock()
            mock_engine = Mock()
            mock_engine.start.return_value = True
            mock_llm_engine.from_engine_args.return_value = mock_engine
            mock_os.getpid.return_value = 123

            result = init_engine(mock_args)

            mock_engine_args.from_cli_args.assert_called_with(mock_args)
            mock_llm_engine.from_engine_args.assert_called_with(mock_engine_args.from_cli_args.return_value)
            mock_engine.start.assert_called_with(api_server_pid=123)
            mock_logger.info.assert_called_with("FastDeploy LLM API server starting... 123")
            self.assertEqual(result, mock_engine)

    @patch("fastdeploy.entrypoints.openai.run_batch.LLMEngine")
    @patch("fastdeploy.entrypoints.openai.run_batch.EngineArgs")
    @patch("fastdeploy.entrypoints.openai.run_batch.api_server_logger")
    def test_init_engine_failure(self, mock_logger, mock_engine_args, mock_llm_engine):
        """测试init_engine初始化失败"""
        with patch("fastdeploy.entrypoints.openai.run_batch.llm_engine", None):
            mock_args = Mock()
            mock_engine_args.from_cli_args.return_value = Mock()
            mock_engine = Mock()
            mock_engine.start.return_value = False
            mock_llm_engine.from_engine_args.return_value = mock_engine

            result = init_engine(mock_args)

            mock_logger.error.assert_called_with("Failed to initialize FastDeploy LLM engine, service exit now!")
            self.assertIsNone(result)

    @patch("fastdeploy.entrypoints.openai.run_batch.LLMEngine")
    def test_init_engine_already_initialized(self, mock_llm_engine):
        """测试init_engine已经初始化的情况"""
        existing_engine = Mock()
        with patch("fastdeploy.entrypoints.openai.run_batch.llm_engine", existing_engine):
            mock_args = Mock()
            result = init_engine(mock_args)

            mock_llm_engine.from_engine_args.assert_not_called()
            self.assertEqual(result, existing_engine)

    @patch("fastdeploy.entrypoints.openai.run_batch.EngineClient")
    async def test_initialize_engine_client(self, mock_engine_client):
        """测试初始化引擎客户端"""
        mock_args = Mock()
        mock_args.model = "test-model"
        mock_args.tokenizer = "test-tokenizer"
        mock_args.max_model_len = 1024
        mock_args.tensor_parallel_size = 1
        mock_args.engine_worker_queue_port = [8000]
        mock_args.local_data_parallel_id = 0
        mock_args.limit_mm_per_prompt = None
        mock_args.mm_processor_kwargs = {}
        mock_args.reasoning_parser = None
        mock_args.data_parallel_size = 1
        mock_args.enable_logprob = False
        mock_args.workers = 1
        mock_args.tool_call_parser = None

        mock_client_instance = AsyncMock()
        mock_engine_client.return_value = mock_client_instance

        pid = 123
        result = await initialize_engine_client(mock_args, pid)

        # 验证EngineClient被正确初始化
        mock_engine_client.assert_called_once()
        mock_client_instance.connection_manager.initialize.assert_called_once()
        mock_client_instance.create_zmq_client.assert_called_once_with(model=pid, mode=unittest.mock.ANY)
        self.assertEqual(mock_client_instance.pid, pid)
        self.assertEqual(result, mock_client_instance)

    @patch("fastdeploy.entrypoints.openai.run_batch.OpenAIServingModels")
    @patch("fastdeploy.entrypoints.openai.run_batch.OpenAIServingChat")
    def test_create_serving_handlers(self, mock_chat_handler, mock_model_handler):
        """测试创建服务处理器"""
        mock_args = Mock()
        mock_args.max_model_len = 1024
        mock_args.ips = "127.0.0.1"
        mock_args.max_waiting_time = 60
        mock_args.enable_mm_output = False
        mock_args.tokenizer_base_url = None

        mock_engine_client = Mock()
        mock_model_paths = [Mock(spec=ModelPath)]
        chat_template = "test_template"
        pid = 123

        mock_model_instance = Mock()
        mock_model_handler.return_value = mock_model_instance

        mock_chat_instance = Mock()
        mock_chat_handler.return_value = mock_chat_instance

        result = create_serving_handlers(mock_args, mock_engine_client, mock_model_paths, chat_template, pid)

        # 验证处理器被正确创建
        mock_model_handler.assert_called_once_with(mock_model_paths, mock_args.max_model_len, mock_args.ips)
        mock_chat_handler.assert_called_once_with(
            mock_engine_client,
            mock_model_instance,
            pid,
            mock_args.ips,
            mock_args.max_waiting_time,
            chat_template,
            mock_args.enable_mm_output,
            mock_args.tokenizer_base_url,
        )
        self.assertEqual(result, mock_chat_instance)

    @patch("fastdeploy.entrypoints.openai.run_batch.determine_process_id")
    @patch("fastdeploy.entrypoints.openai.run_batch.create_model_paths")
    @patch("fastdeploy.entrypoints.openai.run_batch.load_chat_template")
    @patch("fastdeploy.entrypoints.openai.run_batch.initialize_engine_client")
    @patch("fastdeploy.entrypoints.openai.run_batch.create_serving_handlers")
    @patch("fastdeploy.entrypoints.openai.run_batch.console_logger")
    async def test_setup_engine_and_handlers(
        self,
        mock_logger,
        mock_create_handlers,
        mock_init_engine,
        mock_load_template,
        mock_create_paths,
        mock_determine_pid,
    ):
        """测试设置引擎和处理器"""
        mock_args = Mock()
        mock_args.tokenizer = None
        mock_args.model = "test-model"
        mock_args.chat_template = "template_name"

        # 设置mock返回值
        mock_determine_pid.return_value = 123
        mock_create_paths.return_value = [Mock(spec=ModelPath)]
        mock_load_template.return_value = "loaded_template"
        mock_engine_client = AsyncMock()
        mock_init_engine.return_value = mock_engine_client
        mock_chat_handler = Mock()
        mock_create_handlers.return_value = mock_chat_handler

        # 模拟全局llm_engine存在的情况
        mock_llm_engine = Mock()
        mock_llm_engine.engine = Mock()
        mock_llm_engine.engine.data_processor = None

        with patch("fastdeploy.entrypoints.openai.run_batch.llm_engine", mock_llm_engine):
            result = await setup_engine_and_handlers(mock_args)

        # 验证调用链
        mock_determine_pid.assert_called_once()
        mock_logger.info.assert_called_with("Process ID: 123")
        self.assertEqual(mock_args.tokenizer, "test-model")  # 验证tokenizer被设置
        mock_create_paths.assert_called_with(mock_args)
        mock_load_template.assert_called_with("template_name", "test-model")
        mock_init_engine.assert_called_with(mock_args, 123)
        mock_create_handlers.assert_called_with(
            mock_args, mock_engine_client, mock_create_paths.return_value, "loaded_template", 123
        )

        # 验证数据处理器被更新
        self.assertEqual(mock_llm_engine.engine.data_processor, mock_engine_client.data_processor)

        self.assertEqual(result, (mock_engine_client, mock_chat_handler))

    @patch("fastdeploy.entrypoints.openai.run_batch.determine_process_id")
    @patch("fastdeploy.entrypoints.openai.run_batch.create_model_paths")
    @patch("fastdeploy.entrypoints.openai.run_batch.load_chat_template")
    @patch("fastdeploy.entrypoints.openai.run_batch.initialize_engine_client")
    @patch("fastdeploy.entrypoints.openai.run_batch.create_serving_handlers")
    @patch("fastdeploy.entrypoints.openai.run_batch.console_logger")
    async def test_setup_engine_and_handlers_no_llm_engine(
        self,
        mock_logger,
        mock_create_handlers,
        mock_init_engine,
        mock_load_template,
        mock_create_paths,
        mock_determine_pid,
    ):
        """测试设置引擎和处理器（没有全局llm_engine的情况）"""
        mock_args = Mock()
        mock_args.tokenizer = None
        mock_args.model = "test-model"
        mock_args.chat_template = "template_name"

        # 设置mock返回值
        mock_determine_pid.return_value = 123
        mock_create_paths.return_value = [Mock(spec=ModelPath)]
        mock_load_template.return_value = "loaded_template"
        mock_engine_client = AsyncMock()
        mock_init_engine.return_value = mock_engine_client
        mock_chat_handler = Mock()
        mock_create_handlers.return_value = mock_chat_handler

        # 模拟全局llm_engine不存在的情况
        with patch("fastdeploy.entrypoints.openai.run_batch.llm_engine", None):
            result = await setup_engine_and_handlers(mock_args)

        # 验证调用链
        mock_determine_pid.assert_called_once()
        mock_logger.info.assert_called_with("Process ID: 123")
        self.assertEqual(mock_args.tokenizer, "test-model")
        mock_create_paths.assert_called_with(mock_args)
        mock_load_template.assert_called_with("template_name", "test-model")
        mock_init_engine.assert_called_with(mock_args, 123)
        mock_create_handlers.assert_called_with(
            mock_args, mock_engine_client, mock_create_paths.return_value, "loaded_template", 123
        )

        self.assertEqual(result, (mock_engine_client, mock_chat_handler))


class TestBatchProcessing(unittest.TestCase):
    """测试批处理相关函数"""

    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    @patch("fastdeploy.entrypoints.openai.run_batch.setup_engine_and_handlers")
    @patch("fastdeploy.entrypoints.openai.run_batch.read_file")
    @patch("fastdeploy.entrypoints.openai.run_batch.run_request")
    @patch("fastdeploy.entrypoints.openai.run_batch.make_async_error_request_output")
    @patch("fastdeploy.entrypoints.openai.run_batch.write_file")
    @patch("fastdeploy.entrypoints.openai.run_batch.console_logger")
    async def test_run_batch_success(
        self, mock_logger, mock_write_file, mock_make_error, mock_run_request, mock_read_file, mock_setup
    ):
        """测试成功运行批处理"""
        # 模拟参数
        mock_args = Mock()
        mock_args.input_file = "input.jsonl"
        mock_args.output_file = "output.jsonl"
        mock_args.output_tmp_dir = "/tmp"
        mock_args.max_concurrency = 512
        mock_args.workers = 2

        # 模拟设置返回
        mock_engine_client = Mock()
        mock_chat_handler = Mock()
        mock_chat_handler.create_chat_completion = Mock()
        mock_setup.return_value = (mock_engine_client, mock_chat_handler)

        # 模拟输入文件内容
        mock_read_file.return_value = (
            '{"url": "/v1/chat/completions", "custom_id": "1"}\n\n{"url": "/v1/chat/completions", "custom_id": "2"}'
        )

        # 模拟请求处理结果
        mock_response1 = Mock(error=None)
        mock_response2 = Mock(error=None)

        # 模拟异步操作
        future1 = asyncio.Future()
        future1.set_result(mock_response1)
        future2 = asyncio.Future()
        future2.set_result(mock_response2)

        mock_run_request.side_effect = [future1, future2]

        mock_make_error.return_value = asyncio.Future()
        mock_make_error.return_value.set_result(Mock())

        await run_batch(mock_args)

        # 验证日志记录
        mock_logger.info.assert_any_call("concurrency: 512, workers: 2, max_concurrency: 256")
        mock_logger.info.assert_any_call("Reading batch from input.jsonl...")
        mock_logger.info.assert_any_call("Batch processing completed: 2 success, 0 errors")

        # 验证文件写入
        mock_write_file.assert_called_once()

    @patch("fastdeploy.entrypoints.openai.run_batch.setup_engine_and_handlers")
    @patch("fastdeploy.entrypoints.openai.run_batch.read_file")
    @patch("fastdeploy.entrypoints.openai.run_batch.make_async_error_request_output")
    @patch("fastdeploy.entrypoints.openai.run_batch.write_file")
    @patch("fastdeploy.entrypoints.openai.run_batch.console_logger")
    async def test_run_batch_unsupported_endpoint(
        self, mock_logger, mock_write_file, mock_make_error, mock_read_file, mock_setup
    ):
        """测试不支持的端点"""
        mock_args = Mock()
        mock_args.input_file = "input.jsonl"
        mock_args.output_file = "output.jsonl"
        mock_args.output_tmp_dir = "/tmp"
        mock_args.max_concurrency = 512
        mock_args.workers = 1

        mock_setup.return_value = (Mock(), Mock())

        # 模拟不支持的URL
        mock_read_file.return_value = '{"url": "/v1/unsupported", "custom_id": "1"}'

        mock_make_error.return_value = asyncio.Future()
        mock_make_error.return_value.set_result(Mock())

        await run_batch(mock_args)

        # 验证错误处理被调用
        mock_make_error.assert_called_once()
        mock_logger.info.assert_any_call("Batch processing completed: 0 success, 1 errors")

    @patch("fastdeploy.entrypoints.openai.run_batch.setup_engine_and_handlers")
    @patch("fastdeploy.entrypoints.openai.run_batch.read_file")
    @patch("fastdeploy.entrypoints.openai.run_batch.make_async_error_request_output")
    @patch("fastdeploy.entrypoints.openai.run_batch.write_file")
    @patch("fastdeploy.entrypoints.openai.run_batch.console_logger")
    async def test_run_batch_no_chat_handler_for_chat_completions(
        self, mock_logger, mock_write_file, mock_make_error, mock_read_file, mock_setup
    ):
        """测试chat_handler为None时处理chat请求"""
        mock_args = Mock()
        mock_args.input_file = "input.jsonl"
        mock_args.output_file = "output.jsonl"
        mock_args.output_tmp_dir = "/tmp"
        mock_args.max_concurrency = 512
        mock_args.workers = 1

        # 返回None作为chat_handler
        mock_setup.return_value = (Mock(), None)

        mock_read_file.return_value = '{"url": "/v1/chat/completions", "custom_id": "1"}'

        mock_make_error.return_value = asyncio.Future()
        mock_error_output = Mock()
        mock_make_error.return_value.set_result(mock_error_output)

        await run_batch(mock_args)

        # 验证错误处理被调用
        mock_make_error.assert_called_once_with(
            unittest.mock.ANY, error_msg="The model does not support Chat Completions API"
        )
        mock_logger.info.assert_any_call("Batch processing completed: 0 success, 1 errors")

    @patch("fastdeploy.entrypoints.openai.run_batch.retrive_model_from_server")
    @patch("fastdeploy.entrypoints.openai.run_batch.ToolParserManager")
    @patch("fastdeploy.entrypoints.openai.run_batch.init_engine")
    @patch("fastdeploy.entrypoints.openai.run_batch.run_batch")
    @patch("fastdeploy.entrypoints.openai.run_batch.cleanup_resources")
    @patch("fastdeploy.entrypoints.openai.run_batch.console_logger")
    async def test_main_success(
        self, mock_logger, mock_cleanup, mock_run_batch, mock_init_engine, mock_tool_parser, mock_retrieve_model
    ):
        """测试主函数成功执行"""
        mock_args = Mock()
        mock_args.workers = None
        mock_args.max_num_seqs = 64
        mock_args.model = "test-model"
        mock_args.revision = "main"
        mock_args.tool_parser_plugin = None

        mock_retrieve_model.return_value = "retrieved-model"
        mock_init_engine.return_value = True

        await main(mock_args)

        # 验证参数处理
        self.assertEqual(mock_args.workers, 2)
        self.assertEqual(mock_args.model, "retrieved-model")
        mock_retrieve_model.assert_called_with("test-model", "main")
        mock_init_engine.assert_called_with(mock_args)
        mock_run_batch.assert_called_with(mock_args)
        mock_cleanup.assert_called_once()

    @patch("fastdeploy.entrypoints.openai.run_batch.retrive_model_from_server")
    @patch("fastdeploy.entrypoints.openai.run_batch.ToolParserManager")
    @patch("fastdeploy.entrypoints.openai.run_batch.init_engine")
    @patch("fastdeploy.entrypoints.openai.run_batch.run_batch")
    @patch("fastdeploy.entrypoints.openai.run_batch.cleanup_resources")
    @patch("fastdeploy.entrypoints.openai.run_batch.console_logger")
    async def test_main_with_tool_parser_plugin(
        self, mock_logger, mock_cleanup, mock_run_batch, mock_init_engine, mock_tool_parser, mock_retrieve_model
    ):
        """测试主函数使用tool_parser_plugin"""
        mock_args = Mock()
        mock_args.workers = 1
        mock_args.max_num_seqs = 32
        mock_args.model = "test-model"
        mock_args.revision = "main"
        mock_args.tool_parser_plugin = "test_plugin"

        mock_retrieve_model.return_value = "retrieved-model"
        mock_init_engine.return_value = True

        await main(mock_args)

        # 验证工具解析器插件被导入
        mock_tool_parser.import_tool_parser.assert_called_once_with("test_plugin")
        mock_init_engine.assert_called_with(mock_args)
        mock_run_batch.assert_called_with(mock_args)
        mock_cleanup.assert_called_once()

    @patch("fastdeploy.entrypoints.openai.run_batch.retrive_model_from_server")
    @patch("fastdeploy.entrypoints.openai.run_batch.init_engine")
    @patch("fastdeploy.entrypoints.openai.run_batch.cleanup_resources")
    @patch("fastdeploy.entrypoints.openai.run_batch.console_logger")
    async def test_main_init_engine_fails(self, mock_logger, mock_cleanup, mock_init_engine, mock_retrieve_model):
        """测试初始化引擎失败的情况"""
        mock_args = Mock()
        mock_args.workers = None
        mock_args.max_num_seqs = 64
        mock_args.model = "test-model"
        mock_args.revision = "main"
        mock_args.tool_parser_plugin = None

        mock_retrieve_model.return_value = "retrieved-model"
        mock_init_engine.return_value = False  # 初始化失败

        await main(mock_args)

        # 验证没有运行批处理
        mock_init_engine.assert_called_with(mock_args)
        mock_cleanup.assert_called_once()

    @patch("fastdeploy.entrypoints.openai.run_batch.console_logger")
    async def test_cleanup_resources_success(self, mock_logger):
        """测试资源清理成功"""
        # 模拟全局变量
        with (
            patch("fastdeploy.entrypoints.openai.run_batch.llm_engine", None),
            patch("fastdeploy.entrypoints.openai.run_batch.engine_client", None),
        ):
            await cleanup_resources()

            # 验证日志记录
            mock_logger.error.assert_not_called()

    @patch("fastdeploy.entrypoints.openai.run_batch.console_logger")
    async def test_cleanup_resources_with_errors(self, mock_logger):
        """测试资源清理时出现错误"""
        # 模拟有问题的引擎和客户端
        mock_engine = Mock()
        mock_engine._exit_sub_services = Mock(side_effect=Exception("Engine error"))

        mock_client = Mock()
        mock_client.zmq_client = Mock()
        mock_client.zmq_client.close = Mock(side_effect=Exception("ZMQ error"))
        mock_client.connection_manager = AsyncMock()
        mock_client.connection_manager.close = AsyncMock(side_effect=Exception("Connection error"))

        with (
            patch("fastdeploy.entrypoints.openai.run_batch.llm_engine", mock_engine),
            patch("fastdeploy.entrypoints.openai.run_batch.engine_client", mock_client),
        ):
            await cleanup_resources()

            # 验证错误被记录但不会抛出
            self.assertEqual(mock_logger.error.call_count, 3)

    @patch("fastdeploy.entrypoints.openai.run_batch.console_logger")
    async def test_cleanup_resources_partial_errors(self, mock_logger):
        """测试资源清理时部分组件出错"""
        # 模拟只有引擎有问题的情况
        mock_engine = Mock()
        mock_engine._exit_sub_services = Mock(side_effect=Exception("Engine error"))

        with (
            patch("fastdeploy.entrypoints.openai.run_batch.llm_engine", mock_engine),
            patch("fastdeploy.entrypoints.openai.run_batch.engine_client", None),
        ):
            await cleanup_resources()

            # 验证只有引擎错误被记录
            mock_logger.error.assert_called_once()
            mock_logger.error.assert_called_with("Error stopping engine: Engine error")

    @patch("fastdeploy.entrypoints.openai.run_batch.console_logger")
    @patch("gc.collect")
    async def test_cleanup_resources_with_gc(self, mock_gc, mock_logger):
        """测试资源清理包括垃圾回收"""
        # 模拟有引擎和客户端的情况
        mock_engine = Mock()
        mock_engine._exit_sub_services = Mock()

        mock_client = Mock()
        mock_client.zmq_client = Mock()
        mock_client.zmq_client.close = Mock()
        mock_client.connection_manager = AsyncMock()
        mock_client.connection_manager.close = AsyncMock()

        with (
            patch("fastdeploy.entrypoints.openai.run_batch.llm_engine", mock_engine),
            patch("fastdeploy.entrypoints.openai.run_batch.engine_client", mock_client),
        ):
            await cleanup_resources()

            # 验证垃圾回收被调用
            mock_gc.assert_called_once()
            mock_logger.error.assert_not_called()


class TestRunRequest(unittest.TestCase):
    """测试run_request函数"""

    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    @patch("fastdeploy.entrypoints.openai.run_batch.random_uuid")
    @patch("fastdeploy.entrypoints.openai.run_batch.console_logger")
    async def test_run_request_success_chat_completion(self, mock_logger, mock_random_uuid):
        """测试成功返回ChatCompletionResponse的情况"""
        mock_random_uuid.side_effect = ["id1", "req1"]

        # 模拟成功的响应
        mock_response = Mock(spec=ChatCompletionResponse)
        mock_engine = AsyncMock(return_value=mock_response)
        mock_request = Mock()
        mock_request.custom_id = "test-id"
        mock_request.body = "test-body"
        mock_tracker = Mock()
        mock_semaphore = AsyncMock()

        result = await run_request(mock_engine, mock_request, mock_tracker, mock_semaphore)

        # 验证结果
        self.assertEqual(result.custom_id, "test-id")
        self.assertEqual(result.response.status_code, 200)
        self.assertEqual(result.response.body, mock_response)
        self.assertIsNone(result.error)
        mock_tracker.completed.assert_called_once()

    @patch("fastdeploy.entrypoints.openai.run_batch.random_uuid")
    @patch("fastdeploy.entrypoints.openai.run_batch.console_logger")
    async def test_run_request_error_response(self, mock_logger, mock_random_uuid):
        """测试返回ErrorResponse的情况"""
        mock_random_uuid.side_effect = ["id2", "req2"]

        # 模拟错误响应
        mock_error = Mock(spec=ErrorResponse)
        mock_engine = AsyncMock(return_value=mock_error)
        mock_request = Mock()
        mock_request.custom_id = "error-id"
        mock_tracker = Mock()
        mock_semaphore = AsyncMock()

        result = await run_request(mock_engine, mock_request, mock_tracker, mock_semaphore)

        # 验证错误结果
        self.assertEqual(result.response.status_code, 400)
        self.assertEqual(result.error, mock_error)
        mock_tracker.completed.assert_called_once()

    @patch("fastdeploy.entrypoints.openai.run_batch.make_error_request_output")
    @patch("fastdeploy.entrypoints.openai.run_batch.console_logger")
    async def test_run_request_stream_mode_error(self, mock_logger, mock_make_error):
        """测试流模式错误情况"""
        # 模拟非ChatCompletionResponse和ErrorResponse的响应
        mock_engine = AsyncMock(return_value="invalid_response")
        mock_request = Mock()
        mock_tracker = Mock()
        mock_semaphore = AsyncMock()
        mock_error_output = Mock()
        mock_make_error.return_value = mock_error_output

        result = await run_request(mock_engine, mock_request, mock_tracker, mock_semaphore)

        # 验证调用了错误处理函数
        mock_make_error.assert_called_once_with(mock_request, "Request must not be sent in stream mode")
        self.assertEqual(result, mock_error_output)
        mock_tracker.completed.assert_called_once()

    @patch("fastdeploy.entrypoints.openai.run_batch.make_error_request_output")
    @patch("fastdeploy.entrypoints.openai.run_batch.console_logger")
    async def test_run_request_exception(self, mock_logger, mock_make_error):
        """测试异常情况"""
        # 模拟抛出异常
        mock_engine = AsyncMock(side_effect=Exception("Test error"))
        mock_request = Mock()
        mock_request.custom_id = "exception-id"
        mock_tracker = Mock()
        mock_semaphore = AsyncMock()
        mock_error_output = Mock()
        mock_make_error.return_value = mock_error_output

        result = await run_request(mock_engine, mock_request, mock_tracker, mock_semaphore)

        # 验证错误日志和错误处理
        mock_logger.error.assert_called_once()
        mock_make_error.assert_called_once_with(mock_request, "Request processing failed: Test error")
        self.assertEqual(result, mock_error_output)
        mock_tracker.completed.assert_called_once()


class TestDetermineProcessId(unittest.TestCase):
    """测试determine_process_id函数"""

    @patch("multiprocessing.current_process")
    @patch("os.getppid")
    @patch("os.getpid")
    def test_determine_process_id_main_process(self, mock_getpid, mock_getppid, mock_current_process):
        """测试主进程情况"""
        mock_current_process.return_value.name = "MainProcess"
        mock_getpid.return_value = 123

        result = determine_process_id()

        self.assertEqual(result, 123)
        mock_getpid.assert_called_once()
        mock_getppid.assert_not_called()

    @patch("multiprocessing.current_process")
    @patch("os.getppid")
    @patch("os.getpid")
    def test_determine_process_id_child_process(self, mock_getpid, mock_getppid, mock_current_process):
        """测试子进程情况"""
        mock_current_process.return_value.name = "Process-1"
        mock_getppid.return_value = 456

        determine_process_id()

        mock_getpid.assert_called_once()


class TestCreateModelPaths(unittest.TestCase):
    """测试create_model_paths函数"""

    def test_create_model_paths_with_served_model_name(self):
        """测试提供served_model_name的情况"""
        mock_args = Mock()
        mock_args.served_model_name = "custom-model-name"
        mock_args.model = "path/to/model"

        result = create_model_paths(mock_args)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "custom-model-name")
        self.assertEqual(result[0].model_path, "path/to/model")
        self.assertTrue(result[0].verification)

    def test_create_model_paths_without_served_model_name(self):
        """测试不提供served_model_name的情况"""
        mock_args = Mock()
        mock_args.served_model_name = None
        mock_args.model = "path/to/model"

        result = create_model_paths(mock_args)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "path/to/model")
        self.assertEqual(result[0].model_path, "path/to/model")
        self.assertFalse(result[0].verification)


class TestErrorRequestOutput(unittest.TestCase):
    """测试错误请求输出生成函数"""

    def setUp(self):
        # 设置异步测试循环
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    @patch("fastdeploy.entrypoints.openai.run_batch.random_uuid")
    def test_make_error_request_output_basic(self, mock_random_uuid):
        """测试基本功能"""
        mock_random_uuid.side_effect = ["req123", "batch456"]

        mock_request = Mock()
        mock_request.custom_id = "test-id"

        result = make_error_request_output(mock_request, "Test error")

        # 验证基本属性
        self.assertEqual(result.id, "fastdeploy-req123")
        self.assertEqual(result.custom_id, "test-id")
        self.assertEqual(result.error, "Test error")
        self.assertEqual(result.response.status_code, HTTPStatus.BAD_REQUEST)
        self.assertEqual(result.response.request_id, "fastdeploy-batch-batch456")

    @patch("fastdeploy.entrypoints.openai.run_batch.make_error_request_output")
    async def test_make_async_error_request_output(self, mock_make_error):
        """测试异步版本"""
        expected_output = Mock()
        mock_make_error.return_value = expected_output

        mock_request = Mock()
        mock_request.custom_id = "async-test"

        result = await make_async_error_request_output(mock_request, "Async error")

        self.assertEqual(result, expected_output)
        mock_make_error.assert_called_once_with(mock_request, "Async error")


class TestFileOperations(unittest.TestCase):
    """测试文件操作相关函数"""

    def setUp(self):
        # 设置异步测试循环
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    @patch("aiohttp.ClientSession")
    async def test_read_file_http(self, mock_session):
        """测试从HTTP URL读取文件"""
        # 模拟响应
        mock_resp = AsyncMock()
        mock_resp.text = AsyncMock(return_value="HTTP content")
        mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_resp

        result = await read_file("https://example.com/file.txt")

        self.assertEqual(result, "HTTP content")
        mock_session.assert_called_once()

    def create_batch_outputs_from_jsonl(self, jsonl_text):
        """从 JSONL 文本创建 BatchRequestOutput 对象列表"""
        batch_outputs = []
        lines = jsonl_text.strip().split("\n")

        for line in lines:
            if line.strip():
                data = json.loads(line)

                # 解析 response 部分
                response_data = data["response"]
                body_data = response_data["body"]

                # 创建 ChatMessage 对象
                message_data = body_data["choices"][0]["message"]
                chat_message = ChatMessage(
                    role=message_data["role"],
                    content=message_data["content"],
                    multimodal_content=message_data["multimodal_content"],
                    reasoning_content=message_data["reasoning_content"],
                    tool_calls=message_data["tool_calls"],
                    prompt_token_ids=message_data["prompt_token_ids"],
                    completion_token_ids=message_data["completion_token_ids"],
                    prompt_tokens=message_data["prompt_tokens"],
                    completion_tokens=message_data["completion_tokens"],
                )

                # 创建 ChatCompletionResponseChoice 对象
                choice_data = body_data["choices"][0]
                choice = ChatCompletionResponseChoice(
                    index=choice_data["index"],
                    message=chat_message,
                    logprobs=choice_data["logprobs"],
                    finish_reason=choice_data["finish_reason"],
                )

                # 创建 UsageInfo 对象
                usage_data = body_data["usage"]
                usage_info = UsageInfo(
                    prompt_tokens=usage_data["prompt_tokens"],
                    total_tokens=usage_data["total_tokens"],
                    completion_tokens=usage_data["completion_tokens"],
                    prompt_tokens_details=usage_data.get("prompt_tokens_details"),
                )

                # 创建 ChatCompletionResponse 对象
                chat_completion_response = ChatCompletionResponse(
                    id=body_data["id"],
                    object=body_data["object"],
                    created=body_data["created"],
                    model=body_data["model"],
                    choices=[choice],
                    usage=usage_info,
                )

                # 创建 BatchResponseData 对象
                batch_response_data = BatchResponseData(
                    status_code=response_data["status_code"],
                    request_id=response_data["request_id"],
                    body=chat_completion_response,
                )

                # 创建 BatchRequestOutput 对象
                batch_output = BatchRequestOutput(
                    id=data["id"], custom_id=data["custom_id"], response=batch_response_data, error=data["error"]
                )
                batch_outputs.append(batch_output)

        return batch_outputs

    def test_write_local_file_basic(self):
        """测试基础功能：写入文件并验证内容"""
        # 创建测试数据
        batch_outputs = self.create_batch_outputs_from_jsonl(BATCH_RESPONSE)

        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as temp_file:
            temp_path = temp_file.name

        try:
            # 异步调用被测函数
            async def run_test():
                await write_local_file(temp_path, batch_outputs)

            self.loop.run_until_complete(run_test())

            # 验证文件存在
            self.assertTrue(os.path.exists(temp_path))

            # 验证文件不为空
            self.assertGreater(os.path.getsize(temp_path), 0)

            # 读取并验证文件内容
            with open(temp_path, "r", encoding="utf-8") as f:
                written_lines = f.read().strip().split("\n")

            # 验证行数匹配
            self.assertEqual(len(written_lines), 2)

            # 验证每行都是有效的 JSON
            for i, line in enumerate(written_lines):
                data = json.loads(line)
                self.assertIn("id", data)
                self.assertIn("custom_id", data)
                self.assertIn("response", data)
                self.assertIn("error", data)

                # 验证关键字段
                self.assertEqual(data["custom_id"], f"req-0000{i+1}")
                self.assertEqual(data["response"]["status_code"], 200)
                self.assertIn("body", data["response"])
                self.assertIn("choices", data["response"]["body"])

            print("✓ 基础功能测试通过")

        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_write_local_file_content_integrity(self):
        """测试内容完整性：验证写入的内容与原始数据一致"""
        # 创建测试数据
        batch_outputs = self.create_batch_outputs_from_jsonl(BATCH_RESPONSE)

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as temp_file:
            temp_path = temp_file.name

        try:
            # 异步调用被测函数
            async def run_test():
                await write_local_file(temp_path, batch_outputs)

            self.loop.run_until_complete(run_test())

            # 读取写入的文件内容
            with open(temp_path, "r", encoding="utf-8") as f:
                written_content = f.read().strip()

            # 解析原始数据
            original_lines = BATCH_RESPONSE.strip().split("\n")
            written_lines = written_content.split("\n")

            # 验证行数一致
            self.assertEqual(len(original_lines), len(written_lines))

            # 验证每行的关键字段一致
            for i, (orig_line, written_line) in enumerate(zip(original_lines, written_lines)):
                orig_data = json.loads(orig_line)
                written_data = json.loads(written_line)

                # 比较关键标识字段
                self.assertEqual(orig_data["id"], written_data["id"])
                self.assertEqual(orig_data["custom_id"], written_data["custom_id"])
                self.assertEqual(orig_data["response"]["status_code"], written_data["response"]["status_code"])

                # 比较响应内容
                orig_content = orig_data["response"]["body"]["choices"][0]["message"]["content"]
                written_content = written_data["response"]["body"]["choices"][0]["message"]["content"]
                # 内容应该一致
                self.assertEqual(orig_content, written_content)

            print("✓ 内容完整性测试通过")

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_write_local_file_empty_list(self):
        """测试空列表处理"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as temp_file:
            temp_path = temp_file.name

        try:
            # 异步调用函数写入空列表
            async def run_test():
                await write_local_file(temp_path, [])

            self.loop.run_until_complete(run_test())

            # 验证文件存在但为空
            self.assertTrue(os.path.exists(temp_path))

            with open(temp_path, "r", encoding="utf-8") as f:
                content = f.read()

            self.assertEqual(content, "")
            print("✓ 空列表处理测试通过")

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    @patch("builtins.open", new_callable=mock_open, read_data="Local content")
    async def test_read_file_local(self, mock_file):
        """测试从本地文件读取"""
        result = await read_file("/local/path/file.txt")

        self.assertEqual(result, "Local content")
        mock_file.assert_called_once_with("/local/path/file.txt", encoding="utf-8")

    @patch("builtins.open", new_callable=mock_open)
    async def test_write_local_file(self, mock_file):
        """测试写入本地文件"""
        # 创建模拟的batch outputs
        mock_outputs = [
            Mock(spec=BatchRequestOutput, model_dump_json=Mock(return_value='{"id": 1}')),
            Mock(spec=BatchRequestOutput, model_dump_json=Mock(return_value='{"id": 2}')),
        ]

        await write_local_file("/output/path.json", mock_outputs)

        mock_file.assert_called_once_with("/output/path.json", "w", encoding="utf-8")

        # 检查写入调用
        handle = mock_file()
        expected_calls = [unittest.mock.call.write('{"id": 1}\n'), unittest.mock.call.write('{"id": 2}\n')]
        handle.write.assert_has_calls(expected_calls)

    @patch("aiohttp.ClientSession")
    async def test_upload_data_success(self, mock_session):
        """测试成功上传数据"""
        mock_resp = Mock(status=200, text=Mock(return_value="OK"))
        mock_session.return_value.__aenter__.return_value.put.return_value.__aenter__.return_value = mock_resp

        # 测试从文件上传
        with patch("builtins.open", mock_open(read_data=b"file content")):
            await upload_data("https://example.com/upload", "/path/to/file", from_file=True)

        # 测试直接上传数据
        await upload_data("https://example.com/upload", "raw data", from_file=False)

        self.assertEqual(mock_session.call_count, 2)

    @patch("aiohttp.ClientSession")
    @patch("asyncio.sleep", new_callable=AsyncMock)
    async def test_upload_data_retry(self, mock_sleep, mock_session):
        """测试上传失败重试逻辑"""
        # 模拟前两次失败，第三次成功
        mock_resp_fail = Mock(status=500, text=Mock(return_value="Server Error"))
        mock_resp_success = Mock(status=200, text=Mock(return_value="OK"))

        mock_session.return_value.__aenter__.return_value.put.side_effect = [
            Exception("First failure"),
            mock_resp_fail,
            mock_resp_success,
        ]

        # 这次应该成功，经过两次重试
        with patch("builtins.open", mock_open(read_data=b"content")):
            await upload_data("https://example.com/upload", "/path/to/file", from_file=True)

        # 检查重试次数
        self.assertEqual(mock_sleep.call_count, 2)
        self.assertEqual(mock_session.return_value.__aenter__.return_value.put.call_count, 3)

    @patch("aiohttp.ClientSession")
    async def test_upload_data_failure(self, mock_session):
        """测试上传最终失败"""
        mock_session.return_value.__aenter__.return_value.put.side_effect = Exception("Persistent failure")

        with patch("builtins.open", mock_open(read_data=b"content")):
            with self.assertRaises(Exception) as context:
                await upload_data("https://example.com/upload", "/path/to/file", from_file=True)

        self.assertIn("Failed to upload data", str(context.exception))

    @patch("fastdeploy.entrypoints.openai.run_batch.upload_data")
    @patch("fastdeploy.entrypoints.openai.run_batch.write_local_file")
    async def test_write_file_http_with_buffer(self, mock_write_local, mock_upload):
        """测试HTTP输出写入到内存缓冲区"""
        mock_outputs = [Mock(spec=BatchRequestOutput)]

        await write_file("https://example.com/output", mock_outputs, output_tmp_dir=None)

        # 应该调用upload_data，而不是write_local_file
        mock_upload.assert_called_once()
        mock_write_local.assert_not_called()

    @patch("fastdeploy.entrypoints.openai.run_batch.upload_data")
    @patch("tempfile.NamedTemporaryFile")
    @patch("fastdeploy.entrypoints.openai.run_batch.write_local_file")
    async def test_write_file_http_with_tempfile(self, mock_write_local, mock_tempfile, mock_upload):
        """测试HTTP输出写入到临时文件"""
        # 模拟临时文件
        mock_file = Mock()
        mock_file.name = "/tmp/tempfile.json"
        mock_tempfile.return_value.__enter__.return_value = mock_file

        mock_outputs = [Mock(spec=BatchRequestOutput)]

        await write_file("https://example.com/output", mock_outputs, output_tmp_dir="/tmp")

        mock_tempfile.assert_called_once()
        mock_write_local.assert_called_once_with(mock_file.name, mock_outputs)
        mock_upload.assert_called_once_with("https://example.com/output", mock_file.name, from_file=True)

    @patch("fastdeploy.entrypoints.openai.run_batch.write_local_file")
    async def test_write_file_local(self, mock_write_local):
        """测试本地文件输出"""
        mock_outputs = [Mock(spec=BatchRequestOutput)]

        await write_file("/local/output.json", mock_outputs, output_tmp_dir="/tmp")

        mock_write_local.assert_called_once_with("/local/output.json", mock_outputs)


class TestUtilityFunctions(unittest.TestCase):
    """测试工具函数"""

    def test_random_uuid(self):
        """测试生成随机UUID"""
        uuid1 = random_uuid()
        uuid2 = random_uuid()

        self.assertEqual(len(uuid1), 32)
        self.assertTrue(all(c in "0123456789abcdef" for c in uuid1))

        self.assertNotEqual(uuid1, uuid2)


class TestBatchProgressTracker(unittest.TestCase):

    def test_submitted_increments_total(self):
        tracker = BatchProgressTracker()
        self.assertEqual(tracker._total, 0)
        tracker.submitted()
        self.assertEqual(tracker._total, 1)
        tracker.submitted()
        self.assertEqual(tracker._total, 2)

    @patch("fastdeploy.entrypoints.openai.run_batch.console_logger")
    def test_completed_increments_completed_and_logs(self, mock_logger):
        tracker = BatchProgressTracker()
        tracker._total = 20

        # 调用 10 次 -> 应该触发一次日志 (log_interval=2)
        for _ in range(10):
            tracker.completed()

        self.assertEqual(tracker._completed, 10)
        mock_logger.info.assert_called()  # 至少被调用一次
        args, _ = mock_logger.info.call_args
        self.assertIn("Progress: 10/20", args[0])

    @patch("fastdeploy.entrypoints.openai.run_batch.tqdm")
    def test_completed_updates_pbar(self, mock_tqdm):
        mock_pbar = MagicMock()
        mock_tqdm.return_value = mock_pbar

        tracker = BatchProgressTracker()
        tracker._total = 5
        tracker.pbar()  # 初始化 pbar

        tracker.completed()
        mock_pbar.update.assert_called_once()

    @patch("fastdeploy.entrypoints.openai.run_batch.tqdm")
    def test_pbar_returns_tqdm(self, mock_tqdm):
        mock_pbar = MagicMock(spec=tqdm)
        mock_tqdm.return_value = mock_pbar

        tracker = BatchProgressTracker()
        tracker._total = 3
        result = tracker.pbar()

        self.assertIs(result, mock_pbar)
        mock_tqdm.assert_called_once_with(
            total=3,
            unit="req",
            desc="Running batch",
            mininterval=10,
            bar_format=_BAR_FORMAT,
        )


class TestBatchProgressTrackerExtended(unittest.TestCase):
    """扩展的BatchProgressTracker测试"""

    @patch("fastdeploy.entrypoints.openai.run_batch.console_logger")
    def test_completed_with_pbar_no_log(self, mock_logger):
        """测试有进度条时的completed方法，不触发日志记录"""
        tracker = BatchProgressTracker()
        tracker._total = 100  # 设置较大的总数，使得第一次完成不会触发日志
        tracker._pbar = Mock()

        tracker.completed()  # 完成1个，1/100=1%，不会触发日志记录

        tracker._pbar.update.assert_called_once()
        mock_logger.info.assert_not_called()  # 不应该记录日志

    @patch("fastdeploy.entrypoints.openai.run_batch.console_logger")
    def test_completed_log_interval(self, mock_logger):
        """测试日志间隔"""
        tracker = BatchProgressTracker()
        tracker._total = 100
        tracker._last_log_count = 0

        # 触发日志记录（每10个记录一次）
        for i in range(1, 21):
            tracker.completed()
            if i % 10 == 0:
                mock_logger.info.assert_called_with(f"Progress: {i}/100 requests completed")


FD_ENGINE_QUEUE_PORT = int(os.getenv("FD_ENGINE_QUEUE_PORT", 8133))
FD_CACHE_QUEUE_PORT = int(os.getenv("FD_CACHE_QUEUE_PORT", 8333))


class TestFastDeployBatch(unittest.TestCase):
    """测试 FastDeploy 批处理功能的 unittest 测试类"""

    def setUp(self):
        """每个测试方法执行前的准备工作"""
        self.model_path = "baidu/ERNIE-4.5-0.3B-PT"
        self.base_command = ["fastdeploy", "run-batch"]
        self.run_batch_command = ["python", "fastdeploy/entrypoints/openai/run_batch.py"]

    def run_fastdeploy_command(self, input_content, port=None):
        """运行 FastDeploy 命令的辅助方法"""
        if port is None:
            port = str(FD_CACHE_QUEUE_PORT)

        with tempfile.NamedTemporaryFile("w") as input_file, tempfile.NamedTemporaryFile("r") as output_file:

            input_file.write(input_content)
            input_file.flush()

            param = [
                "-i",
                input_file.name,
                "-o",
                output_file.name,
                "--model",
                self.model_path,
                "--cache-queue-port",
                port,
                "--tensor-parallel-size",
                "1",
                "--quantization",
                "wint4",
                "--max-model-len",
                "5120",
                "--max-num-seqs",
                "64",
                "--load-choices",
                "default_v1",
                "--engine-worker-queue-port",
                str(FD_ENGINE_QUEUE_PORT),
            ]

            # command = self.base_command + param
            run_batch_command = self.run_batch_command + param

            proc = subprocess.Popen(run_batch_command)
            proc.communicate()
            return_code = proc.wait()

            # 读取输出文件内容
            output_file.seek(0)
            contents = output_file.read()

            return return_code, contents, proc

    def test_completions(self):
        """测试正常的批量chat请求"""
        return_code, contents, proc = self.run_fastdeploy_command(INPUT_BATCH, port="2235")

        self.assertEqual(return_code, 0, f"进程返回非零码: {return_code}, 进程信息: {proc}")

        # 验证每行输出都符合 OpenAI API 格式
        lines = contents.strip().split("\n")
        for line in lines:
            if line:  # 跳过空行
                # 验证应该抛出异常如果 schema 错误
                try:
                    BatchRequestOutput.model_validate_json(line)
                except Exception as e:
                    self.fail(f"输出格式验证失败: {e}\n行内容: {line}")

    def test_vaild_input(self):
        """测试输入数据格式的正确性"""
        return_code, contents, proc = self.run_fastdeploy_command(INVALID_INPUT_BATCH)

        self.assertNotEqual(return_code, 0, f"进程返回非零码: {return_code}, 进程信息: {proc}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
