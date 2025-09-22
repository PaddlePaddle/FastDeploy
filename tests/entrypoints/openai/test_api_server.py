import os
import socket
import unittest
from unittest import mock
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from fastdeploy.entrypoints.openai import api_server
from fastdeploy.entrypoints.openai.api_server import (
    ApiServerApp,
    ControllerServerApp,
    MetricsServerApp,
    run_api_server_worker_proc,
)
from fastdeploy.entrypoints.openai.protocol import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    CompletionResponse,
    CompletionResponseChoice,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
    DeltaMessage,
    ModelInfo,
    ModelList,
    UsageInfo,
)

# 直接从本地模块导入
make_arg_parser = api_server.make_arg_parser
rewrite_args = api_server.rewrite_args
load_engine = api_server.load_engine
load_data_service = api_server.load_data_service


class TestApiServer(unittest.TestCase):

    def setUp(self):
        self.test_args = MagicMock()
        self.test_args.port = 123123
        self.test_args.host = "0.0.0.0"
        self.test_args.workers = 1
        self.test_args.metrics_port = 12334
        self.test_args.controller_port = 12231
        self.test_args.max_waiting_time = -1
        self.test_args.max_concurrency = 512
        self.test_args.enable_mm_output = False
        self.test_args.timeout_graceful_shutdown = 0
        self.test_args.model = "test_model"
        self.test_args.revision = None
        self.test_args.tokenizer = None
        self.test_args.max_model_len = 2048
        self.test_args.tensor_parallel_size = 1
        self.test_args.engine_worker_queue_port = [8002]
        self.test_args.local_data_parallel_id = 0
        self.test_args.limit_mm_per_prompt = None
        self.test_args.mm_processor_kwargs = None
        self.test_args.reasoning_parser = None
        self.test_args.data_parallel_size = 1
        self.test_args.enable_logprob = False
        self.test_args.tool_call_parser = None
        self.test_args.dynamic_load_weight = False
        self.test_args.served_model_name = None
        self.test_args.hidden_size = 11
        self.test_args.num_attention_heads = 11
        self.test_args.chat_template = None
        self.test_args.ips = "127.0.0.1"
        self.test_args.tokenizer_base_url = None

    @patch("fastdeploy.entrypoints.openai.api_server.LLMEngine")
    def test_load_engine(self, mock_engine):
        mock_engine_instance = MagicMock()
        mock_engine.from_engine_args.return_value = mock_engine_instance
        mock_engine_instance.start.return_value = True

        with patch("fastdeploy.entrypoints.openai.api_server.llm_engine", None):
            result = load_engine(self.test_args)
            self.assertEqual(result, mock_engine_instance)
            mock_engine.from_engine_args.assert_called_once()
            mock_engine_instance.start.assert_called_once_with(api_server_pid=os.getpid())

    @patch("fastdeploy.entrypoints.openai.api_server.ExpertService")
    @patch("fastdeploy.engine.args_utils.EngineArgs.from_cli_args")
    @patch("fastdeploy.engine.args_utils.EngineArgs.create_engine_config")
    @patch("fastdeploy.entrypoints.openai.api_server.os.getpid")
    @patch("fastdeploy.entrypoints.openai.api_server.api_server_logger.info")
    def test_load_data_service(self, mock_logger, mock_getpid, mock_create_config, mock_from_cli, mock_service):
        """测试 load_data_service 函数的完整行为"""
        # Setup mocks
        mock_getpid.return_value = 12345

        # 创建详细的配置对象
        config = MagicMock()
        config.parallel_config.local_data_parallel_id = 0
        config.hidden_size = 768
        config.num_attention_heads = 12
        config.worker_num_per_node = 1
        config.nnode = 1
        config.parallel_config.data_parallel_size = 1
        config.parallel_config.tensor_parallel_size = 1
        config.splitwise_role = "mixed"
        config.scheduler_config = MagicMock(name="default")
        config.cache_config = MagicMock(rdma_comm_ports=[], pd_comm_port=[8000])
        config.device_ids = "0"
        config.engine_worker_queue_port = [8000]
        config.host_ip = "127.0.0.1"
        config.disaggregate_info = None
        config.print = MagicMock()

        engine_args = MagicMock()
        engine_args.create_engine_config.return_value = config
        mock_from_cli.return_value = engine_args

        mock_service_instance = MagicMock()
        mock_service.return_value = mock_service_instance
        mock_service_instance.start.return_value = True

        # 调用函数
        result = load_data_service(self.test_args)
        # 验证点1: EngineArgs.from_cli_args 被正确调用
        mock_from_cli.assert_called_once_with(self.test_args)
        # 验证点2: create_engine_config 被正确调用
        engine_args.create_engine_config.assert_called_once()
        # 验证点3: ExpertService 被正确初始化
        mock_service.assert_called_once_with(config, 0)
        # 验证点4: start 方法被正确调用
        mock_service_instance.start.assert_called_once_with(12345, 0)
        # 验证点5: 函数返回预期的 ExpertService 实例
        self.assertEqual(result, mock_service_instance)
        # 验证日志记录
        mock_logger.assert_called()

    def test_make_arg_parser(self):
        parser = make_arg_parser(MagicMock())
        self.assertTrue(hasattr(parser, "add_argument"))

    def test_rewrite_args(self):
        self.test_args.workers = None
        self.test_args.max_num_seqs = 64
        self.test_args.tool_parser_plugin = None
        self.test_args.model = "test_model"
        with patch("fastdeploy.entrypoints.openai.api_server.retrive_model_from_server", return_value="test_model"):
            result = rewrite_args(self.test_args)
            self.assertEqual(result.workers, None)  # 64 // 32 = 2

    @patch("multiprocessing.get_context")
    @patch("multiprocessing.connection.wait")
    @patch("fastdeploy.utils.kill_process_tree")  # 将mock提升到方法级别
    def test_run_multi_api_server(self, mock_kill_process_tree, mock_ready_sentinels, mock_spawn_context):
        mocked_spawn_context = MagicMock()
        mocked_process = MagicMock()
        mocked_process.sentinel = "test_sentinel"
        mocked_process.exitcode = 1
        mocked_process.name = "test_process"
        mocked_process.is_alive.return_value = False
        mocked_process.pid = 1
        mocked_spawn_context.Process.return_value = mocked_process
        mock_spawn_context.return_value = mocked_spawn_context
        mock_sentinels = ["test_sentinel"]
        mock_ready_sentinels.return_value = mock_sentinels
        with (
            patch("fastdeploy.entrypoints.openai.api_server.set_ulimit"),
            patch("fastdeploy.entrypoints.openai.api_server.create_server_socket"),
            patch("fastdeploy.entrypoints.openai.api_server.signal.SIGKILL"),
        ):
            with self.assertRaises(RuntimeError):
                api_server.run_multi_api_server(self.test_args)

    @patch("fastdeploy.entrypoints.openai.api_server.LLMEngine")
    @patch("uvicorn.run")
    @patch("socket.socket")
    @patch("fastdeploy.entrypoints.openai.api_server.retrive_model_from_server", return_value="test_model")
    def test_api_server_app(self, mock_retrieve_model, mock_socket, mock_run, mock_engine):
        mock_engine_instance = MagicMock()
        mock_engine.from_engine_args.return_value = mock_engine_instance
        mock_engine_instance.start.return_value = True

        app = ApiServerApp(self.test_args)
        TestClient(app.build_app())
        app.launch_api_server()
        mock_run.assert_called_once()

    @patch("fastdeploy.entrypoints.openai.api_server.LLMEngine")
    @patch("uvicorn.run")
    @patch("socket.socket")
    def test_metrics_server_app(self, mock_socket, mock_run, mock_engine):
        app = MetricsServerApp(self.test_args)
        TestClient(app.build_app())
        app.launch_metrics_server()
        mock_run.assert_called_once()

    @patch("fastdeploy.entrypoints.openai.api_server.get_filtered_metrics")
    def test_metrics_server_app_metrics(self, mock_get_filtered_metrics):
        mock_get_filtered_metrics.return_value = "test_metrics_data"
        server_app = MetricsServerApp(self.test_args)
        app = server_app.build_app()
        test_client = TestClient(app)
        response = test_client.get("/metrics")
        assert response.status_code == 200

    @patch("fastdeploy.entrypoints.openai.api_server.LLMEngine")
    @patch("uvicorn.run")
    @patch("socket.socket")
    def test_controller_server_app(self, mock_socket, mock_run, mock_engine):

        app = ControllerServerApp(self.test_args)
        TestClient(app.build_app())
        app.launch_controller_server()
        mock_run.assert_called_once()

    @patch("fastdeploy.entrypoints.openai.api_server.LLMEngine")
    def test_controller_server_app_reset_scheduler(self, mock_llm_engine):
        mock_llm_engine.engine = MagicMock()
        server_app = ControllerServerApp(self.test_args)
        mock_llm_engine.engine.scheduler = MagicMock()
        mock_llm_engine.engine.scheduler.reset()
        with patch("fastdeploy.entrypoints.openai.api_server.llm_engine", mock_llm_engine):
            app = server_app.build_app()
            test_client = TestClient(app)
            response = test_client.post("/controller/reset_scheduler")
            assert response.status_code == 200

    @patch("fastdeploy.entrypoints.openai.api_server.LLMEngine")
    def test_controller_server_app_scheduler(self, mock_llm_engine):
        mock_llm_engine.engine = MagicMock()
        server_app = ControllerServerApp(self.test_args)
        mock_llm_engine.engine.scheduler = MagicMock()
        mock_llm_engine.engine.scheduler.reset()
        with patch("fastdeploy.entrypoints.openai.api_server.llm_engine", mock_llm_engine):
            app = server_app.build_app()
            test_client = TestClient(app)
            response = test_client.post(
                "/controller/scheduler", json={"reset": True, "load_shards_num": 8, "reallocate_shard": True}
            )
            assert response.status_code == 200

    @patch("fastdeploy.entrypoints.openai.api_server.LLMEngine")
    @patch("uvicorn.run")
    @patch("socket.socket")
    def test_main(self, mock_socket, mock_run, mock_engine):
        with patch("fastdeploy.entrypoints.openai.api_server.retrive_model_from_server", return_value="test_model"):
            api_server.main(self.test_args)

    @patch("sys.platform", "win32")
    @patch("fastdeploy.entrypoints.openai.api_server.api_server_logger")
    def test_windows_platform(self, mock_logger):
        """Test that ulimit is not set on Windows"""
        api_server.set_ulimit()
        mock_logger.info.assert_called_with("Windows detected, skipping ulimit adjustment.")

    @patch("sys.platform", "linux")
    @patch("resource.getrlimit")
    @patch("resource.setrlimit")
    @patch("fastdeploy.entrypoints.openai.api_server.api_server_logger")
    def test_linux_increase_limit_success(self, mock_logger, mock_set, mock_get):
        """Test successful ulimit increase on Linux"""
        # Setup mock
        mock_get.return_value = (1024, 65535)  # current soft, hard limits

        # Call function
        api_server.set_ulimit()

        # Verify
        mock_logger.warning.assert_not_called()

    @patch("sys.platform", "linux")
    @patch("resource.getrlimit")
    @patch("resource.setrlimit")
    @patch("fastdeploy.entrypoints.openai.api_server.api_server_logger")
    def test_linux_increase_limit_failure(self, mock_logger, mock_set, mock_get):
        """Test failed ulimit increase on Linux"""
        # Setup mock
        mock_get.return_value = (1024, 65535)
        mock_set.side_effect = ValueError("Permission denied")

        # Call function
        api_server.set_ulimit()

        # Verify
        mock_logger.warning.assert_called()

    @patch("sys.platform", "linux")
    @patch("resource.getrlimit")
    @patch("resource.setrlimit")
    @patch("fastdeploy.entrypoints.openai.api_server.api_server_logger")
    def test_linux_limit_already_high(self, mock_logger, mock_set, mock_get):
        """Test when current limit is already higher than target"""
        # Setup mock
        mock_get.return_value = (65536, 65536)  # already higher

        # Call function
        api_server.set_ulimit()

        # Verify
        mock_set.assert_not_called()
        mock_logger.warning.assert_not_called()

    @patch("socket.socket")
    @patch("fastdeploy.entrypoints.openai.api_server.is_valid_ipv6_address", return_value=False)
    def test_ipv4_socket_creation(self, mock_ipv6_check, mock_socket):
        """Test IPv4 socket creation"""
        # Setup mock
        mock_sock = MagicMock()
        mock_socket.return_value = mock_sock

        # Test data
        test_addr = ("127.0.0.1", 8000)

        # Call function
        result = api_server.create_server_socket(test_addr)

        # Verify
        mock_ipv6_check.assert_called_with(test_addr[0])
        mock_sock.bind.assert_called_with(test_addr)
        self.assertEqual(result, mock_sock)

    @patch("socket.socket")
    @patch("fastdeploy.entrypoints.openai.api_server.is_valid_ipv6_address", return_value=True)
    def test_ipv6_socket_creation(self, mock_ipv6_check, mock_socket):
        """Test IPv6 socket creation"""
        # Setup mock
        mock_sock = MagicMock()
        mock_socket.return_value = mock_sock

        # Test data
        test_addr = ("::1", 8000)

        # Call function
        result = api_server.create_server_socket(test_addr)

        # Verify
        mock_ipv6_check.assert_called_with(test_addr[0])
        mock_sock.bind.assert_called_with(test_addr)
        self.assertEqual(result, mock_sock)

    @patch("socket.socket")
    @patch("fastdeploy.entrypoints.openai.api_server.is_valid_ipv6_address", return_value=False)
    def test_bind_failure(self, mock_ipv6_check, mock_socket):
        """Test socket bind failure"""
        # Setup mock
        mock_sock = MagicMock()
        mock_socket.return_value = mock_sock
        mock_sock.bind.side_effect = OSError("Bind failed")

        # Test data
        test_addr = ("127.0.0.1", 8000)

        # Verify exception is raised
        with self.assertRaises(OSError):
            api_server.create_server_socket(test_addr)


class TestRunApiServerWorkerProc(unittest.TestCase):
    def setUp(self):
        self.args = mock.MagicMock()
        self.args.host = "127.0.0.1"
        self.args.port = 8000
        self.listen_address = "http://127.0.0.1:8000"
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def tearDown(self):
        self.sock.close()

    @mock.patch("uvloop.run")
    @mock.patch("setproctitle.setproctitle")
    def test_basic_functionality(self, mock_setproctitle, mock_uvloop_run):
        """Test basic functionality without client_config"""
        run_api_server_worker_proc(self.args, self.listen_address, self.sock)

        mock_setproctitle.assert_called_once_with("APIServer::0")
        mock_uvloop_run.assert_called_once()

    @mock.patch("uvloop.run")
    @mock.patch("setproctitle.setproctitle")
    def test_with_client_config(self, mock_setproctitle, mock_uvloop_run):
        """Test with client_config parameter"""
        client_config = {"client_count": 2, "client_index": 1}

        run_api_server_worker_proc(self.args, self.listen_address, self.sock, client_config=client_config)

        mock_setproctitle.assert_called_once_with("APIServer::1")
        mock_uvloop_run.assert_called_once()

    @mock.patch("uvloop.run")
    @mock.patch("setproctitle.setproctitle")
    def test_with_uvicorn_kwargs(self, mock_setproctitle, mock_uvloop_run):
        """Test with uvicorn kwargs"""
        uvicorn_kwargs = {"log_level": "debug", "timeout_graceful_shutdown": 30}

        run_api_server_worker_proc(self.args, self.listen_address, self.sock, **uvicorn_kwargs)

        mock_setproctitle.assert_called_once_with("APIServer::0")
        mock_uvloop_run.assert_called_once()


class TestApiServerApi(unittest.TestCase):

    def setUp(self):
        self.test_args = MagicMock()
        self.test_args.port = 123123
        self.test_args.host = "0.0.0.0"
        self.test_args.workers = 1
        self.test_args.metrics_port = 12334
        self.test_args.controller_port = 12231
        self.test_args.max_waiting_time = -1
        self.test_args.max_concurrency = 512
        self.test_args.enable_mm_output = False
        self.test_args.timeout_graceful_shutdown = 0
        self.test_args.model = "test_model"
        self.test_args.revision = None
        self.test_args.tokenizer = None
        self.test_args.max_model_len = 2048
        self.test_args.tensor_parallel_size = 1
        self.test_args.engine_worker_queue_port = [8002]
        self.test_args.local_data_parallel_id = 0
        self.test_args.limit_mm_per_prompt = None
        self.test_args.mm_processor_kwargs = None
        self.test_args.reasoning_parser = None
        self.test_args.data_parallel_size = 1
        self.test_args.enable_logprob = False
        self.test_args.tool_call_parser = None
        self.test_args.dynamic_load_weight = False
        self.test_args.served_model_name = None
        self.test_args.hidden_size = 11
        self.test_args.num_attention_heads = 11
        self.test_args.chat_template = None
        self.test_args.ips = "127.0.0.1"
        self.test_args.tokenizer_base_url = None

    @patch("fastdeploy.entrypoints.openai.api_server.LLMEngine")
    @patch("fastdeploy.entrypoints.openai.api_server.retrive_model_from_server", return_value="test_model")
    def test_api_server_app_helth(self, mock_retrieve_model, mock_engine):
        server_app = ApiServerApp(self.test_args)
        app = server_app.build_app()
        app.state.engine_client = mock_engine
        mock_engine.is_workers_alive.return_value = (True, "")
        mock_engine.check_health.return_value = (True, "")
        test_client = TestClient(app)
        response = test_client.get("/health")
        assert response.status_code == 200

    @patch("fastdeploy.entrypoints.openai.api_server.LLMEngine")
    @patch("fastdeploy.entrypoints.openai.api_server.retrive_model_from_server", return_value="test_model")
    def test_api_server_app_load(self, mock_retrieve_model, mock_engine):
        server_app = ApiServerApp(self.test_args)
        app = server_app.build_app()
        app.state.engine_client = mock_engine
        test_client = TestClient(app)
        response = test_client.get("/load")
        assert response.status_code == 200

    @patch("fastdeploy.entrypoints.openai.api_server.LLMEngine")
    @patch("fastdeploy.entrypoints.openai.api_server.retrive_model_from_server", return_value="test_model")
    def test_api_server_app_ping(self, mock_retrieve_model, mock_engine):
        server_app = ApiServerApp(self.test_args)
        app = server_app.build_app()
        app.state.engine_client = mock_engine
        mock_engine.is_workers_alive.return_value = (True, "")
        mock_engine.check_health.return_value = (True, "")
        test_client = TestClient(app)
        response = test_client.get("/ping")
        assert response.status_code == 200

    @patch("fastdeploy.entrypoints.openai.api_server.LLMEngine")
    @patch("fastdeploy.entrypoints.openai.api_server.retrive_model_from_server", return_value="test_model")
    def test_api_server_app_chat_completion(self, mock_retrieve_model, mock_engine):
        server_app = ApiServerApp(self.test_args)
        app = server_app.build_app()
        app.state.engine_client = mock_engine
        app.state.dynamic_load_weight = True
        chat_handler = MagicMock()
        create_chat_completion_mock = AsyncMock()
        chat_handler.create_chat_completion = create_chat_completion_mock
        create_chat_completion_mock.return_value = ChatCompletionResponse(
            id="test_id",
            created=1677900000,
            model="test_model",
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="Hello World!"),
                    finish_reason=None,
                ),
            ],
            usage=UsageInfo(
                prompt_tokens=10,
                completion_tokens=10,
                total_tokens=20,
            ),
        )

        app.state.chat_handler = chat_handler
        mock_engine.is_workers_alive.return_value = (True, "")
        test_client = TestClient(app)
        json = {
            "model": "gptj",
            "messages": [
                {"role": "user", "content": "Hello!"},
            ],
        }
        response = test_client.post("/v1/chat/completions", json=json)
        assert response.status_code == 200

    @patch("fastdeploy.entrypoints.openai.api_server.LLMEngine")
    @patch("fastdeploy.entrypoints.openai.api_server.retrive_model_from_server", return_value="test_model")
    def test_api_server_app_chat_completion_stream(self, mock_retrieve_model, mock_engine):
        server_app = ApiServerApp(self.test_args)
        app = server_app.build_app()
        app.state.engine_client = mock_engine
        app.state.dynamic_load_weight = True
        chat_handler = MagicMock()
        create_chat_completion_mock = AsyncMock(side_effect=gen_chat_values)
        chat_handler.create_chat_completion = create_chat_completion_mock
        app.state.chat_handler = chat_handler
        mock_engine.is_workers_alive.return_value = (True, "")
        test_client = TestClient(app)
        json = {
            "model": "gptj",
            "messages": [
                {"role": "user", "content": "Hello!"},
            ],
        }
        response = test_client.post("/v1/chat/completions", json=json)
        assert response.status_code == 200

    @patch("fastdeploy.entrypoints.openai.api_server.LLMEngine")
    @patch("fastdeploy.entrypoints.openai.api_server.retrive_model_from_server", return_value="test_model")
    def test_api_server_app_completion(self, mock_retrieve_model, mock_engine):
        server_app = ApiServerApp(self.test_args)
        app = server_app.build_app()
        app.state.engine_client = mock_engine
        app.state.dynamic_load_weight = True
        completion_handler = MagicMock()
        create_chat_completion_mock = AsyncMock()
        completion_handler.create_completion = create_chat_completion_mock
        create_chat_completion_mock.return_value = CompletionResponse(
            id="test_id",
            created=1677900000,
            model="test_model",
            choices=[
                CompletionResponseChoice(
                    index=0,
                    text="Hello World!",
                    finish_reason=None,
                ),
            ],
            usage=UsageInfo(
                prompt_tokens=10,
                completion_tokens=10,
                total_tokens=20,
            ),
        )

        app.state.completion_handler = completion_handler
        mock_engine.is_workers_alive.return_value = (True, "")
        test_client = TestClient(app)
        json = {
            "model": "gptj",
            "prompt": "Hello!",
        }
        response = test_client.post("/v1/completions", json=json)
        assert response.status_code == 200

    @patch("fastdeploy.entrypoints.openai.api_server.LLMEngine")
    @patch("fastdeploy.entrypoints.openai.api_server.retrive_model_from_server", return_value="test_model")
    def test_api_server_app_completion_stream(self, mock_retrieve_model, mock_engine):
        server_app = ApiServerApp(self.test_args)
        app = server_app.build_app()
        app.state.engine_client = mock_engine
        app.state.dynamic_load_weight = True
        completion_handler = MagicMock()
        completion_mock = AsyncMock(side_effect=gen_completion_values)
        completion_handler.create_completion = completion_mock
        app.state.completion_handler = completion_handler
        mock_engine.is_workers_alive.return_value = (True, "")
        test_client = TestClient(app)
        json = {
            "model": "gptj",
            "prompt": "Hello!",
        }
        response = test_client.post("/v1/completions", json=json)
        assert response.status_code == 200

    @patch("fastdeploy.entrypoints.openai.api_server.LLMEngine")
    @patch("fastdeploy.entrypoints.openai.api_server.retrive_model_from_server", return_value="test_model")
    def test_api_server_app_v1_models(self, mock_retrieve_model, mock_engine):
        server_app = ApiServerApp(self.test_args)
        app = server_app.build_app()
        app.state.engine_client = mock_engine
        app.state.dynamic_load_weight = True
        model_handler = MagicMock()
        list_models_mock = AsyncMock()
        list_models_mock.return_value = ModelList(
            object="list",
            data=[
                ModelInfo(
                    id="test_model",
                    owned_by="test_owner",
                    permission=[],
                    root="test_root",
                    parent="test_parent",
                    ready=True,
                )
            ],
        )
        model_handler.list_models = list_models_mock
        app.state.model_handler = model_handler
        mock_engine.is_workers_alive.return_value = (True, "")
        test_client = TestClient(app)
        response = test_client.get("/v1/models")
        assert response.status_code == 200

    @patch("fastdeploy.entrypoints.openai.api_server.LLMEngine")
    @patch("fastdeploy.entrypoints.openai.api_server.retrive_model_from_server", return_value="test_model")
    def test_api_server_app_update_model_weight(self, mock_retrieve_model, mock_engine):
        server_app = ApiServerApp(self.test_args)
        app = server_app.build_app()
        app.state.engine_client = mock_engine
        mock_engine.update_model_weight.return_value = (True, "")
        app.state.dynamic_load_weight = True
        test_client = TestClient(app)
        response = test_client.get("/update_model_weight")
        assert response.status_code == 200

    @patch("fastdeploy.entrypoints.openai.api_server.LLMEngine")
    @patch("fastdeploy.entrypoints.openai.api_server.retrive_model_from_server", return_value="test_model")
    def test_api_server_app_clear_load_weight(self, mock_retrieve_model, mock_engine):
        server_app = ApiServerApp(self.test_args)
        app = server_app.build_app()
        app.state.engine_client = mock_engine
        mock_engine.clear_load_weight.return_value = (True, "")
        app.state.dynamic_load_weight = True
        test_client = TestClient(app)
        response = test_client.get("/clear_load_weight")
        assert response.status_code == 200


async def gen_chat_values(token):
    chunk = ChatCompletionStreamResponse(
        id="test_id",
        created=1677900000,
        model="test_model",
        choices=[
            ChatCompletionResponseStreamChoice(
                index=0,
                delta=DeltaMessage(role="assistant", content="Hello World!"),
                finish_reason=None,
            ),
        ],
        usage=UsageInfo(
            prompt_tokens=10,
            completion_tokens=10,
            total_tokens=20,
        ),
    )
    data = f"data: {chunk.model_dump_json(exclude_unset=True)} \n\n"
    yield data
    yield data


async def gen_completion_values(token):
    chunk = CompletionStreamResponse(
        id="test_id",
        created=1677900000,
        model="test_model",
        choices=[
            CompletionResponseStreamChoice(
                index=0,
                text="Hello World!",
                arrival_time=111,
            ),
        ],
        usage=UsageInfo(
            prompt_tokens=10,
            completion_tokens=10,
            total_tokens=20,
        ),
    )
    data = f"data: {chunk.model_dump_json(exclude_unset=True)} \n\n"
    yield data
    yield data


if __name__ == "__main__":
    unittest.main()
