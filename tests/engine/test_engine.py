from unittest.mock import MagicMock, patch

import pytest

from fastdeploy.engine.engine import LLMEngine as Engine


class TestLaunchComponents:
    @pytest.fixture
    def engine(self):
        eng = MagicMock()
        eng.cfg = MagicMock()
        eng.cfg.parallel_config = MagicMock()
        eng.cfg.parallel_config.enable_expert_parallel = False
        eng.cfg.scheduler_config = MagicMock()
        eng.engine = MagicMock()
        eng.engine.scheduler = MagicMock()
        eng.engine.split_connector = MagicMock()
        eng.launched_expert_service_signal = MagicMock()
        eng.dp_processed = []
        eng.dp_engine_worker_queue_server = []
        return eng

    def test_launch_components_splitwise_role_not_mixed(self, engine):
        engine.cfg.scheduler_config.splitwise_role = "not_mixed"
        engine.cfg.scheduler_config.name = "splitwise"

        with patch("threading.Thread") as mock_thread:
            Engine.launch_components(engine)

            mock_thread.assert_called_once_with(target=engine.engine.split_connector.start_receiver, args=())
            mock_thread.return_value.start.assert_called_once()

    def test_launch_components_splitwise_scheduler(self, engine):
        engine.cfg.scheduler_config.splitwise_role = "role1"
        engine.cfg.scheduler_config.name = "splitwise"
        engine.cfg.host_ip = "127.0.0.1"
        engine.cfg.disaggregate_info = "info"

        Engine.launch_components(engine)

        engine.engine.scheduler.start.assert_called_once_with("role1", "127.0.0.1", "info")

    def test_launch_components_dp_scheduler(self, engine):
        engine.cfg.scheduler_config.name = "dp"
        engine.cfg.node_rank = 1
        engine.cfg.worker_num_per_node = 2
        engine.cfg.parallel_config = MagicMock()
        engine.cfg.parallel_config.data_parallel_size = 2

        with patch("multiprocessing.Queue"):
            Engine.launch_components(engine)

            assert len(engine.engine.scheduler.start.call_args[0]) == 3
            assert engine.engine.scheduler.start.call_args[0][0] == 0  # calculated node rank

    def test_launch_components_expert_parallel(self, engine):
        engine.cfg.scheduler_config.name = "dp"
        engine.cfg.parallel_config = MagicMock()
        engine.cfg.parallel_config.enable_expert_parallel = True
        engine.cfg.parallel_config.data_parallel_size = 4
        engine.cfg.nnode = 2
        engine.cfg.master_ip = "127.0.0.1"
        engine.cfg.parallel_config.engine_worker_queue_port = [0, 1234, 5678]
        engine.cfg.parallel_config.tensor_parallel_size = 2
        engine.launched_expert_service_signal.value = [1] * 4  # mock 默认启动成功，避免阻塞

        with (
            patch("multiprocessing.Queue"),
            patch("multiprocessing.Process") as mock_process,
            patch("fastdeploy.engine.engine.EngineWorkerQueue") as mock_queue,
            patch("fastdeploy.utils.is_port_available", return_value=True),
            patch("fastdeploy.engine.expert_service.start_data_parallel_service"),
            patch.dict("os.environ", {"FD_ENABLE_MULTI_API_SERVER": "0", "FD_ENGINE_TASK_QUEUE_WITH_SHM": "0"}),
        ):
            mock_process.return_value.start = MagicMock()
            Engine.launch_components(engine)

            assert mock_process.call_count == 1
            assert mock_queue.call_count == 1
            assert engine.launched_expert_service_signal.value[0] == 1

    def test_launch_components_port_unavailable(self, engine):
        engine.cfg.scheduler_config.name = "dp"
        engine.cfg.parallel_config = MagicMock()
        engine.cfg.parallel_config.enable_expert_parallel = True
        engine.cfg.parallel_config.data_parallel_size = 4
        engine.cfg.nnode = 2
        engine.cfg.master_ip = "127.0.0.1"
        engine.cfg.parallel_config.engine_worker_queue_port = [8000, 8001, 8002, 8003]
        engine.launched_expert_service_signal.value = [0] * 4

        with (
            patch("multiprocessing.Queue"),
            patch("fastdeploy.engine.engine.is_port_available", return_value=False),
            patch.dict("os.environ", {"FD_ENABLE_MULTI_API_SERVER": "0", "FD_ENGINE_TASK_QUEUE_WITH_SHM": "0"}),
        ):

            with pytest.raises(Exception) as excinfo:
                Engine.launch_components(engine)
            assert "is already in use" in str(excinfo.value)

    def test_launch_components_with_shm(self, engine):
        engine.cfg.scheduler_config.name = "dp"
        engine.cfg.parallel_config = MagicMock()
        engine.cfg.parallel_config.enable_expert_parallel = True
        engine.cfg.parallel_config.data_parallel_size = 4
        engine.cfg.nnode = 1
        engine.cfg.master_ip = "127.0.0.1"
        engine.cfg.parallel_config.engine_worker_queue_port = [8000, 8001, 8002, 8003]
        engine.launched_expert_service_signal.value = [1] * 4  # mock 默认启动成功，避免阻塞

        with (
            patch("multiprocessing.Queue"),
            patch("multiprocessing.Process"),
            patch("fastdeploy.engine.engine.EngineWorkerQueue") as mock_queue,
            patch("fastdeploy.engine.engine.is_port_available"),
            patch("fastdeploy.engine.engine.start_data_parallel_service"),
            patch.dict("os.environ", {"FD_ENABLE_MULTI_API_SERVER": "0", "FD_ENGINE_TASK_QUEUE_WITH_SHM": "1"}),
        ):

            Engine.launch_components(engine)

            calls = mock_queue.call_args_list
            assert calls[0].kwargs["address"] == "/dev/shm/fd_task_queue_8001.sock"
            assert calls[1].kwargs["address"] == "/dev/shm/fd_task_queue_8002.sock"
            assert calls[2].kwargs["address"] == "/dev/shm/fd_task_queue_8003.sock"
