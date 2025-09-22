import signal
import unittest
from unittest.mock import MagicMock, patch

import psutil

from fastdeploy.utils import kill_process_tree


class TestKillProcessTree(unittest.TestCase):
    @patch("psutil.Process")
    @patch("os.kill")
    def test_kill_process_tree_success(self, mock_os_kill, mock_process):
        # Setup mock process tree
        parent_process = MagicMock()
        child1 = MagicMock()
        child1.pid = 1001
        child2 = MagicMock()
        child2.pid = 1002
        parent_process.children.return_value = [child1, child2]
        mock_process.return_value = parent_process

        # Call function
        kill_process_tree(1234)

        # Verify
        mock_process.assert_called_once_with(1234)
        parent_process.children.assert_called_once_with(recursive=True)
        self.assertEqual(mock_os_kill.call_count, 3)  # 2 children + parent
        mock_os_kill.assert_any_call(1001, signal.SIGKILL)
        mock_os_kill.assert_any_call(1002, signal.SIGKILL)
        mock_os_kill.assert_any_call(1234, signal.SIGKILL)

    @patch("psutil.Process")
    def test_kill_process_tree_no_such_process(self, mock_process):
        mock_process.side_effect = psutil.NoSuchProcess(1234)

        # Should not raise exception
        kill_process_tree(1234)

        mock_process.assert_called_once_with(1234)

    @patch("psutil.Process")
    @patch("os.kill")
    def test_kill_process_tree_child_kill_failure(self, mock_os_kill, mock_process):
        parent_process = MagicMock()
        child = MagicMock()
        child.pid = 1001
        parent_process.children.return_value = [child]
        mock_process.return_value = parent_process

        # First child kill fails, parent kill succeeds
        mock_os_kill.side_effect = [ProcessLookupError, None]

        # Should not raise exception
        kill_process_tree(1234)

        mock_os_kill.assert_any_call(1001, signal.SIGKILL)
        mock_os_kill.assert_any_call(1234, signal.SIGKILL)


if __name__ == "__main__":
    unittest.main()
