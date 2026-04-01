import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.gpu


def test_vocab_parallel_embedding_deterministic():
    """Launch vocab parallel embedding deterministic test on 2 GPUs."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(current_dir, "vocab_parallel_embedding_deterministic.py")

    command = [sys.executable, "-m", "paddle.distributed.launch", "--gpus", "0,1,2,3", script]

    print(f"Executing command: {' '.join(command)}")

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    try:
        stdout, stderr = process.communicate(timeout=300)
        return_code = process.returncode
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        return_code = -1

    print("\n" + "=" * 50 + " STDOUT " + "=" * 50)
    print(stdout)
    print("\n" + "=" * 50 + " STDERR " + "=" * 50)
    print(stderr)

    assert return_code == 0, f"Process exited with code {return_code}\nSTDERR: {stderr[-500:] if stderr else 'N/A'}"
