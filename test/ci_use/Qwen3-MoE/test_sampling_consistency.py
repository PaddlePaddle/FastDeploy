import os
import signal
import socket
import subprocess
import sys
import time

import requests


def is_port_open(host: str, port: int, timeout=1.0):
    """Check if a TCP port is open on the given host."""
    try:
        with socket.create_connection((host, port), timeout):
            return True
    except Exception:
        return False


def kill_process_on_port(port: int):
    """Kill processes that are listening on the given port."""
    try:
        output = subprocess.check_output(f"lsof -i:{port} -t", shell=True).decode().strip()
        for pid in output.splitlines():
            os.kill(int(pid), signal.SIGKILL)
            print(f"Killed process on port {port}, pid={pid}")
    except subprocess.CalledProcessError:
        pass


def clean_specific_ports(ports_list):
    """Kill all processes occupying the specified ports."""
    for port in ports_list:
        kill_process_on_port(port)


def create_server_process_with_sampling(sampling_class: str, api_port: int, queue_port: int, metrics_port: int):
    """Create and start the API server process with specified sampling class and ports."""
    base_path = os.getenv("MODEL_PATH")
    if base_path:
        model_path = os.path.join(base_path, "Qwen3-30B-A3B")
    else:
        model_path = "./Qwen3-30B-A3B"

    log_path = f"server_{sampling_class}_{api_port}.log"
    cmd = [
        sys.executable,
        "-m",
        "fastdeploy.entrypoints.openai.api_server",
        "--model",
        model_path,
        "--port",
        str(api_port),
        "--tensor-parallel-size",
        "1",
        "--engine-worker-queue-port",
        str(queue_port),
        "--metrics-port",
        str(metrics_port),
        "--max-model-len",
        "32768",
        "--max-num-seqs",
        "50",
        "--quantization",
        "wint4",
    ]

    env = os.environ.copy()
    env["FD_SAMPLING_CLASS"] = sampling_class

    print(f"Starting server with FD_SAMPLING_CLASS={sampling_class} on port {api_port}")

    with open(log_path, "w") as logfile:
        process = subprocess.Popen(
            cmd,
            stdout=logfile,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )

    return process


def wait_for_server_ready_on_port(api_port: int, timeout=300):
    """Wait for the API server to be ready on specified port."""
    for _ in range(timeout):
        if is_port_open("127.0.0.1", api_port):
            print(f"API server is up on port {api_port}")
            return True
        time.sleep(1)
    return False


def test_seed_consistency_rejection_sampling():
    """
    Test seed consistency for rejection sampling - multiple runs should produce identical results.
    """
    sampling_class = "rejection"
    api_port = 8288
    queue_port = 8334
    metrics_port = 8433
    num_runs = 5

    ports_to_clean = [api_port, queue_port, metrics_port]
    api_url = f"http://0.0.0.0:{api_port}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}

    consistent_payload = {
        "messages": [
            {
                "role": "user",
                "content": "用一句话介绍 PaddlePaddle, 30字以内 /no_think",
            }
        ],
        "temperature": 0.8,
        "seed": 42,  # Fixed seed
        "max_tokens": 50,
    }

    print(f"\n===== Testing seed consistency for {sampling_class.upper()} sampling =====")

    # Clean ports and start server
    clean_specific_ports(ports_to_clean)
    time.sleep(2)
    process = create_server_process_with_sampling(sampling_class, api_port, queue_port, metrics_port)

    try:
        if not wait_for_server_ready_on_port(api_port, timeout=300):
            raise RuntimeError(f"Server failed to start for {sampling_class}")

        # Run multiple requests with same seed
        results = []
        print(f"Running {num_runs} requests with fixed seed=42:")

        for i in range(num_runs):
            resp = requests.post(api_url, headers=headers, json=consistent_payload)
            assert resp.status_code == 200, f"Request {i+1} failed"

            content = resp.json()["choices"][0]["message"]["content"]
            results.append(content)
            time.sleep(1)

        # Check if all results are identical
        reference_result = results[0]
        all_identical = all(result == reference_result for result in results)

        print(f"\n--- {sampling_class.upper()} Sampling Results ---")
        if all_identical:
            print(f"ALL {num_runs} runs produced IDENTICAL results")
            print(f"   Result: {reference_result}")
        else:
            print(" Results are NOT identical:")
            for i, result in enumerate(results):
                status = "yes" if result == reference_result else "no"
                print(f"   Run {i+1} {status}: {result}")

        return all_identical, results

    finally:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            print(f"Server terminated for {sampling_class}")
        except Exception as e:
            print(f"Failed to terminate server: {e}")
        time.sleep(3)


def test_seed_consistency_air_sampling():
    """
    Test seed consistency for AIR sampling - multiple runs should produce identical results.
    """
    sampling_class = "air"
    api_port = 8123
    queue_port = 8534
    metrics_port = 8643
    num_runs = 5

    ports_to_clean = [api_port, queue_port, metrics_port]
    api_url = f"http://0.0.0.0:{api_port}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}

    consistent_payload = {
        "messages": [
            {
                "role": "user",
                "content": "用一句话介绍 PaddlePaddle, 30字以内 /no_think",
            }
        ],
        "temperature": 0.8,
        "seed": 42,  # Fixed seed
        "max_tokens": 50,
    }

    print(f"\n===== Testing seed consistency for {sampling_class.upper()} sampling =====")

    # Clean ports and start server
    clean_specific_ports(ports_to_clean)
    time.sleep(2)
    process = create_server_process_with_sampling(sampling_class, api_port, queue_port, metrics_port)

    try:
        if not wait_for_server_ready_on_port(api_port, timeout=300):
            print(f"Server startup failed. Checking log file: server_{sampling_class}_{api_port}.log")
            # 打印日志文件的最后几行来诊断问题
            log_file = f"server_{sampling_class}_{api_port}.log"
            if os.path.exists(log_file):
                with open(log_file, "r") as f:
                    lines = f.readlines()
                    print("Last 10 lines of server log:")
                    for line in lines[-10:]:
                        print(f"  {line.strip()}")
            raise RuntimeError(f"Server failed to start for {sampling_class}")

        # Run multiple requests with same seed
        results = []
        print(f"Running {num_runs} requests with fixed seed=42:")

        for i in range(num_runs):
            resp = requests.post(api_url, headers=headers, json=consistent_payload)
            assert resp.status_code == 200, f"Request {i+1} failed"

            content = resp.json()["choices"][0]["message"]["content"]
            results.append(content)
            time.sleep(1)

        # Check if all results are identical
        reference_result = results[0]
        all_identical = all(result == reference_result for result in results)

        print(f"\n--- {sampling_class.upper()} Sampling Results ---")
        if all_identical:
            print(f"ALL {num_runs} runs produced IDENTICAL results")
            print(f"   Result: {reference_result}")
        else:
            print(" Results are NOT identical:")
            for i, result in enumerate(results):
                status = "yes" if result == reference_result else "no"
                print(f"   Run {i+1} {status}: {result}")

        return all_identical, results

    finally:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            print(f"Server terminated for {sampling_class}")
        except Exception as e:
            print(f"Failed to terminate server: {e}")
        time.sleep(3)


def test_both_sampling_seed_consistency():
    """
    Test seed consistency for both rejection and AIR sampling methods.
    """
    print("\n" + "=" * 80)
    print("TESTING SEED CONSISTENCY FOR BOTH SAMPLING METHODS")
    print("=" * 80)

    # Test rejection sampling
    rejection_consistent, rejection_results = test_seed_consistency_rejection_sampling()

    # Test AIR sampling
    air_consistent, air_results = test_seed_consistency_air_sampling()

    # Summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print(f"REJECTION Sampling: {'CONSISTENT' if rejection_consistent else ' NOT CONSISTENT'}")
    if rejection_consistent:
        print(f"  All runs produced: {rejection_results[0]}")
    else:
        print(f"  {len(set(rejection_results))} different outputs detected")

    print(f"AIR Sampling: {'CONSISTENT' if air_consistent else ' NOT CONSISTENT'}")
    if air_consistent:
        print(f"  All runs produced: {air_results[0]}")
    else:
        print(f"  {len(set(air_results))} different outputs detected")

    # Assert both should be consistent with fixed seed
    assert rejection_consistent, "Rejection sampling should be consistent with fixed seed"
    assert air_consistent, "AIR sampling should be consistent with fixed seed"

    print("\n🎉 Both sampling methods show proper seed consistency!")


if __name__ == "__main__":
    # 可以直接运行这个文件进行测试
    test_both_sampling_seed_consistency()
