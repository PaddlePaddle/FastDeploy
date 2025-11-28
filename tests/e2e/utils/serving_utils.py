import os
import signal
import socket
import subprocess
import time

import requests

# Read ports from environment variables; use default values if not set
FD_API_PORT = int(os.getenv("FD_API_PORT", 8188))
FD_ENGINE_QUEUE_PORT = int(os.getenv("FD_ENGINE_QUEUE_PORT", 8133))
FD_METRICS_PORT = int(os.getenv("FD_METRICS_PORT", 8233))
FD_CACHE_QUEUE_PORT = int(os.getenv("FD_CACHE_QUEUE_PORT", 8333))

# List of ports to clean before and after tests
PORTS_TO_CLEAN = [FD_API_PORT, FD_ENGINE_QUEUE_PORT, FD_METRICS_PORT, FD_CACHE_QUEUE_PORT]


def is_port_open(host: str, port: int, timeout=1.0):
    """
    Check if a TCP port is open on the given host.
    Returns True if connection succeeds, False otherwise.
    """
    try:
        with socket.create_connection((host, port), timeout):
            return True
    except Exception:
        return False


def _clean_cuda_process():
    """
    Kill processes that are using CUDA devices.
    NOTE: Do not call this function directly, use the `clean` function instead.
    """
    try:
        subprocess.run("fuser -k /dev/nvidia*", shell=True, timeout=5)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        pass


def kill_process_on_port(port: int):
    """
    Kill processes that are listening on the given port.
    Uses multiple methods to ensure thorough cleanup.
    """
    current_pid = os.getpid()
    parent_pid = os.getppid()

    # Method 1: Use lsof to find processes
    try:
        output = subprocess.check_output(f"lsof -i:{port} -t", shell=True).decode().strip()
        for pid in output.splitlines():
            pid = int(pid)
            if pid in (current_pid, parent_pid):
                print(f"Skip killing current process (pid={pid}) on port {port}")
                continue
            try:
                # First try SIGTERM for graceful shutdown
                os.kill(pid, signal.SIGTERM)
                time.sleep(1)
                # Then SIGKILL if still running
                os.kill(pid, signal.SIGKILL)
                print(f"Killed process on port {port}, pid={pid}")
            except ProcessLookupError:
                pass  # Process already terminated
    except subprocess.CalledProcessError:
        pass

    # Method 2: Use netstat and fuser as backup
    try:
        # Find processes using netstat and awk
        cmd = f"netstat -tulpn 2>/dev/null | grep :{port} | awk '{{print $7}}' | cut -d'/' -f1"
        output = subprocess.check_output(cmd, shell=True).decode().strip()
        for pid in output.splitlines():
            if pid and pid.isdigit():
                pid = int(pid)
                if pid in (current_pid, parent_pid):
                    continue
                try:
                    os.kill(pid, signal.SIGKILL)
                    print(f"Killed process (netstat) on port {port}, pid={pid}")
                except ProcessLookupError:
                    pass
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Method 3: Use fuser if available
    try:
        subprocess.run(f"fuser -k {port}/tcp", shell=True, timeout=5)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        pass


def clean_ports(ports=None):
    """
    Kill all processes occupying the ports
    """
    if ports is None:
        ports = PORTS_TO_CLEAN

    print(f"Cleaning ports: {ports}")
    for port in ports:
        kill_process_on_port(port)

    # Double check and retry if ports are still in use
    time.sleep(2)
    for port in ports:
        if is_port_open("127.0.0.1", port, timeout=0.1):
            print(f"Port {port} still in use, retrying cleanup...")
            kill_process_on_port(port)
            time.sleep(1)


def clean(ports=None):
    """
    Clean up resources used during testing.
    """
    clean_ports(ports)

    # Clean CUDA devices before and after tests.
    # NOTE: It is dangerous to use this flag on development machines, as it may kill other processes
    clean_cuda = int(os.getenv("CLEAN_CUDA", "0")) == 1
    if clean_cuda:
        _clean_cuda_process()


def check_service_health(base_url: str, timeout: int = 3) -> bool:
    """
    Check the health status of a service.

    Args:
        base_url (str): The base URL of the service, e.g. "http://127.0.0.1:8080"
        timeout (int): Request timeout in seconds.

    Returns:
        bool: True if the service is healthy, False otherwise.
    """
    if not base_url.startswith("http"):
        base_url = f"http://{base_url}"
    url = f"{base_url.rstrip('/')}/health"
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 200:
            return True
        else:
            return False
    except Exception:
        return False


def get_registered_number(router_url) -> dict:
    """
    Get the registered model counts by type from the router.

    Args:
        router_url (str): The base URL of the router, e.g. "http://localhost:8080".

    Returns:
        dict: A dictionary containing registered model counts with keys "mixed", "prefill", and "decode".
    """
    if not router_url.startswith("http"):
        router_url = f"http://{router_url}"

    try:
        response = requests.get(f"{router_url}/registered_number", timeout=60)
        registered_numbers = response.json()
        return registered_numbers
    except Exception:
        return {"mixed": 0, "prefill": 0, "decode": 0}
