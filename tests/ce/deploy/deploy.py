import ast
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import traceback

import requests
import yaml
from flask import Flask, Response, jsonify, request

current_dir = os.path.dirname(os.path.abspath(__file__))
tests_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))

sys.path.insert(0, tests_dir)

from e2e.utils.serving_utils import (
    FD_API_PORT,
    FD_CACHE_QUEUE_PORT,
    FD_ENGINE_QUEUE_PORT,
    FD_METRICS_PORT,
    clean_ports,
)

app = Flask(__name__)


def get_base_port():
    """获取base port"""
    nv_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not nv_visible_devices or nv_visible_devices.lower() == "all":
        return 8000
    # 提取第一个数字
    match = re.search(r"\d+", nv_visible_devices)
    if match:
        return int(match.group(0)) * 100 + 8000
    return 8000


def is_port_in_use(port):
    """检查端口是否被占用"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def get_available_port(env_key: str, default_start: int):
    """从环境变量读取端口，如果未设置或已被占用，则从default_start开始寻找空闲端口"""
    port_str = os.environ.get(env_key)
    if port_str and port_str.isdigit():
        port = int(port_str)
        if not is_port_in_use(port):
            return port
        else:
            print(f"Warning: Port {port} from {env_key} is in use, searching for a free port...")

    # 从 default_start 开始查找空闲端口
    port = default_start
    while is_port_in_use(port):
        port += 1
    return port


# 默认参数值
PID_FILE = "pid_port"
LOG_FILE = "server.log"
base_port = get_base_port()
FLASK_PORT = get_available_port("FLASK_PORT", base_port + 1)
DEFAULT_PARAMS = {
    "--port": FD_API_PORT,
    "--engine-worker-queue-port": FD_ENGINE_QUEUE_PORT,
    "--metrics-port": FD_METRICS_PORT,
    "--cache-queue-port": FD_CACHE_QUEUE_PORT,
    "--enable-logprob": True,
}


def build_command(config):
    """根据配置构建启动命令"""
    # 基础命令
    cmd = [
        "python",
        "-m",
        "fastdeploy.entrypoints.openai.api_server",
    ]

    # 添加配置参数
    for key, value in config.items():
        if "--enable" in key:
            value = bool(value if isinstance(value, bool) else eval(value))
            if value:
                cmd.append(key)
        else:
            cmd.extend([key, str(value)])

    return cmd


def merge_configs(base_config, override_config):
    """合并配置，优先级：override_config > base_config"""
    merged = base_config.copy()

    if override_config:
        for key in override_config:
            merged[key] = override_config[key]

    return merged


def get_server_pid():
    """获取服务进程ID PORT"""
    if os.path.exists(PID_FILE):
        with open(PID_FILE, "r") as f:
            data = yaml.safe_load(f)
            return data
    return None


def is_server_running():
    """检查服务是否正在运行"""
    pid_port = get_server_pid()
    if pid_port is None:
        return False, {"status": "Server not running..."}

    _, port = pid_port["PID"], pid_port["PORT"]
    health_check_endpoint = f"http://0.0.0.0:{port}/health"

    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            msg = f.readlines()
    result = parse_tqdm_progress(msg)

    try:
        response = requests.get(health_check_endpoint, timeout=2)
        return response.status_code == 200, result
    except requests.exceptions.RequestException as e:
        print(f"Failed to check server health: {e}")
        return False, result


def parse_tqdm_progress(log_lines):
    """
    解析 tqdm 风格的进度条
    """
    tqdm_pattern = re.compile(
        r"(?P<prefix>.+?):\s+(?P<percent>\d+)%\|(?P<bar>.+?)\|\s+(?P<step>\d+/\d+)\s+\[(?P<elapsed>\d+:\d+)<(?P<eta>\d+:\d+),\s+(?P<speed>[\d\.]+it/s)\]"
    )

    for line in reversed(log_lines):
        match = tqdm_pattern.search(line)
        if match:
            data = match.groupdict()
            return {
                "status": "服务启动中",
                "progress": {
                    "percent": int(data["percent"]),
                    "step": data["step"],
                    "speed": data["speed"],
                    "eta": data["eta"],
                    "elapsed": data["elapsed"],
                    "bar": data["bar"].strip(),
                },
                "raw_line": line.strip(),
            }
    return {"status": "服务启动中", "progress": {}, "raw_line": log_lines[-1] if log_lines else "server.log为空"}


def stop_server(signum=None, frame=None):
    """停止大模型推理服务"""
    pid_port = get_server_pid()
    if pid_port is None:
        if signum:
            sys.exit(0)
        return jsonify({"status": "error", "message": "Service is not running"}), 400

    server_pid, _ = pid_port["PID"], pid_port["PORT"]

    # 清理PID文件
    if os.path.exists(PID_FILE):
        os.remove(PID_FILE)
    if os.path.exists("gemm_profiles.json"):
        os.remove("gemm_profiles.json")

    try:
        clean_ports()
        # 终止进程组（包括所有子进程）
        os.killpg(os.getpgid(pid_port["PID"]), signal.SIGTERM)
    except Exception as e:
        print(f"Failed to stop server: {e}, {str(traceback.format_exc())}")
    try:
        result = subprocess.run(
            f"ps -efww | grep {FD_CACHE_QUEUE_PORT} | grep -v grep", shell=True, capture_output=True, text=True
        )
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split()
            pid = int(parts[1])
            print(f"Killing PID: {pid}")
            os.kill(pid, signal.SIGKILL)
    except Exception as e:
        print(f"Failed to kill cache manager process: {e}, {str(traceback.format_exc())}")

    for port in [FD_API_PORT, FD_ENGINE_QUEUE_PORT, FD_METRICS_PORT, FD_CACHE_QUEUE_PORT]:
        try:
            output = subprocess.check_output(f"lsof -i:{port} -t", shell=True).decode().strip()
            for pid in output.splitlines():
                os.kill(int(pid), signal.SIGKILL)
                print(f"Killed process on port {port}, pid={pid}")
        except Exception as e:
            print(f"Failed to kill process on port: {e}, {str(traceback.format_exc())}")
    # 若log目录存在，则重命名为log_timestamp
    if os.path.isdir("./log"):
        os.rename("./log", "./log_{}".format(time.strftime("%Y%m%d%H%M%S")))
    if os.path.exists("gemm_profiles.json"):
        os.remove("gemm_profiles.json")

    if signum:
        sys.exit(0)

    return jsonify({"status": "success", "message": "Service stopped", "pid": server_pid}), 200


# 捕获 SIGINT (Ctrl+C) 和 SIGTERM (kill)
signal.signal(signal.SIGINT, stop_server)
signal.signal(signal.SIGTERM, stop_server)


@app.route("/start", methods=["POST"])
def start_service():
    """启动大模型推理服务"""
    # 检查服务是否已在运行
    if is_server_running()[0]:
        return Response(
            json.dumps({"status": "error", "message": "服务已启动，无需start"}, ensure_ascii=False),
            status=400,
            content_type="application/json",
        )

    try:
        base_config = DEFAULT_PARAMS

        override_config = request.get_json() or {}
        print("override_config", override_config)

        final_config = merge_configs(base_config, override_config)

        global FD_API_PORT
        global FD_ENGINE_QUEUE_PORT
        global FD_METRICS_PORT
        FD_API_PORT = final_config["--port"]
        FD_ENGINE_QUEUE_PORT = final_config["--engine-worker-queue-port"]
        FD_METRICS_PORT = final_config["--metrics-port"]

        # 构建命令
        cmd = build_command(final_config)
    except Exception as e:
        error_msg = f"Failed to start service: {e}, {str(traceback.format_exc())}"
        print(error_msg)
        return Response(
            json.dumps({"status": "error", "message": error_msg}, ensure_ascii=False),
            status=500,
            content_type="application/json",
        )

    print("cmd", cmd)

    try:
        # 设置环境变量并启动进程
        env = os.environ.copy()

        with open(LOG_FILE, "w") as log:
            process = subprocess.Popen(cmd, stdout=log, stderr=log, env=env, start_new_session=True)

        # 保存进程ID,port到yaml文件
        with open(PID_FILE, "w") as f:
            yaml.dump({"PID": process.pid, "PORT": final_config["--port"]}, f)

        json_data = {
            "status": "success",
            "message": "服务启动命令已执行",
            "pid": process.pid,
            "config": final_config,
            "log_file": LOG_FILE,
            "cmd": cmd,
            "port_info": {
                "api_port": FD_API_PORT,
                "queue_port": FD_ENGINE_QUEUE_PORT,
                "metrics_port": FD_METRICS_PORT,
            },
        }

        return Response(json.dumps(json_data, ensure_ascii=False), status=200, content_type="application/json")
    except Exception as e:
        error_msg = f"Failed to start service: {e}, {str(traceback.format_exc())}"
        print(error_msg)
        return Response(
            json.dumps({"status": "error", "message": error_msg}, ensure_ascii=False),
            status=500,
            content_type="application/json",
        )


@app.route("/switch", methods=["POST"])
def switch_service():
    """切换模型服务"""
    # kill掉已有服务
    stop_server()
    time.sleep(10)

    try:
        base_config = DEFAULT_PARAMS

        override_config = request.get_json() or {}

        final_config = merge_configs(base_config, override_config)

        global FD_API_PORT
        global FD_ENGINE_QUEUE_PORT
        global FD_METRICS_PORT
        FD_API_PORT = final_config["--port"]
        FD_ENGINE_QUEUE_PORT = final_config["--engine-worker-queue-port"]
        FD_METRICS_PORT = final_config["--metrics-port"]

        # 构建命令
        cmd = build_command(final_config)
    except Exception as e:
        error_msg = f"Failed to switch service: {e}, {str(traceback.format_exc())}"
        print(error_msg)
        return Response(
            json.dumps({"status": "error", "message": error_msg}, ensure_ascii=False),
            status=500,
            content_type="application/json",
        )

    print("cmd", cmd)

    try:
        # 设置环境变量并启动进程
        env = os.environ.copy()

        with open(LOG_FILE, "w") as log:
            process = subprocess.Popen(cmd, stdout=log, stderr=log, env=env, start_new_session=True)

        # 保存进程ID,port到yaml文件
        with open(PID_FILE, "w") as f:
            yaml.dump({"PID": process.pid, "PORT": final_config["--port"]}, f)

        json_data = {
            "status": "success",
            "message": "服务启动命令已执行",
            "pid": process.pid,
            "config": final_config,
            "log_file": LOG_FILE,
            "cmd": cmd,
            "port_info": {
                "api_port": FD_API_PORT,
                "queue_port": FD_ENGINE_QUEUE_PORT,
                "metrics_port": FD_METRICS_PORT,
            },
        }

        return Response(json.dumps(json_data, ensure_ascii=False), status=200, content_type="application/json")
    except Exception as e:
        error_msg = f"Failed to switch service: {e}, {str(traceback.format_exc())}"
        print(error_msg)
        return Response(
            json.dumps({"status": "error", "message": error_msg}, ensure_ascii=False),
            status=500,
            content_type="application/json",
        )


@app.route("/status", methods=["GET", "POST"])
def service_status():
    """检查服务状态"""
    health, msg = is_server_running()

    if not health:
        return Response(json.dumps(msg, ensure_ascii=False), status=500, content_type="application/json")

    # 检查端口是否监听
    ports_status = {
        "api_port": FD_API_PORT if is_port_in_use(FD_API_PORT) else None,
        "queue_port": FD_ENGINE_QUEUE_PORT if is_port_in_use(FD_ENGINE_QUEUE_PORT) else None,
        "metrics_port": FD_METRICS_PORT if is_port_in_use(FD_METRICS_PORT) else None,
    }

    msg["status"] = "服务启动完成"
    msg["ports_status"] = ports_status

    return Response(json.dumps(msg, ensure_ascii=False), status=200, content_type="application/json")


@app.route("/stop", methods=["POST"])
def stop_service():
    """停止大模型推理服务"""
    res, status_code = stop_server()

    return res, status_code


@app.route("/config", methods=["GET"])
def get_config():
    """获取当前server配置"""
    health, msg = is_server_running()

    if not health:
        return Response(json.dumps(msg, ensure_ascii=False), status=500, content_type="application/json")

    if not os.path.exists("log/api_server.log"):
        return Response(
            json.dumps({"message": "api_server.log不存在"}, ensure_ascii=False),
            status=500,
            content_type="application/json",
        )

    try:
        # 筛选出包含"args:"的行
        with open("log/api_server.log", "r") as f:
            lines = [line for line in f.readlines() if "args:" in line]

        last_line = lines[-1] if lines else ""

        # 使用正则表达式提取JSON格式的配置
        match = re.search(r"args\s*[:：]\s*(.*)", last_line)
        if not match:
            return Response(
                json.dumps({"message": "api_server.log中没有args信息，请检查log"}, ensure_ascii=False),
                status=500,
                content_type="application/json",
            )

        # 尝试解析JSON
        config_json = match.group(1).strip()
        config_data = ast.literal_eval(config_json)
        print("config_data", config_data, type(config_data))
        return Response(
            json.dumps({"server_config": config_data}, ensure_ascii=False), status=200, content_type="application/json"
        )

    except Exception as e:
        error_msg = f"{e}, {str(traceback.format_exc())}"
        print(error_msg)
        return Response(
            json.dumps({"message": "api_server.log解析失败，请检查log", "error": error_msg}, ensure_ascii=False),
            status=500,
            content_type="application/json",
        )


@app.route("/wait_for_infer", methods=["POST"])
def wait_for_infer():
    timeout = int(request.args.get("timeout", 120))  # 可选超时时间，默认120秒
    interval = 2
    response_interval = 10
    start_time = time.time()
    next_response_time = start_time

    def generate():
        nonlocal next_response_time
        while True:
            health, msg = is_server_running()
            now = time.time()

            elapsed = time.time() - start_time

            if health:
                ports_status = {
                    "api_port": FD_API_PORT if is_port_in_use(FD_API_PORT) else None,
                    "queue_port": FD_ENGINE_QUEUE_PORT if is_port_in_use(FD_ENGINE_QUEUE_PORT) else None,
                    "metrics_port": FD_METRICS_PORT if is_port_in_use(FD_METRICS_PORT) else None,
                }
                msg["status"] = "服务启动完成"
                msg["ports_status"] = ports_status
                yield json.dumps(msg, ensure_ascii=False) + "\n"
                break

            if elapsed >= timeout:

                def tail_file(path, lines=50):
                    try:
                        with open(path, "r", encoding="utf-8", errors="ignore") as f:
                            return "".join(f.readlines()[-lines:])
                    except Exception as e:
                        return f"[无法读取 {path}]: {e}, {str(traceback.format_exc())}\n"

                result = f"服务启动超时，耗时：[{timeout}s]\n\n"
                result += "==== server.log tail 50 ====\n"
                result += tail_file("server.log")
                result += "\n==== log/workerlog.0 tail 50 ====\n"
                result += tail_file("log/workerlog.0")

                yield result
                break

            if now >= next_response_time:
                msg["status"] = f"服务启动中，耗时：[{int(elapsed)}s]"
                yield json.dumps(msg, ensure_ascii=False) + "\n"
                next_response_time += response_interval

            time.sleep(interval)

    return Response(generate(), status=200, content_type="text/plain")


if __name__ == "__main__":
    print(f"FLASK_PORT: {FLASK_PORT}")
    print(f"FD_API_PORT: {FD_API_PORT}")
    print(f"FD_ENGINE_QUEUE_PORT: {FD_ENGINE_QUEUE_PORT}")
    print(f"FD_METRICS_PORT: {FD_METRICS_PORT}")
    app.run(host="0.0.0.0", port=FLASK_PORT, debug=False)
