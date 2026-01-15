# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
XPU CI测试框架 - 通用配置和辅助函数

这个文件包含了所有测试case共用的函数和fixture。
主要功能:
1. 进程管理 - 启动和停止API服务器
2. 健康检查 - 等待服务启动成功
3. 资源清理 - 清理日志、core文件、消息队列等
4. 环境配置 - 设置XPU相关环境变量
"""

import json
import os
import shutil
import subprocess
import time

import pytest


def get_xpu_id():
    """获取XPU_ID环境变量"""
    return int(os.getenv("XPU_ID", "0"))


def get_port_num():
    """根据XPU_ID计算端口号"""
    xpu_id = get_xpu_id()
    return 8188 + xpu_id * 100


def stop_processes():
    """
    停止所有相关进程（最小改动版，避免误杀 pytest）
    """
    xpu_id = get_xpu_id()  # noqa: F841
    port_num = get_port_num()

    # 获取 pytest 主进程 PID
    try:
        pytest_pids = subprocess.check_output("pgrep -f pytest || true", shell=True).decode().strip().split()
    except subprocess.CalledProcessError:
        pytest_pids = []

    def safe_kill_cmd(cmd):
        """执行 kill 命令，但排除 pytest 进程"""
        try:
            # 先执行命令获取到候选 PID（kill -9 替换成 cat）
            list_cmd = cmd.replace("kill -9", "cat")
            output = subprocess.check_output(list_cmd, shell=True, stderr=subprocess.DEVNULL).decode().strip().split()

            # 过滤：排除 pytest
            safe_pids = [pid for pid in output if pid and pid not in pytest_pids]

            # 真正 kill
            for pid in safe_pids:
                subprocess.run(f"kill -9 {pid}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

    commands = [
        "ps -efww | grep -E 'cache_transfer_manager.py' | grep -v grep | awk '{print $2}' | xargs echo",
        "ps -efww | grep -E 'api_server' | grep -v grep | awk '{print $2}' | xargs echo",
        "ps -efww | grep -E 'multiprocessing' | grep -v grep | awk '{print $2}' | xargs echo",
        "ps -efww | grep -E 'fastdeploy' | grep -v grep | awk '{print $2}' | xargs echo",
        "ps -efww | grep -E 'gunicorn: master' | grep -v grep | awk '{print $2}' | xargs echo",
        "ps -efww | grep -E 'gunicorn: worker' | grep -v grep | awk '{print $2}' | xargs echo",
        f"ps -efww | grep -E '{port_num}' | grep -v grep | awk '{{print $2}}' | xargs echo",
        f"lsof -t -i :{port_num} | xargs echo",
        f"lsof -t -i :{port_num + 47873} | xargs echo",
    ]

    # Kill additional ports
    for port in range(port_num + 10, port_num + 41):
        commands.append(f"lsof -t -i :{port} | xargs echo")

    # Kill processes using netstat
    commands.extend(
        [
            f"netstat -tunlp 2>/dev/null | grep {port_num + 2} | awk '{{print $NF}}' | awk -F'/' '{{print $1}}' | xargs echo",
            f"netstat -tunlp 2>/dev/null | grep {port_num + 2} | awk '{{print $(NF-1)}}' | cut -d/ -f1 | grep -E '^[0-9]+$' | xargs echo",
        ]
    )

    for cmd in commands:
        safe_kill_cmd(cmd)


def cleanup_resources():
    """
    清理资源

    包括:
    1. 删除log目录
    2. 删除core文件
    3. 清空消息队列
    """
    # 删除log目录
    if os.path.exists("log"):
        shutil.rmtree("log")

    # 删除core文件
    subprocess.run("rm -f core*", shell=True)

    # 清空消息队列
    subprocess.run(
        "ipcrm --all=msg 2>/dev/null || true", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


def wait_for_health_check(timeout=900, interval=10):
    """
    等待服务健康检查通过

    Args:
        timeout: 超时时间(秒), 默认15分钟
        interval: 检查间隔(秒), 默认10秒

    Returns:
        bool: 服务是否启动成功
    """
    port_num = get_port_num()
    health_endpoint = f"http://0.0.0.0:{port_num}/health"
    models_endpoint = f"http://0.0.0.0:{port_num}/v1/models"
    start_time = time.time()

    print(f"开始服务健康检查,最长等待时间:{timeout}秒")

    # 第一阶段: 等待 /health 返回 200
    while True:
        elapsed = int(time.time() - start_time)

        # 超时判断
        if elapsed >= timeout:
            print(f"\n服务启动超时:经过 {timeout//60} 分钟服务仍未启动!")
            return False

        # 发送健康检查请求
        try:
            result = subprocess.run(
                f'curl -s -o /dev/null -w "%{{http_code}}" -m 2 {health_endpoint}',
                shell=True,
                capture_output=True,
                text=True,
            )
            http_code = result.stdout.strip()
        except Exception:
            http_code = "000"

        print(f"\r服务健康检查中... 已等待 {elapsed} 秒,当前状态码:{http_code}", end="", flush=True)

        if http_code == "200":
            print(f"\n健康检查通过!耗时 {elapsed} 秒")
            break

        time.sleep(interval)

    # 第二阶段: 等待 /v1/models 返回有效模型列表,确保模型完全就绪
    print("开始验证模型是否就绪...")
    while True:
        elapsed = int(time.time() - start_time)

        # 超时判断
        if elapsed >= timeout:
            print(f"\n模型就绪超时:经过 {timeout//60} 分钟模型仍未就绪!")
            return False

        # 检查模型列表
        try:
            result = subprocess.run(f"curl -s -m 5 {models_endpoint}", shell=True, capture_output=True, text=True)
            response = result.stdout.strip()
            if response:
                data = json.loads(response)
                # 检查是否有模型数据
                if data.get("data") and len(data["data"]) > 0:
                    model_id = data["data"][0].get("id", "unknown")
                    print(f"\n模型就绪!模型ID: {model_id}, 总耗时 {elapsed} 秒")
                    return True
        except (json.JSONDecodeError, Exception) as e:  # noqa: F841
            pass

        print(f"\r等待模型就绪中... 已等待 {elapsed} 秒", end="", flush=True)
        time.sleep(interval)


def print_logs_on_failure():
    """失败时打印日志"""
    print("\n========== server.log ==========")
    if os.path.exists("server.log"):
        with open("server.log", "r") as f:
            print(f.read())

    print("\n========== log/workerlog.0 ==========")
    if os.path.exists("log/workerlog.0"):
        with open("log/workerlog.0", "r") as f:
            print(f.read())


def start_server(server_args, wait_before_check=60):
    """
    启动API服务器

    Args:
        server_args: 服务器启动参数列表
        wait_before_check: 启动后等待多少秒再进行健康检查,默认60秒

    Returns:
        bool: 服务是否启动成功
    """
    # 停止旧进程
    stop_processes()

    # 清理资源
    cleanup_resources()

    # 构建启动命令
    cmd = ["python", "-m", "fastdeploy.entrypoints.openai.api_server"] + server_args

    # 启动服务(后台运行)
    with open("server.log", "w") as log_file:
        subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True)

    print(f"服务启动命令: {' '.join(cmd)}")
    print(f"等待 {wait_before_check} 秒...")
    time.sleep(wait_before_check)

    # 健康检查
    if not wait_for_health_check():
        print_logs_on_failure()
        stop_processes()
        return False

    return True


@pytest.fixture(scope="function")
def xpu_env():
    """
    设置XPU环境变量

    这个fixture会在每个测试开始时设置XPU_VISIBLE_DEVICES环境变量
    测试结束后自动清理
    """
    xpu_id = get_xpu_id()

    # 设置XPU_VISIBLE_DEVICES
    if xpu_id == 0:
        os.environ["XPU_VISIBLE_DEVICES"] = "0,1,2,3"
    else:
        os.environ["XPU_VISIBLE_DEVICES"] = "4,5,6,7"

    print(f"\n设置环境变量: XPU_VISIBLE_DEVICES={os.environ['XPU_VISIBLE_DEVICES']}")

    yield

    # 测试结束后停止进程
    print("\n测试结束,停止服务...")
    stop_processes()


def get_model_path():
    """获取MODEL_PATH环境变量"""
    model_path = os.getenv("MODEL_PATH")
    if not model_path:
        raise ValueError("MODEL_PATH environment variable is not set")
    return model_path


def setup_ep_env():
    """
    设置EP(Expert Parallel)相关环境变量

    Returns:
        dict: 原始环境变量值,用于后续恢复
    """
    env_vars = {
        "BKCL_ENABLE_XDR": "1",
        "BKCL_RDMA_NICS": "eth1,eth1,eth2,eth2",
        "BKCL_TRACE_TOPO": "1",
        "BKCL_PCIE_RING": "1",
        "XSHMEM_MODE": "1",
        "XSHMEM_QP_NUM_PER_RANK": "32",
        "BKCL_RDMA_VERBS": "1",
        "MOE_FFN_USE_DENSE_INPUT": "1",
    }

    # 保存原始值
    original_values = {}
    for key in env_vars:
        original_values[key] = os.environ.get(key)

    # 设置新值
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"设置环境变量: {key}={value}")

    # 设置BKCL_RDMA_NICS
    rdma_nics = get_rdma_nics()
    if rdma_nics:
        os.environ["BKCL_RDMA_NICS"] = rdma_nics
        print(f"设置环境变量: BKCL_RDMA_NICS={rdma_nics}")
    return original_values


def restore_env(original_values):
    """
    恢复环境变量

    Args:
        original_values: setup_ep_env()返回的原始环境变量值
    """
    for key, value in original_values.items():
        if value is None:
            if key in os.environ:
                del os.environ[key]
                print(f"删除环境变量: {key}")
        else:
            os.environ[key] = value
            print(f"恢复环境变量: {key}={value}")


def download_and_build_xdeepep():
    """下载并编译xDeepEP(用于EP并行测试)"""
    if os.path.exists("xDeepEP"):
        print("xDeepEP已存在,跳过下载")
        return True

    print("下载xDeepEP...")
    result = subprocess.run("wget -q https://paddle-qa.bj.bcebos.com/xpu_third_party/xDeepEP.tar.gz", shell=True)
    if result.returncode != 0:
        print("下载xDeepEP失败")
        return False

    print("解压xDeepEP...")
    result = subprocess.run("tar -xzf xDeepEP.tar.gz", shell=True)
    if result.returncode != 0:
        print("解压xDeepEP失败")
        return False

    print("编译xDeepEP...")
    result = subprocess.run("cd xDeepEP && bash build.sh && cd -", shell=True)
    if result.returncode != 0:
        print("编译xDeepEP失败")
        return False

    return True


# ============ PD分离相关函数 ============


def get_script_dir():
    """获取scripts目录路径"""
    # conftest.py在tests/xpu_ci_pytest/下,scripts在项目根目录下
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    return os.path.join(project_root, "scripts")


def get_rdma_nics():
    """
    获取RDMA网卡配置

    Returns:
        str: KVCACHE_RDMA_NICS的值,失败返回空字符串
    """
    script_path = os.path.join(get_script_dir(), "get_rdma_nics.sh")

    try:
        result = subprocess.run(f"bash {script_path} xpu", shell=True, capture_output=True, text=True)
        output = result.stdout.strip()
        # 解析 KVCACHE_RDMA_NICS=xxx 格式
        if output.startswith("KVCACHE_RDMA_NICS="):
            return output.split("=", 1)[1]
        return output
    except Exception as e:
        print(f"获取RDMA网卡失败: {e}")
        return ""


def setup_pd_env():
    """
    设置PD分离相关环境变量

    Returns:
        dict: 原始环境变量值,用于后续恢复
    """
    original_values = {}
    env_keys = ["KVCACHE_GDRCOPY_FLUSH_ENABLE", "KVCACHE_RDMA_NICS", "CUDA_ENABLE_P2P_NO_UVA"]

    # 保存原始值
    for key in env_keys:
        original_values[key] = os.environ.get(key)

    # 设置新值
    os.environ["KVCACHE_GDRCOPY_FLUSH_ENABLE"] = "1"
    os.environ["CUDA_ENABLE_P2P_NO_UVA"] = "1"  # 开启peer mem
    print("设置环境变量: KVCACHE_GDRCOPY_FLUSH_ENABLE=1")
    print("设置环境变量: CUDA_ENABLE_P2P_NO_UVA=1")

    # 获取并设置RDMA网卡
    rdma_nics = get_rdma_nics()
    if rdma_nics:
        os.environ["KVCACHE_RDMA_NICS"] = rdma_nics
        print(f"设置环境变量: KVCACHE_RDMA_NICS={rdma_nics}")

    return original_values


def restore_pd_env(original_values):
    """
    恢复PD分离相关环境变量

    Args:
        original_values: setup_pd_env()返回的原始环境变量值
    """
    env_keys = ["KVCACHE_GDRCOPY_FLUSH_ENABLE", "KVCACHE_RDMA_NICS", "CUDA_ENABLE_P2P_NO_UVA"]

    for key in env_keys:
        if key in original_values:
            if original_values[key] is None:
                if key in os.environ:
                    del os.environ[key]
                    print(f"删除环境变量: {key}")
            else:
                os.environ[key] = original_values[key]
                print(f"恢复环境变量: {key}={original_values[key]}")


def setup_pd_ep_env():
    """
    设置PD分离+EP相关环境变量

    Returns:
        dict: 原始环境变量值,用于后续恢复
    """
    original_values_pd = setup_pd_env()
    original_values_ep = setup_ep_env()
    original_values = {**original_values_pd, **original_values_ep}
    return original_values


def restore_pd_ep_env(original_values):
    """
    恢复PD分离相关环境变量

    Args:
        original_values: setup_ep_env()返回的原始环境变量值
    """
    restore_env(original_values)
    restore_pd_env(original_values)


def setup_logprobs_env():
    """
    设置logprobs相关环境变量

    Returns:
        dict: 原始环境变量值,用于后续恢复
    """
    env_vars = {
        "FD_USE_GET_SAVE_OUTPUT_V1": "1",
    }
    os.system("sysctl -w kernel.msgmax=131072")
    os.system("sysctl -w kernel.msgmnb=33554432")

    # 保存原始值
    original_values = {}
    for key in env_vars:
        original_values[key] = os.environ.get(key)

    # 设置新值
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"设置环境变量: {key}={value}")
    return original_values
