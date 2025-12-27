#!/usr/bin/env python3
"""
Google Colab GPU 测试脚本
用于在 Colab 环境中测试 FastDeploy 引擎功能
"""

import sys
import os
import subprocess
import time

def check_gpu():
    """检查 GPU 可用性"""
    print("=== 检查 GPU 环境 ===")

    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ GPU 可用:")
            print(result.stdout.split('\n')[0])  # 只显示第一行
            return True
        else:
            print("❌ GPU 不可用")
            return False
    except FileNotFoundError:
        print("❌ nvidia-smi 命令不存在")
        return False

def install_dependencies():
    """安装依赖"""
    print("\n=== 安装依赖 ===")

    commands = [
        "pip install --upgrade pip",
        "pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple",
        # 安装 PaddlePaddle GPU 版本
        "pip install paddlepaddle-gpu==3.2.1 -f https://www.paddlepaddle.org.cn/packages/stable/cu126/",
        # 安装其他依赖
        "pip install pytest pytest-cov fastapi uvicorn redis pyzmq orjson",
        "pip install opentelemetry-prometheus-client msgpack tqdm requests",
    ]

    for cmd in commands:
        print(f"执行: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ 失败: {result.stderr}")
        else:
            print("✅ 成功")

def clone_and_setup():
    """克隆代码并设置"""
    print("\n=== 克隆代码并设置 ===")

    commands = [
        "git clone https://github.com/PaddlePaddle/FastDeploy.git",
        "cd FastDeploy",
        "export FD_LOG_DIR=/tmp/fastdeploy_logs",
        "mkdir -p $FD_LOG_DIR",
        "pip install -e . --no-build-isolation",
    ]

    for cmd in commands:
        print(f"执行: {cmd}")
        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            print(f"❌ 失败: {cmd}")
            return False
        else:
            print("✅ 成功")

    return True

def run_basic_tests():
    """运行基础测试"""
    print("\n=== 运行基础测试 ===")

    os.chdir("FastDeploy")

    # 测试导入
    test_commands = [
        "python -c \"import fastdeploy; print('FastDeploy 导入成功')\"",
        "python -c \"from fastdeploy.engine.common_engine import EngineService; print('EngineService 导入成功')\"",
        # 运行简单的单元测试
        "python -m pytest tests/engine/test_common_engine.py::TestCommonEngine::test_initialization -v --tb=short",
    ]

    for cmd in test_commands:
        print(f"\n执行: {cmd}")
        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            print(f"❌ 测试失败")
        else:
            print("✅ 测试成功")

def main():
    print("🚀 FastDeploy Colab GPU 测试")
    print("=" * 50)

    # 检查 GPU
    if not check_gpu():
        print("⚠️  没有 GPU，某些测试可能无法运行")
        return

    # 安装依赖
    install_dependencies()

    # 克隆和设置
    if not clone_and_setup():
        print("❌ 设置失败")
        return

    # 运行测试
    run_basic_tests()

    print("\n" + "=" * 50)
    print("🎉 测试完成！")
    print("注意: Colab 的 GPU 会话有时限，请及时保存结果")

if __name__ == "__main__":
    main()



