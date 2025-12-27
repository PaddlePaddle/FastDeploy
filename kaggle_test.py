#!/usr/bin/env python3
"""
Kaggle GPU 测试脚本
用于在 Kaggle 环境中测试 FastDeploy 引擎功能
"""

import sys
import os
import subprocess

def check_gpu():
    """检查 GPU"""
    print("=== 检查 GPU 环境 ===")

    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        print("✅ GPU 信息:")
        print(result.stdout.split('\n')[0:5])  # 显示前5行
        return True
    except:
        print("❌ GPU 不可用")
        return False

def setup_kaggle():
    """Kaggle 环境设置"""
    print("\n=== Kaggle 环境设置 ===")

    commands = [
        "pip install --upgrade pip",
        "pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple",
        # Kaggle 有预装的 GPU 驱动
        "pip install paddlepaddle-gpu==3.2.1 -f https://www.paddlepaddle.org.cn/packages/stable/cu126/",
        "pip install pytest fastapi uvicorn redis pyzmq orjson",
        "pip install opentelemetry-prometheus-client msgpack tqdm requests",
        # 克隆你的 fork
        "git clone https://github.com/YOUR_USERNAME/FastDeploy.git",  # 替换为你的用户名
        "cd FastDeploy",
        "export FD_LOG_DIR=/tmp/fastdeploy_logs",
        "mkdir -p $FD_LOG_DIR",
        "pip install -e . --no-build-isolation",
    ]

    for cmd in commands:
        print(f"执行: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ 失败: {result.stderr[:200]}...")
        else:
            print("✅ 成功")

def run_tests():
    """运行测试"""
    print("\n=== 运行测试 ===")

    os.chdir("FastDeploy")

    test_commands = [
        "python -c \"import fastdeploy; print('导入成功')\"",
        "python -m pytest tests/engine/test_common_engine.py -v --tb=short -x",
    ]

    for cmd in test_commands:
        print(f"\n执行: {cmd}")
        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            print("❌ 测试失败")
        else:
            print("✅ 测试成功")

if __name__ == "__main__":
    print("🖥️  Kaggle GPU 测试")
    print("=" * 40)

    check_gpu()
    setup_kaggle()
    run_tests()

    print("\n💡 提示: Kaggle notebook 会自动保存，30小时后会断开")



