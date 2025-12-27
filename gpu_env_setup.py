#!/usr/bin/env python3
"""
GPU 环境自动配置脚本
用于 Colab/Kaggle 等平台的一键配置
"""

import os
import sys
import subprocess
import platform

def check_system():
    """检查系统环境"""
    print("=== 系统环境检查 ===")

    print(f"Python版本: {sys.version}")
    print(f"系统: {platform.system()} {platform.release()}")

    # 检查 CUDA
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            cuda_version = result.stdout.split('release')[1].split(',')[0].strip()
            print(f"CUDA版本: {cuda_version}")
        else:
            print("CUDA: 未检测到")
    except:
        print("CUDA: 未安装")

    # 检查 GPU
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            gpu_info = result.stdout.split('\n')[1]
            print(f"GPU: {gpu_info}")
        else:
            print("GPU: 未检测到")
    except:
        print("GPU: 检测失败")

def install_paddle_gpu():
    """安装 PaddlePaddle GPU 版本"""
    print("\n=== 安装 PaddlePaddle GPU ===")

    # 检测 CUDA 版本并选择合适的 PaddlePaddle 版本
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        cuda_version = "cu126"  # 默认最新版本

        if "release 12.6" in result.stdout:
            cuda_version = "cu126"
        elif "release 12.4" in result.stdout:
            cuda_version = "cu124"
        elif "release 12.2" in result.stdout:
            cuda_version = "cu122"

        print(f"检测到 CUDA 版本，使用 {cuda_version}")

        # 安装命令
        commands = [
            "pip install --upgrade pip",
            f"pip install paddlepaddle-gpu==3.2.1 -f https://www.paddlepaddle.org.cn/packages/stable/{cuda_version}/",
            "pip install paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/cu126/ || echo 'nightly版本安装失败，使用stable版本'",
        ]

        for cmd in commands:
            print(f"执行: {cmd}")
            result = subprocess.run(cmd, shell=True)
            if result.returncode == 0:
                print("✅ 成功")
            else:
                print("❌ 失败")

    except Exception as e:
        print(f"自动检测失败，使用默认版本: {e}")
        subprocess.run("pip install paddlepaddle-gpu==3.2.1 -f https://www.paddlepaddle.org.cn/packages/stable/cu126/", shell=True)

def install_fastdeploy_deps():
    """安装 FastDeploy 依赖"""
    print("\n=== 安装 FastDeploy 依赖 ===")

    deps = [
        "pytest",
        "pytest-cov",
        "fastapi",
        "uvicorn",
        "redis",
        "pyzmq",
        "orjson",
        "opentelemetry-prometheus-client",
        "msgpack",
        "tqdm",
        "requests",
        "numpy",
    ]

    # 分批安装避免超时
    batch_size = 5
    for i in range(0, len(deps), batch_size):
        batch = deps[i:i+batch_size]
        cmd = f"pip install {' '.join(batch)}"
        print(f"安装: {cmd}")
        result = subprocess.run(cmd, shell=True)
        if result.returncode == 0:
            print("✅ 成功")
        else:
            print("❌ 失败")

def setup_fastdeploy_env():
    """设置 FastDeploy 环境变量"""
    print("\n=== 设置 FastDeploy 环境 ===")

    env_vars = {
        "FD_LOG_DIR": "/tmp/fastdeploy_logs",
        "MODEL_PATH": "/tmp/test_models",
        "FD_ENGINE_QUEUE_PORT": "6780",
        "FD_CACHE_QUEUE_PORT": "6781",
        "PYTHONPATH": f"{os.getcwd()}:$PYTHONPATH",
        "OMP_NUM_THREADS": "1",  # 避免多线程冲突
        "CUDA_VISIBLE_DEVICES": "0",  # 使用第一个GPU
    }

    # 创建日志目录
    os.makedirs("/tmp/fastdeploy_logs", exist_ok=True)
    os.makedirs("/tmp/test_models", exist_ok=True)

    # 设置环境变量
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"export {key}={value}")

    # 写入环境文件供后续使用
    with open("/tmp/fastdeploy_env.sh", "w") as f:
        f.write("#!/bin/bash\n")
        for key, value in env_vars.items():
            f.write(f"export {key}={value}\n")
        f.write("echo 'FastDeploy环境变量已设置'\n")

    print("✅ 环境变量设置完成")

def clone_and_install_fastdeploy():
    """克隆并安装 FastDeploy"""
    print("\n=== 克隆并安装 FastDeploy ===")

    commands = [
        "git clone https://github.com/PaddlePaddle/FastDeploy.git || echo '仓库已存在'",
        "cd FastDeploy",
        "git pull",  # 更新到最新
        "pip install -e . --no-build-isolation",
    ]

    for cmd in commands:
        print(f"执行: {cmd}")
        result = subprocess.run(cmd, shell=True)
        if result.returncode == 0:
            print("✅ 成功")
        else:
            print("❌ 失败")

def verify_installation():
    """验证安装"""
    print("\n=== 验证安装 ===")

    test_commands = [
        "python -c \"import paddle; print(f'PaddlePaddle版本: {paddle.__version__}')\"",
        "python -c \"import paddle; print(f'CUDA可用: {paddle.is_compiled_with_cuda()}')\"",
        "python -c \"import fastdeploy; print('FastDeploy导入成功')\"",
        "python -c \"from fastdeploy.engine.common_engine import EngineService; print('EngineService导入成功')\"",
    ]

    for cmd in test_commands:
        print(f"测试: {cmd}")
        result = subprocess.run(cmd, shell=True)
        if result.returncode == 0:
            print("✅ 通过")
        else:
            print("❌ 失败")

def create_test_script():
    """创建测试脚本"""
    print("\n=== 创建测试脚本 ===")

    test_script = '''#!/bin/bash
# FastDeploy GPU 测试脚本

echo "=== 开始 FastDeploy GPU 测试 ==="

# 加载环境变量
source /tmp/fastdeploy_env.sh

cd FastDeploy

echo "=== 运行基础测试 ==="
python -c "
import fastdeploy
from fastdeploy.engine.common_engine import EngineService
print('✅ 导入测试通过')
"

echo "=== 运行单元测试 ==="
python -m pytest tests/engine/test_common_engine.py -v --tb=short -x --disable-warnings

echo "=== 测试完成 ==="
'''

    with open("/tmp/run_fastdeploy_test.sh", "w") as f:
        f.write(test_script)

    os.chmod("/tmp/run_fastdeploy_test.sh", 0o755)
    print("✅ 测试脚本已创建: /tmp/run_fastdeploy_test.sh")

def main():
    """主函数"""
    print("🚀 FastDeploy GPU 环境自动配置")
    print("=" * 50)

    check_system()
    install_paddle_gpu()
    install_fastdeploy_deps()
    setup_fastdeploy_env()
    clone_and_install_fastdeploy()
    verify_installation()
    create_test_script()

    print("\n" + "=" * 50)
    print("🎉 配置完成！")
    print("运行测试: bash /tmp/run_fastdeploy_test.sh")
    print("=" * 50)

if __name__ == "__main__":
    main()



