#!/bin/bash
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

# XPU CI Test Runner Script
# This script sets up the environment and runs XPU CI tests using pytest.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
echo "Project root: $PROJECT_ROOT"

# Change to project root directory
cd "${PROJECT_ROOT}"

# =============================================================================
# Environment Setup
# =============================================================================

setup_environment() {
    echo "============================设置环境!============================"

    # Install lsof tool
    apt install -y lsof 2>/dev/null || true

    # Set XPU visible devices based on XPU_ID
    if [[ "$XPU_ID" == "0" ]]; then
        export XPU_VISIBLE_DEVICES="0,1,2,3"
    else
        export XPU_VISIBLE_DEVICES="4,5,6,7"
    fi

    # Download and setup XRE
    mkdir -p /workspace/deps
    cd /workspace/deps
    if [ ! -f "xre-Linux-x86_64-5.0.21.21.tar.gz" ]; then
        wget -q https://klx-sdk-release-public.su.bcebos.com/xre/kl3-release/5.0.21.21/xre-Linux-x86_64-5.0.21.21.tar.gz
        tar -zxf xre-Linux-x86_64-5.0.21.21.tar.gz && mv xre-Linux-x86_64-5.0.21.21 xre
    fi
    cd -
    export PATH=/workspace/deps/xre/bin:$PATH

    # Reset XPU devices
    xpu-smi -r -i $XPU_VISIBLE_DEVICES
    xpu-smi
}

install_dependencies() {
    echo "============================安装依赖!============================"

    # Install requirements
    python -m pip install -r requirements.txt

    # Uninstall old versions
    python -m pip uninstall paddlepaddle-xpu -y || true
    python -m pip uninstall fastdeploy-xpu -y || true

    # Install paddlepaddle-xpu
    python -m pip install https://paddle-whl.bj.bcebos.com/nightly/xpu-p800/paddlepaddle-xpu/paddlepaddle_xpu-3.3.0.dev20251118-cp310-cp310-linux_x86_64.whl

    # Build FastDeploy
    echo "============================构建FastDeploy!============================"
    bash custom_ops/xpu_ops/download_dependencies.sh develop
    export CLANG_PATH=$(pwd)/custom_ops/xpu_ops/third_party/xtdk
    export XVLLM_PATH=$(pwd)/custom_ops/xpu_ops/third_party/xvllm
    bash build.sh || exit 1

    # Install test dependencies
    python -m pip install openai -U
    python -m pip uninstall -y triton || true
    python -m pip install triton==3.3.0
    python -m pip install pytest pytest-timeout

    # Unset proxy settings
    unset http_proxy
    unset https_proxy
    unset no_proxy
}

setup_xdeepep() {
    echo "============================设置xDeepEP!============================"

    if [ ! -d "xDeepEP" ]; then
        wget -q https://paddle-qa.bj.bcebos.com/xpu_third_party/xDeepEP.tar.gz
        tar -xzf xDeepEP.tar.gz
        cd xDeepEP
        bash build.sh
        cd -
    fi
}

# =============================================================================
# Test Running Functions
# =============================================================================

run_test() {
    local test_marker=$1
    local test_name=$2

    echo "============================开始 ${test_name} 测试!============================"
    echo "当前工作目录: $(pwd)"
    echo "测试路径: tests/xpu_ci/cases/"
    echo "测试 marker: ${test_marker}"

    python -m pytest tests/xpu_ci/cases/ \
        -m "${test_marker}" \
        --model-path "${MODEL_PATH}" \
        --xpu-id "${XPU_ID:-0}" \
        -v -s

    local exit_code=$?
    echo "pytest 退出码: ${exit_code}"

    # pytest exit codes:
    # 0: All tests passed
    # 1: Tests were collected and run but some failed
    # 2: Test execution was interrupted by the user
    # 3: Internal error happened while executing tests
    # 4: pytest command line usage error
    # 5: No tests were collected

    if [ ${exit_code} -eq 5 ]; then
        echo "警告: 没有找到 ${test_name} 的测试用例 (marker: ${test_marker})"
        echo "请检查测试文件是否存在以及 marker 是否正确"
        return 1
    elif [ ${exit_code} -ne 0 ]; then
        echo "${test_name} 测试失败，退出码: ${exit_code}"
        echo "请检查上面的测试输出"
        return 1
    fi

    echo "${test_name} 测试通过!"
    return 0
}

run_all_tests() {
    local failed_tests=()

    # V1 Mode Test
    if ! run_test "v1_mode" "V1模式"; then
        failed_tests+=("V1模式")
    fi
    sleep 5

    # W4A8 Test
    if ! run_test "w4a8" "W4A8"; then
        failed_tests+=("W4A8")
    fi
    sleep 5

    # VL Model Test
    if ! run_test "vl_model" "VL模型"; then
        failed_tests+=("VL模型")
    fi
    sleep 5

    # Setup xDeepEP for Expert Parallel tests
    setup_xdeepep

    # EP4TP4 Test
    if ! run_test "ep4tp4" "EP4TP4"; then
        failed_tests+=("EP4TP4")
    fi
    sleep 5

    # EP4TP1 Test
    if ! run_test "ep4tp1" "EP4TP1"; then
        failed_tests+=("EP4TP1")
    fi
    sleep 5

    # EP4TP4 all2all Test
    if ! run_test "all2all" "EP4TP4 all2all"; then
        failed_tests+=("EP4TP4 all2all")
    fi

    # Report results
    echo ""
    echo "============================测试结果汇总!============================"
    if [ ${#failed_tests[@]} -eq 0 ]; then
        echo "所有测试通过!"
        return 0
    else
        echo "失败的测试: ${failed_tests[*]}"
        return 1
    fi
}

run_single_test() {
    local marker=$1

    case $marker in
        v1|v1_mode)
            run_test "v1_mode" "V1模式"
            ;;
        w4a8)
            run_test "w4a8" "W4A8"
            ;;
        vl|vl_model)
            run_test "vl_model" "VL模型"
            ;;
        ep4tp4)
            setup_xdeepep
            run_test "ep4tp4" "EP4TP4"
            ;;
        ep4tp1)
            setup_xdeepep
            run_test "ep4tp1" "EP4TP1"
            ;;
        all2all)
            setup_xdeepep
            run_test "all2all" "EP4TP4 all2all"
            ;;
        ep|expert_parallel)
            setup_xdeepep
            run_test "expert_parallel" "Expert Parallel"
            ;;
        *)
            echo "未知的测试类型: $marker"
            echo "可用的测试类型: v1, w4a8, vl, ep4tp4, ep4tp1, all2all, ep"
            return 1
            ;;
    esac
}

# =============================================================================
# Main Script
# =============================================================================

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --setup-only     Only setup environment, don't run tests"
    echo "  --test <marker>  Run specific test (v1, w4a8, vl, ep4tp4, ep4tp1, all2all, ep)"
    echo "  --all            Run all tests (default)"
    echo "  --help           Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  MODEL_PATH       Path to model files (required)"
    echo "  XPU_ID           XPU device ID (0 or 1, default: 0)"
}

main() {
    local setup_only=false
    local test_marker=""
    local run_all=true

    while [[ $# -gt 0 ]]; do
        case $1 in
            --setup-only)
                setup_only=true
                shift
                ;;
            --test)
                test_marker=$2
                run_all=false
                shift 2
                ;;
            --all)
                run_all=true
                shift
                ;;
            --help)
                print_usage
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                print_usage
                exit 1
                ;;
        esac
    done

    # Check MODEL_PATH
    if [ -z "${MODEL_PATH}" ]; then
        echo "Error: MODEL_PATH environment variable is not set"
        exit 1
    fi

    # Setup environment
    setup_environment
    install_dependencies

    if [ "$setup_only" = true ]; then
        echo "环境设置完成!"
        exit 0
    fi

    # Run tests
    if [ -n "$test_marker" ]; then
        run_single_test "$test_marker"
    elif [ "$run_all" = true ]; then
        run_all_tests
    fi
}

main "$@"
