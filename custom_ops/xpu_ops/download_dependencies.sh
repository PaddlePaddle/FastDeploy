#!/bin/bash

if [ $# -ne 1 ] || { [ "$1" != "stable" ] && [ "$1" != "develop" ]; }; then
    echo "Usage: $0 <stable|develop>"
    exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
THIRDPARTY_DIR="$SCRIPT_DIR/third_party"

rm -rf "$THIRDPARTY_DIR"
mkdir -p "$THIRDPARTY_DIR" || exit 1

if [ "$1" == "stable" ]; then
    version_xvllm="20251017"
    version_xtdk="3.4.0.1"
else
    version_xvllm="latest"
    version_xtdk="latest"
fi

(
    cd "$THIRDPARTY_DIR" || exit 1

    # Clean previous installation
    rm -rf output* xvllm* xtdk-llvm* output.tar.gz xtdk-llvm*tar.gz

    # Download and install xvllm
    if ! wget "https://klx-sdk-release-public.su.bcebos.com/xinfer/daily/eb/${version_xvllm}/output.tar.gz"; then
        echo "Error downloading xvllm"
        exit 2
    fi
    tar -zxf output.tar.gz && mv output xvllm && rm output.tar.gz

    # Download and install xtdk
    if ! wget "https://klx-sdk-release-public.su.bcebos.com/xtdk_15fusion/dev/${version_xtdk}/xtdk-llvm15-ubuntu2004_x86_64.tar.gz"; then
        echo "Error downloading xtdk"
        exit 3
    fi
    tar -zxf xtdk-llvm15-ubuntu2004_x86_64.tar.gz && \
    mv xtdk-llvm15-ubuntu2004_x86_64 xtdk && \
    rm xtdk-llvm15-ubuntu2004_x86_64.tar.gz
)

if [ $? -ne 0 ]; then
    echo "Installation failed"
    exit 4
fi

echo "Installation completed in: $THIRDPARTY_DIR"
echo "You can set environment variables as follows to use XVLLM and XTDK:"
echo " export CLANG_PATH=$THIRDPARTY_DIR/xtdk"
echo " export XVLLM_PATH=$THIRDPARTY_DIR/xvllm"
echo ""
