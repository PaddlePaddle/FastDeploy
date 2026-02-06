#!/bin/bash
set -e

pip install -U "paddleocr[doc-parser]"
pip install opencv-contrib-python-headless==4.10.0.84

export MACA_PATH=/opt/maca
if [ ! -d ${HOME}/cu-bridge ]; then
            `${MACA_PATH}/tools/cu-bridge/tools/pre_make`
fi

export CUDA_PATH=${HOME}/cu-bridge/CUDA_DIR
export LD_LIBRARY_PATH=${CUDA_PATH}/lib64:${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:$LD_LIBRARY_PATH
export PADDLE_XCCL_BACKEND=metax_gpu
export FLAGS_weight_only_linear_arch=80
export FD_MOE_BACKEND=cutlass # 或 triton
export FD_METAX_KVCACHE_MEM=8
export ENABLE_V1_KVCACHE_SCHEDULER=1
export FD_ENC_DEC_BLOCK_NUM=2
export FD_SAMPLING_CLASS="rejection"
# export PADDLE_PDX_DISABLE_DEV_MODEL_WL=true

export MODEL_ROOT_PATH="/data/models/PaddlePaddle"
export INPUT_IMAGE_PATH="/data/material/paddleocr_vl_demo.png"
export OCR_OUTPUTS_PATH="${PWD}/paddleocr_outputs"
export SERVER_TIMEOUT_SEC=300

SERVER_LOG_FILE="${PWD}/paddle_ocr_server.log"
CLIENT_LOG_FILE="${PWD}/paddle_ocr_client.log"


# SERVER_CMD="paddleocr genai_server --model_name PaddleOCR-VL-1.5-0.9B --backend fastdeploy --port 8118 --model_dir ${MODEL_ROOT_PATH}/PaddleOCR-VL-1.5/"
SERVER_CMD="python  -m fastdeploy.entrypoints.openai.api_server \
                    --model ${MODEL_ROOT_PATH}/PaddleOCR-VL-1.5/ \
                    --max-model-len 16384 \
                    --max-num-batched-tokens 16384 \
                    --gpu-memory-utilization 0.7 \
                    --max-num-seqs 256 \
                    --graph-optimization-config '{\"use_cudagraph\": false}' \
                    --workers 4 \
                    --max-concurrency 8192 \
                    --port 8118 \
                    --metrics-port 8201 \
                    --engine-worker-queue-port 8202"

TRIGGER_KEY="Application startup complete"
CLIENT_CMD="python tests/metax_ci/test_paddle_ocr.py"


SERVER_PID=0
CLIENT_PID=0


cleanup() {
    echo -e "\n[INFO] Start terminating residual processes."
    if [ $SERVER_PID -ne 0 ] && kill -0 $SERVER_PID >/dev/null 2>&1; then
        kill -9 $SERVER_PID >/dev/null 2>&1 && echo  -e "[INFO] Successfully killed the server process (PID: $SERVER_PID)"
    else
        echo -e "[WARNING] Server process not running or exited (PID: $SERVER_PID)"
    fi

    if [ $CLIENT_PID -ne 0 ] && kill -0 $CLIENT_PID >/dev/null 2>&1; then
        kill -9 $CLIENT_PID >/dev/null 2>&1 && echo -e "[INFO] Successfully killed the client process (PID: $CLIENT_PID)"
    else
        echo -e "[WARNING] Client process not running or exited (PID: $CLIENT_PID)"
    fi
    echo -e "[INFO] Process termination completed."
}


trap cleanup SIGINT SIGTERM EXIT


echo "[INFO] Start the server service"
echo "[INFO] Server cmd - $SERVER_CMD"

eval $SERVER_CMD > "$SERVER_LOG_FILE" 2>&1 &
SERVER_PID=$!
if ! kill -0 $SERVER_PID >/dev/null 2>&1; then
    echo -e "[ERROR] Server command execution failed, process not started!"
    exit 1
fi

echo -e "[INFO] Waiting server service start completed ..."
while true; do
    POLL_COUNT=$((POLL_COUNT + 1))
    if [ $POLL_COUNT -ge $SERVER_TIMEOUT_SEC ]; then
        cat ${SERVER_LOG_FILE}
        cat log/workerlog.0
        echo "[TIMEOUT] Server process is about to terminate and exit the script!"
        exit 1
    fi

    if ! kill -0 $SERVER_PID >/dev/null 2>&1; then
        cat ${SERVER_LOG_FILE}
        cat log/workerlog.0
        echo "[ERROR] Server process(PID: $SERVER_PID) has exited abnormally and no keywords were detected!"
        exit 1
    fi

    if [ -s "$SERVER_LOG_FILE" ]; then
        if tail -n 30 "$SERVER_LOG_FILE" | grep -qF "$TRIGGER_KEY"; then
            break
        fi
    fi

    sleep 1
done

echo "[INFO] Start the Client service"
echo "[INFO] Client cmd - $CLIENT_CMD"
eval $CLIENT_CMD > "$CLIENT_LOG_FILE" 2>&1 &
CLIENT_PID=$!
wait $CLIENT_PID
CLIENT_EXIT_CODE=$?
if (( CLIENT_EXIT_CODE != 0 )); then
    echo -e "\n=========== PaddleOCR client error exit! - [ ${CLIENT_LOG_FILE} ] ==========="
    cat ${CLIENT_LOG_FILE}
else
    cat ${OCR_OUTPUTS_PATH}/paddleocr_vl_demo.md
fi
echo "[INFO] Client service running completed, exit code: $CLIENT_EXIT_CODE"

exit $CLIENT_EXIT_CODE
