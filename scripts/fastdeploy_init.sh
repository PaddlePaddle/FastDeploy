
CURRENT_EXE_DIR=$(pwd)

cd $(dirname $BASH_SOURCE)
INSTALLED_PREBUILT_FASTDEPLOY_DIR=$(pwd)


echo "Import environment variable from FastDeploy..."

export LD_LIBRARY_PATH=${INSTALLED_PREBUILT_FASTDEPLOY_DIR}/lib:${LD_LIBRARY_PATH}
echo "FastDeploy Lib: ${INSTALLED_PREBUILT_FASTDEPLOY_DIR}/lib"

if [ -d ${INSTALLED_PREBUILT_FASTDEPLOY_DIR}/third_libs/install/fast_tokenizer ]; then
	export LD_LIBRARY_PATH=${INSTALLED_PREBUILT_FASTDEPLOY_DIR}/third_libs/install/fast_tokenizer/lib:${LD_LIBRARY_PATH}
	echo "FastTokenizer Lib: ${INSTALLED_PREBUILT_FASTDEPLOY_DIR}/third_libs/install/fast_tokenizer/lib"
fi

if [ -d ${INSTALLED_PREBUILT_FASTDEPLOY_DIR}/third_libs/install/paddle2onnx ]; then
	export LD_LIBRARY_PATH=${INSTALLED_PREBUILT_FASTDEPLOY_DIR}/third_libs/install/paddle2onnx/lib:${LD_LIBRARY_PATH}
	echo "Paddle2ONNX Lib: ${INSTALLED_PREBUILT_FASTDEPLOY_DIR}/third_libs/install/paddle2onnx/lib"
fi

if [ -d ${INSTALLED_PREBUILT_FASTDEPLOY_DIR}/third_libs/install/onnxruntime ]; then
	export LD_LIBRARY_PATH=${INSTALLED_PREBUILT_FASTDEPLOY_DIR}/third_libs/install/onnxruntime/lib:${LD_LIBRARY_PATH}
	echo "ONNX Runtime Lib: ${INSTALLED_PREBUILT_FASTDEPLOY_DIR}/third_libs/install/onnxruntime/lib"
fi

if [ -d ${INSTALLED_PREBUILT_FASTDEPLOY_DIR}/third_libs/install/opencv ]; then
	export LD_LIBRARY_PATH=${INSTALLED_PREBUILT_FASTDEPLOY_DIR}/third_libs/install/opencv/lib:${LD_LIBRARY_PATH}
	echo "OpenCV Lib: ${INSTALLED_PREBUILT_FASTDEPLOY_DIR}/third_libs/install/opencv/lib"
fi

if [ -d ${INSTALLED_PREBUILT_FASTDEPLOY_DIR}/third_libs/install/openvino ]; then
	export LD_LIBRARY_PATH=${INSTALLED_PREBUILT_FASTDEPLOY_DIR}/third_libs/install/openvino/lib:${LD_LIBRARY_PATH}
	echo "OpenVINO Lib: ${INSTALLED_PREBUILT_FASTDEPLOY_DIR}/third_libs/install/openvino/lib"
fi

if [ -d ${INSTALLED_PREBUILT_FASTDEPLOY_DIR}/third_libs/install/tensorrt ]; then
	export LD_LIBRARY_PATH=${INSTALLED_PREBUILT_FASTDEPLOY_DIR}/third_libs/install/tensorrt/lib:${LD_LIBRARY_PATH}
	echo "TensorRT Lib: ${INSTALLED_PREBUILT_FASTDEPLOY_DIR}/third_libs/install/tensorrt/lib"
fi

if [ -d ${INSTALLED_PREBUILT_FASTDEPLOY_DIR}/third_libs/install/paddle_inference ]; then
	export LD_LIBRARY_PATH=${INSTALLED_PREBUILT_FASTDEPLOY_DIR}/third_libs/install/paddle_inference/paddle/lib:${LD_LIBRARY_PATH}
	echo "Paddle Inference Lib: ${INSTALLED_PREBUILT_FASTDEPLOY_DIR}/third_libs/install/paddle_inference/paddle/lib"
	export LD_LIBRARY_PATH=${INSTALLED_PREBUILT_FASTDEPLOY_DIR}/third_libs/install/paddle_inference/third_party/install/mkldnn/lib:${LD_LIBRARY_PATH}
	echo "MKLDNN Lib: ${INSTALLED_PREBUILT_FASTDEPLOY_DIR}/third_libs/install/paddle_inference/third_party/install/mkldnn/lib"
	export LD_LIBRARY_PATH=${INSTALLED_PREBUILT_FASTDEPLOY_DIR}/third_libs/install/paddle_inference/third_party/install/mklml/lib:${LD_LIBRARY_PATH}
	echo "MKLML Lib: ${INSTALLED_PREBUILT_FASTDEPLOY_DIR}/third_libs/install/paddle_inference/third_party/install/mklml/lib"
fi

cd ${CURRENT_EXE_DIR}
