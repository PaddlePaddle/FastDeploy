// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package main

// #cgo CFLAGS:  -I./fastdeploy_capi
// #cgo LDFLAGS: -L./fastdeploy-linux-x64-0.0.0/lib -lfastdeploy
// #include <fastdeploy_capi/vision.h>
// #include <stdio.h>
// #include <stdbool.h>
// #include <stdlib.h>
import "C"
import (
	"flag"
	"fmt"
	"unsafe"
)

func FDBooleanToGo(b C.FD_C_Bool) bool {
	var cFalse C.FD_C_Bool
	if b != cFalse {
		return true
	}
	return false
}

func CpuInfer(modelFile *C.char, imageFile *C.char) {

	var option *C.FD_C_RuntimeOptionWrapper = C.FD_C_CreateRuntimeOptionWrapper()
	C.FD_C_RuntimeOptionWrapperUseCpu(option)

	var model *C.FD_C_YOLOv5Wrapper = C.FD_C_CreateYOLOv5Wrapper(
		modelFile, C.CString(""), option, C.FD_C_ModelFormat_ONNX)

	if !FDBooleanToGo(C.FD_C_YOLOv5WrapperInitialized(model)) {
		fmt.Printf("Failed to initialize.\n")
		C.FD_C_DestroyRuntimeOptionWrapper(option)
		C.FD_C_DestroyYOLOv5Wrapper(model)
		return
	}

	var image C.FD_C_Mat = C.FD_C_Imread(imageFile)

	var result *C.FD_C_DetectionResult = C.FD_C_CreateDetectionResult()

	if !FDBooleanToGo(C.FD_C_YOLOv5WrapperPredict(model, image, result)) {
		fmt.Printf("Failed to predict.\n")
		C.FD_C_DestroyRuntimeOptionWrapper(option)
		C.FD_C_DestroyYOLOv5Wrapper(model)
		C.FD_C_DestroyMat(image)
		C.free(unsafe.Pointer(result))
		return
	}

	var visImage C.FD_C_Mat = C.FD_C_VisDetection(image, result, 0.5, 1, 0.5)

	C.FD_C_Imwrite(C.CString("vis_result.jpg"), visImage)
	fmt.Printf("Visualized result saved in ./vis_result.jpg\n")

	C.FD_C_DestroyRuntimeOptionWrapper(option)
	C.FD_C_DestroyYOLOv5Wrapper(model)
	C.FD_C_DestroyDetectionResult(result)
	C.FD_C_DestroyMat(image)
	C.FD_C_DestroyMat(visImage)
}

func GpuInfer(modelFile *C.char, imageFile *C.char) {

	var option *C.FD_C_RuntimeOptionWrapper = C.FD_C_CreateRuntimeOptionWrapper()
	C.FD_C_RuntimeOptionWrapperUseGpu(option, 0)

	var model *C.FD_C_YOLOv5Wrapper = C.FD_C_CreateYOLOv5Wrapper(
		modelFile, C.CString(""), option, C.FD_C_ModelFormat_ONNX)

	if !FDBooleanToGo(C.FD_C_YOLOv5WrapperInitialized(model)) {
		fmt.Printf("Failed to initialize.\n")
		C.FD_C_DestroyRuntimeOptionWrapper(option)
		C.FD_C_DestroyYOLOv5Wrapper(model)
		return
	}

	var image C.FD_C_Mat = C.FD_C_Imread(imageFile)

	var result *C.FD_C_DetectionResult = C.FD_C_CreateDetectionResult()

	if !FDBooleanToGo(C.FD_C_YOLOv5WrapperPredict(model, image, result)) {
		fmt.Printf("Failed to predict.\n")
		C.FD_C_DestroyRuntimeOptionWrapper(option)
		C.FD_C_DestroyYOLOv5Wrapper(model)
		C.FD_C_DestroyMat(image)
		C.free(unsafe.Pointer(result))
		return
	}

	var visImage C.FD_C_Mat = C.FD_C_VisDetection(image, result, 0.5, 1, 0.5)

	C.FD_C_Imwrite(C.CString("vis_result.jpg"), visImage)
	fmt.Printf("Visualized result saved in ./vis_result.jpg\n")

	C.FD_C_DestroyRuntimeOptionWrapper(option)
	C.FD_C_DestroyYOLOv5Wrapper(model)
	C.FD_C_DestroyDetectionResult(result)
	C.FD_C_DestroyMat(image)
	C.FD_C_DestroyMat(visImage)
}

var (
	modelFile  string
	imageFile  string
	deviceType int
)

func init() {
	flag.StringVar(&modelFile, "model", "", "paddle detection model to use")
	flag.StringVar(&imageFile, "image", "", "image to predict")
	flag.IntVar(&deviceType, "device", 0, "The data type of run_option is int, 0: run with cpu; 1: run with gpu")
}

func main() {
	flag.Parse()

	if modelFile != "" && imageFile != "" {
		if deviceType == 0 {
			CpuInfer(C.CString(modelFile), C.CString(imageFile))
		} else if deviceType == 1 {
			GpuInfer(C.CString(modelFile), C.CString(imageFile))
		}
	} else {
		fmt.Printf("Usage: ./infer -model path/to/model_dir -image path/to/image -device run_option \n")
		fmt.Printf("e.g ./infer -model yolov5s.onnx -image 000000014439.jpg -device 0 \n")
	}

}
