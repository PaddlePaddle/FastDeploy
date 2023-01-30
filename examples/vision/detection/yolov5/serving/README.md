English | [简体中文](README_CN.md)
# YOLOv5 Serving Deployment Demo

## Launch Serving

```bash
#Download yolov5 model file
wget https://bj.bcebos.com/paddlehub/fastdeploy/yolov5s.onnx

# Save the model under models/infer/1 and rename it as model.onnx
mv yolov5s.onnx models/infer/1/

# Pull fastdeploy image, x.y.z is FastDeploy version, example 1.0.0.
docker pull paddlepaddle/fastdeploy:x.y.z-gpu-cuda11.4-trt8.4-21.10

# Run the docker. The docker name is fd_serving, and the current directory is mounted as the docker's /yolov5_serving directory
nvidia-docker run -it --net=host --name fd_serving -v `pwd`/:/yolov5_serving paddlepaddle/fastdeploy:x.y.z-gpu-cuda11.4-trt8.4-21.10  bash

# Start the service (Without setting the CUDA_VISIBLE_DEVICES environment variable, it will have scheduling privileges for all GPU cards)
CUDA_VISIBLE_DEVICES=0 fastdeployserver --model-repository=models --backend-config=python,shm-default-byte-size=10485760
```

Output the following contents if serving is launched

```
......
I0928 04:51:15.784517 206 grpc_server.cc:4117] Started GRPCInferenceService at 0.0.0.0:8001
I0928 04:51:15.785177 206 http_server.cc:2815] Started HTTPService at 0.0.0.0:8000
I0928 04:51:15.826578 206 http_server.cc:167] Started Metrics Service at 0.0.0.0:8002
```

## Client Requests

Execute the following command in the physical machine to send a grpc request and output the result

```
#Download test images
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg

#Installing client-side dependencies
python3 -m pip install tritonclient\[all\]

# Send requests
python3 yolov5_grpc_client.py
```

When the request is sent successfully, the results are returned in json format and printed out:

```
output_name: detction_result
{'boxes': [[268.48028564453125, 81.05305480957031, 298.69476318359375, 169.43902587890625], [104.73116302490234, 45.66197204589844, 127.58382415771484, 93.44938659667969], [378.9093933105469, 39.75013732910156, 395.6086120605469, 84.24342346191406], [158.552978515625, 80.36149597167969, 199.18576049804688, 168.18191528320312], [414.37530517578125, 90.94805908203125, 506.3218994140625, 280.40521240234375], [364.00341796875, 56.608917236328125, 381.97857666015625, 115.96823120117188], [351.7251281738281, 42.635345458984375, 366.9103088378906, 98.04837036132812], [505.8882751464844, 114.36674499511719, 593.1248779296875, 275.99530029296875], [327.7086181640625, 38.36369323730469, 346.84991455078125, 80.89302062988281], [583.493408203125, 114.53289794921875, 612.3546142578125, 175.87353515625], [186.4706573486328, 44.941375732421875, 199.6645050048828, 61.037628173828125], [169.6158905029297, 48.01460266113281, 178.1415557861328, 60.88859558105469], [25.81019401550293, 117.19969177246094, 59.88878631591797, 152.85012817382812], [352.1452941894531, 46.71272277832031, 381.9460754394531, 106.75212097167969], [1.875, 150.734375, 37.96875, 173.78125], [464.65728759765625, 15.901412963867188, 472.512939453125, 34.11640930175781], [64.625, 135.171875, 84.5, 154.40625], [57.8125, 151.234375, 103.0, 174.15625], [165.890625, 88.609375, 527.90625, 339.953125], [101.40625, 152.5625, 118.890625, 169.140625]], 'scores': [0.8965693116188049, 0.8695310950279236, 0.8684297800064087, 0.8429877758026123, 0.8358422517776489, 0.8151364326477051, 0.8089362382888794, 0.801361083984375, 0.7947245836257935, 0.7606497406959534, 0.6325908303260803, 0.6139386892318726, 0.5906146764755249, 0.505328893661499, 0.40457233786582947, 0.3460320234298706, 0.33283042907714844, 0.3325657248497009, 0.2594234347343445, 0.25389009714126587], 'label_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 0, 24, 24, 33, 24], 'masks': [], 'contain_masks': False}
```

## Modify Configs



The default is to run ONNXRuntime on CPU. If developers need to run it on GPU or other inference engines, please see the  [Configs File](../../../../../serving/docs/EN/model_configuration-en.md) to modify the configs in `models/runtime/config.pbtxt`.

## Use VisualDL for serving deployment visualization

You can use VisualDL for [serving deployment visualization](../../../../../serving/docs/EN/vdl_management-en.md) , the above model preparation, deployment, configuration modification and client request operations can all be performed based on VisualDL.

The serving deployment of yolov5 by VisualDL only needs the following three steps:
```text
1. Load the model repository: ./vision/detection/yolov5/serving/models
2. Download the model resource file: click the runtime model, click the version number 1 to add the pre-training model, and select the detection model yolov5s to download.
3. Start the service: Click the "launch server" button and input the launch parameters.
```
 <p align="center">
  <img src="https://user-images.githubusercontent.com/22424850/211709339-023fef22-3ffc-4b3d-bce5-ea4202bb9c61.gif" width="100%"/>
</p>
