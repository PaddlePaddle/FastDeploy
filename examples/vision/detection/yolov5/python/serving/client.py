import requests
import json
import cv2
import fastdeploy as fd
from fastdeploy.serving.utils import cv2_to_base64

if __name__ == '__main__':
    url = "http://127.0.0.1:8000/fd/yolov5s"
    headers = {"Content-Type": "application/json"}

    im = cv2.imread("000000014439.jpg")
    data = {"data": {"image": cv2_to_base64(im)}, "parameters": {}}

    resp = requests.post(url=url, headers=headers, data=json.dumps(data))
    if resp.status_code == 200:
        r_json = json.loads(resp.json()["result"])
        det_result = fd.vision.utils.json_to_detection(r_json)
        vis_im = fd.vision.vis_detection(im, det_result, score_threshold=0.5)
        cv2.imwrite("visualized_result.jpg", vis_im)
        print("Visualized result save in ./visualized_result.jpg")
    else:
        print("Error code:", resp.status_code)
        print(resp.text)
