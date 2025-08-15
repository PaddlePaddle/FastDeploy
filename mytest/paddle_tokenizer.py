from fastdeploy.engine.request import Request
from fastdeploy.input.preprocess import InputPreprocessor
import json


def main():
    model_name = "./data/models/paddle/Qwen2.5-VL-3B-Instruct"
    # model_name = "./data/models/paddle/ERNIE-4.5-0.3B-Paddle"
    input_processor = InputPreprocessor(
        model_name_or_path=model_name,
        reasoning_parser=None,
        limit_mm_per_prompt=None,
        mm_processor_kwargs=None,
        enable_mm=True
    )
    data_processor = input_processor.create_processor()

    prompt = {
        "request_id": "123",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "file:///home/liudongdong/github/FastDeploy/data/images/demo.jpeg"}
                    },
                    {
                        "type": "video_url",
                        "video_url": {"url": "file:///home/liudongdong/github/FastDeploy/data/images/3_frame_video.mp4"}
                    },
                    {
                        "type": "text",
                        "text": "Describe image and video."
                    },
                ]
            }
        ]
    }

    request = Request.from_dict(prompt)
    result = data_processor.process_request(request, 1024*100)

    print(result)

    with open("out/token_ids.json", "w") as f:
        f.write(json.dumps(result.prompt_token_ids))

    # with open("out/pixel.json", "w") as f:
    #     f.write(json.dumps(result.multimodal_inputs["images"].tolist()))

    # with open("out/grid_thw.json", "w") as f:
    #     f.write(json.dumps(result.multimodal_inputs["grid_thw"].tolist()))

    # with open("out/pixel_values_videos.json", "w") as f:
    #     f.write(json.dumps(result.multimodal_inputs["images"].tolist()))

    # with open("out/video_grid_thw.json", "w") as f:
    #     f.write(json.dumps(result.multimodal_inputs["grid_thw"].tolist()))

    with open("out/position_ids.json", "w") as f:
        f.write(json.dumps(result.multimodal_inputs["position_ids"].tolist()))


if __name__ == "__main__":
    main()

