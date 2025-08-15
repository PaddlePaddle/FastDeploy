import json
import numpy as np


"""
dd if=out/pixel_values_videos.json bs=1 skip=2000 count=100


dd if=文件名 bs=1 skip=起始位置 count=10 2>/dev/null
    - bs=1: 每次读取 1 字节（确保精确跳转）。
    - skip=N: 跳过前 N 个字节（从 N+1 字节开始读取）。
    - count=10: 读取 10 个字节。
    - 2>/dev/null: 屏蔽 dd 的警告信息。
"""


def load_numpy(filename):
    with open(filename, "r") as f:
        return np.array(json.loads(f.read()))


def main():
    token_ids_1 = load_numpy("out/token_ids.json")
    token_ids_2 = load_numpy("../llm/out/token_ids.json")[0]

    diff_indices = np.where(token_ids_1 != token_ids_2)
    print(diff_indices)


    # pixel_1 = load_numpy("out/pixel.json")
    # pixel_2 = load_numpy("../llm/out/pixel.json")

    # diff_indices = np.where(pixel_1 != pixel_2)
    # print(diff_indices)

    # video_pixel_1 = load_numpy("out/pixel_values_videos.json")
    # video_pixel_2 = load_numpy("../llm/out/pixel_values_videos.json")

    # diff_indices = np.where(video_pixel_1 != video_pixel_2)
    # print(diff_indices)

    position_ids_1 = load_numpy("out/position_ids.json")
    position_ids_2 = load_numpy("../llm/out/position_ids.json")

    diff_indices = np.where(position_ids_1 != position_ids_2)
    print(diff_indices)


if __name__=="__main__":
    main()