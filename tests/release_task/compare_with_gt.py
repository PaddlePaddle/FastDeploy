import numpy as np
import re

diff_score_threshold = {
    "linux-x64": {
        "label_diff": 0,
        "score_diff": 1e-4,
        "boxes_diff_ratio": 1e-4,
        "boxes_diff": 1e-3
    },
    "linux-aarch64": {
        "label_diff": 0,
        "score_diff": 1e-4,
        "boxes_diff_ratio": 1e-4,
        "boxes_diff": 1e-3
    },
    "osx-x86_64": {
        "label_diff": 0,
        "score_diff": 1e-4,
        "boxes_diff_ratio": 2e-4,
        "boxes_diff": 1e-3
    },
    "osx-arm64": {
        "label_diff": 0,
        "score_diff": 1e-4,
        "boxes_diff_ratio": 2e-4,
        "boxes_diff": 1e-3
    },
    "win-x64": {
        "label_diff": 0,
        "score_diff": 5e-4,
        "boxes_diff_ratio": 1e-3,
        "boxes_diff": 1e-3
    }
}


def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt_path",
        type=str,
        required=True,
        help="Path of ground truth result path.")
    parser.add_argument(
        "--result_path",
        type=str,
        required=True,
        help="Path of inference result path.")
    parser.add_argument(
        "--platform", type=str, required=True, help="Testcase platform.")
    parser.add_argument(
        "--device", type=str, required=True, help="Testcase device.")
    parser.add_argument(
        "--conf_threshold",
        type=float,
        required=False,
        default=0,
        help="The threshold to filter inference result.")
    args = parser.parse_args()
    return args


def convert2numpy(result_file, conf_threshold):
    result = []
    with open(result_file, "r+") as f:
        for line in f.readlines():
            data = re.findall(r"\d+\.?\d*", line)
            if len(data) == 6:
                if float(data[-2]) < conf_threshold:
                    continue
                else:
                    result.append([float(num) for num in data])
    return np.array(result)


def write2file(error_file):
    import os
    if not os.path.exists(error_file):
        with open(error_file, "w+") as f:
            f.write("Failed Cases:\n")
    with open(error_file, "a+") as f:
        from platform import python_version
        py_version = python_version()
        f.write(args.platform + " " + py_version + " " +
                args.result_path.split(".")[0] + "\n")


def save_numpy_result(file_path, error_msg):
    np.savetxt(file_path, error_msg, fmt='%f', delimiter=',')


def check_result(gt_result, infer_result, args):
    platform = args.platform
    if len(gt_result) != len(infer_result):
        infer_result = infer_result[-len(gt_result):]
    diff = np.abs(gt_result - infer_result)
    label_diff = diff[:, -1]
    score_diff = diff[:, -2]
    boxes_diff = diff[:, :-2]
    boxes_diff_ratio = boxes_diff / (infer_result[:, :-2] + 1e-6)
    is_diff = False
    backend = args.result_path.split(".")[0]
    if (label_diff > diff_score_threshold[platform]["label_diff"]).any():
        print(args.platform, args.device, "label diff ", label_diff)
        is_diff = True
        label_diff_bool_file = args.platform + "_" + backend + "_" + "label_diff_bool.txt"
        save_numpy_result(label_diff_bool_file, label_diff > 0)
    if (score_diff > diff_score_threshold[platform]["score_diff"]).any():
        print(args.platform, args.device, "score diff ", score_diff)
        is_diff = True
        score_diff_bool_file = args.platform + "_" + backend + "_" + "score_diff_bool.txt"
        save_numpy_result(score_diff_bool_file, score_diff > 1e-4)
    if (boxes_diff_ratio > diff_score_threshold[platform]["boxes_diff_ratio"]
        ).any() and (
            boxes_diff > diff_score_threshold[platform]["boxes_diff"]).any():
        print(args.platform, args.device, "boxes diff ", boxes_diff_ratio)
        is_diff = True
        boxes_diff_bool_file = args.platform + "_" + backend + "_" + "boxes_diff_bool.txt"
        boxes_diff_ratio_file = args.platform + "_" + backend + "_" + "boxes_diff_ratio.txt"
        boxes_diff_ratio_bool_file = args.platform + "_" + backend + "_" + "boxes_diff_ratio_bool"
        save_numpy_result(boxes_diff_bool_file, boxes_diff > 1e-3)
        save_numpy_result(boxes_diff_ratio_file, boxes_diff_ratio)
        save_numpy_result(boxes_diff_ratio_bool_file, boxes_diff_ratio > 1e-4)
    if is_diff:
        write2file("result.txt")
    else:
        print(args.platform, args.device, "No diff")


if __name__ == '__main__':
    args = parse_arguments()

    gt_numpy = convert2numpy(args.gt_path, args.conf_threshold)
    infer_numpy = convert2numpy(args.result_path, args.conf_threshold)
    check_result(gt_numpy, infer_numpy, args)
