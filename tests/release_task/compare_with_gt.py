import numpy as np
import re


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
    args = parser.parse_args()
    return args


def convert2numpy(result_file):
    result = []
    with open(result_file, "r+") as f:
        for line in f.readlines():
            data = re.findall(r"\d+\.?\d*", line)
            if len(data) == 6:
                result.append([float(num) for num in data])
    return np.array(result)


def write2file(error_file):
    with open(error_file, "w+") as f:
        from platform import python_version
        py_version = python_version()
        f.write(args.platform + " " + py_version + " " +
                args.result_path.split(".")[0] + "\n")


def check_result(gt_result, infer_result, args):
    if len(gt_result) != len(infer_result):
        infer_result = infer_result[-len(gt_result):]
    diff = np.abs(gt_result - infer_result)
    if (diff > 1e-5).all():
        print(args.platform, args.device, "diff ", diff)
        write2file("result.txt")
    else:
        print(args.platform, args.device, "No diff")


if __name__ == '__main__':
    args = parse_arguments()

    gt_numpy = convert2numpy(args.gt_path)
    infer_numpy = convert2numpy(args.result_path)
    check_result(gt_numpy, infer_numpy, args)
