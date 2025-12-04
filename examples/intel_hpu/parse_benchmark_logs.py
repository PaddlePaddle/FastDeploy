import csv
import os
import re
import sys

log_patterns = [
    re.compile(
        r"benchmarkdata_(.+?)_inputlength_(\d+)_outputlength_(\d+)_batchsize_(\d+)_numprompts_(\d+)_.*(?<!_profile)\.log$"
    ),
    re.compile(r"benchmarkdata_(.+?)_sharegpt_prompts_(\d+)_concurrency_(\d+)_.*(?<!_profile)\.log$"),
]

metrics = [
    ("Mean Decode", r"Mean Decode:\s+([\d\.]+)"),
    ("Mean TTFT (ms)", r"Mean TTFT \(ms\):\s+([\d\.]+)"),
    ("Mean S_TTFT (ms)", r"Mean S_TTFT \(ms\):\s+([\d\.]+)"),
    ("Mean TPOT (ms)", r"Mean TPOT \(ms\):\s+([\d\.]+)"),
    ("Mean ITL (ms)", r"Mean ITL \(ms\):\s+([\d\.]+)"),
    ("Mean S_ITL (ms)", r"Mean S_ITL \(ms\):\s+([\d\.]+)"),
    ("Mean E2EL (ms)", r"Mean E2EL \(ms\):\s+([\d\.]+)"),
    ("Mean S_E2EL (ms)", r"Mean S_E2EL \(ms\):\s+([\d\.]+)"),
    ("Mean Input Length", r"Mean Input Length:\s+([\d\.]+)"),
    ("Mean Output Length", r"Mean Output Length:\s+([\d\.]+)"),
    ("Request throughput (req/s)", r"Request throughput \(req/s\):\s+([\d\.]+)"),
    ("Output token throughput (tok/s)", r"Output token throughput \(tok/s\):\s+([\d\.]+)"),
    ("Total Token throughput (tok/s)", r"Total Token throughput \(tok/s\):\s+([\d\.]+)"),
]


def parse_benchmark_log_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    result = {}
    for name, pattern in metrics:
        match = re.search(pattern, content)
        result[name] = match.group(1) if match else ""
    return result


def parse_profile_log_file(file_path):
    prepare_input_times = []
    model_times = []
    postprocessing_times = []
    steppaddle_times = []

    with open(file_path, "r") as file:
        for line in file:
            prepare_input_match = re.search(r"_prepare_inputs time\(ms\): (\d+\.\d+)", line)
            model_match = re.search(r"Model execution time\(ms\): (\d+\.\d+)", line)
            postprocessing_match = re.search(r"PostProcessing execution time\(ms\): (\d+\.\d+)", line)
            steppaddle_match = re.search(r"StepPaddle execution time\(ms\): (\d+\.\d+)", line)

            if prepare_input_match:
                prepare_input_times.append(float(prepare_input_match.group(1)))
            if model_match:
                model_times.append(float(model_match.group(1)))
            if postprocessing_match:
                postprocessing_times.append(float(postprocessing_match.group(1)))
            if steppaddle_match:
                steppaddle_times.append(float(steppaddle_match.group(1)))

    return prepare_input_times, model_times, postprocessing_times, steppaddle_times


def calculate_times(times, separate_first):
    if len(times) < 2:
        return times[0], None
    if separate_first:
        first_time = times[0]
        average_time = sum(times[1:]) / len(times[1:])
        return first_time, average_time
    else:
        return None, sum(times) / len(times)


def main():
    if len(sys.argv) > 1:
        log_dir = sys.argv[1]
    else:
        log_dir = "."
    try:
        from natsort import natsorted

        natsort_available = True
    except ImportError:
        natsort_available = False
    all_files = set(os.listdir(log_dir))
    files = []
    for f in os.listdir(log_dir):
        for pat in log_patterns:
            if pat.match(f):
                files.append(f)
                break
    if natsort_available:
        files = natsorted(files)
    else:
        import re as _re

        def natural_key(s):
            return [int(text) if text.isdigit() else text.lower() for text in _re.split("([0-9]+)", s)]

        files.sort(key=natural_key)
    rows = []

    for file in files:
        m = None
        matched_idx = -1
        for idx, pat in enumerate(log_patterns):
            m = pat.match(file)
            if m:
                matched_idx = idx
                break
        if not m:
            continue
        # model_name, input_len, output_len, batch_size, num_prompts
        # model_name, num_prompts, max_concurrency
        if matched_idx == 0:
            model_name, input_len, output_len, batch_size, num_prompts = m.groups()
        elif matched_idx == 1:
            model_name, num_prompts, max_concurrency = m.groups()
            input_len = "-"
            output_len = "-"
        if file.endswith(".log"):
            profile_file = file[:-4] + "_profile.log"
        else:
            profile_file = ""
        model_first = model_average = postprocessing_average = steppaddle_average = ""
        if profile_file in all_files:
            prepare_input_times, model_times, postprocessing_times, steppaddle_times = parse_profile_log_file(
                os.path.join(log_dir, profile_file)
            )
            _, pia = calculate_times(prepare_input_times, False)
            mf, ma = calculate_times(model_times, True)
            _, pa = calculate_times(postprocessing_times, False)
            _, sa = calculate_times(steppaddle_times, False)
            prepare_input_average = pia if pia is not None else ""
            model_first = mf if mf is not None else ""
            model_average = ma if ma is not None else ""
            postprocessing_average = pa if pa is not None else ""
            steppaddle_average = sa if sa is not None else ""
        data = parse_benchmark_log_file(os.path.join(log_dir, file))
        data["dataset"] = "Fixed-Length" if matched_idx == 0 else "ShareGPT"
        data["model_name"] = model_name
        data["input_length"] = input_len
        data["output_length"] = output_len
        data["batch_size"] = batch_size if matched_idx == 0 else max_concurrency
        data["num_prompts"] = num_prompts
        data["prepare_input_average"] = prepare_input_average
        data["model_execute_first"] = model_first
        data["model_execute_average"] = model_average
        data["postprocessing_execute_average"] = postprocessing_average
        data["steppaddle_execute_average"] = steppaddle_average
        rows.append(data)

    import datetime

    import pytz

    shanghai_tz = pytz.timezone("Asia/Shanghai")
    now = datetime.datetime.now(shanghai_tz)
    ts = now.strftime("%Y%m%d_%H%M%S")
    log_dir_name = os.path.basename(os.path.abspath(log_dir))
    if log_dir_name == "" or log_dir == "." or log_dir == "/":
        csv_filename = f"benchmark_summary_{ts}.csv"
    else:
        csv_filename = f"benchmark_summary_{log_dir_name}_{ts}.csv"
    fieldnames = (
        [
            "model_name",
            "dataset",
            "input_length",
            "output_length",
            "batch_size",
            "num_prompts",
        ]
        + [name for name, _ in metrics]
        + [
            "prepare_input_average",
            "model_execute_first",
            "model_execute_average",
            "postprocessing_execute_average",
            "steppaddle_execute_average",
        ]
    )
    with open(csv_filename, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"CSV saved as: {csv_filename}")


if __name__ == "__main__":
    print("Starting to parse logs...")
    main()
