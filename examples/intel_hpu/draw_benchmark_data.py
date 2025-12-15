import csv
import os
import re
import sys
from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

log_patterns = [
    re.compile(
        r"benchmarkdata_(.+?)_inputlength_(\d+)_outputlength_(\d+)_batchsize_(\d+)_numprompts_(\d+)_.*_profile\.log$"
    ),
]


def draw_time_graph(log_dir, log_filename, max_num_seqs, mode):
    # Store extracted time and BT values
    timestamps_model = []
    times_model = []
    bt_values_model = []
    block_list_shapes_model = []
    block_indices_shapes_model = []
    timestamps_pp = []
    times_pp = []
    bt_values_pp = []

    # Use regex to extract Model execution time and BT information
    pattern_model = re.compile(
        r"(\d+-\d+-\d+ \d+:\d+:\d+,\d+) .* Model execution time\(ms\): ([\d\.]+), BT=(\d+), block_list_shape=\[(\d+)\], block_indices_shape=\[(\d+)\]"
    )
    pattern_pp = re.compile(
        r"(\d+-\d+-\d+ \d+:\d+:\d+,\d+) .* PostProcessing execution time\(ms\): ([\d\.]+), BT=(\d+)"
    )
    # Read log file
    with open(os.path.join(log_dir, log_filename), "r") as file:
        for line in file:
            match_model = pattern_model.search(line)
            if match_model:
                bt_value = int(match_model.group(3))
                timestamps_model.append(datetime.strptime(match_model.group(1), "%Y-%m-%d %H:%M:%S,%f"))
                if mode == "prefill" and bt_value <= max_num_seqs:
                    times_model.append(None)
                    bt_values_model.append(None)
                    continue
                if mode == "decode" and bt_value > max_num_seqs:
                    times_model.append(None)
                    bt_values_model.append(None)
                    continue
                times_model.append(float(match_model.group(2)))
                bt_values_model.append(bt_value)
                block_list_shapes_model.append(int(match_model.group(4)))
                block_indices_shapes_model.append(int(match_model.group(5)))
            else:
                match_pp = pattern_pp.search(line)
                if match_pp:
                    bt_value = int(match_pp.group(3))
                    timestamps_pp.append(datetime.strptime(match_pp.group(1), "%Y-%m-%d %H:%M:%S,%f"))
                    if mode == "prefill" and bt_value <= max_num_seqs:
                        times_pp.append(None)
                        bt_values_pp.append(None)
                        continue
                    if mode == "decode" and bt_value > max_num_seqs:
                        times_pp.append(None)
                        bt_values_pp.append(None)
                        continue
                    times_pp.append(float(match_pp.group(2)))
                    bt_values_pp.append(bt_value)

    # Plot graphs
    plt.figure(figsize=(15, 7))

    date_format = mdates.DateFormatter("%m-%d %H:%M:%S")
    # Plot time graph
    plt.subplot(2, 1, 1)
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax1.plot(timestamps_model, times_model, label="Model Execution Time (ms)", color="blue")
    ax2.plot(timestamps_pp, times_pp, label="PostProcessing Time (ms)", color="red")
    ax1.set_ylabel("Model Execution Time (ms)")
    ax2.set_ylabel("PostProcessing Time (ms)")
    ax1.xaxis.set_major_formatter(date_format)
    # Merge legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2)

    # Plot BT value graph
    plt.subplot(2, 1, 2)
    plt.plot(timestamps_model, bt_values_model, label="BT  [" + mode + "]", color="orange")
    plt.ylabel("BT Value")
    plt.xlabel(log_filename, fontsize=8)

    plt.gca().xaxis.set_major_formatter(date_format)
    plt.legend()

    plt.tight_layout()
    output_filename = log_filename[:-4] + "_analysis_" + mode + ".png"
    plt.savefig(os.path.join(log_dir, output_filename), dpi=300)
    plt.close()

    # Write to CSV file
    if mode == "all":
        csv_filename = log_filename[:-4] + "_analysis.csv"
        with open(os.path.join(log_dir, csv_filename), "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "Timestamp",
                    "ModelTime(ms)",
                    "BT",
                    "block_list_shape",
                    "block_indices_shape",
                    "Timestamp",
                    "PostProcessing(ms)",
                    "BT",
                ]
            )
            for i in range(len(times_model)):
                writer.writerow(
                    [
                        timestamps_model[i],
                        times_model[i],
                        bt_values_model[i],
                        block_list_shapes_model[i],
                        block_indices_shapes_model[i],
                        timestamps_pp[i],
                        times_pp[i],
                        bt_values_pp[i],
                    ]
                )


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

    for file in files:
        for idx, pat in enumerate(log_patterns):
            m = pat.match(file)
            if m:
                draw_time_graph(log_dir, file, 128, "prefill")
                draw_time_graph(log_dir, file, 128, "decode")
                draw_time_graph(log_dir, file, 128, "all")


if __name__ == "__main__":
    print("Starting to draw logs...")
    main()
