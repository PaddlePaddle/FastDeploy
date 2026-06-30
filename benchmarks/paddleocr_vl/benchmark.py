#!/usr/bin/env python

import argparse
import glob
import json
import os
import sys
import time
import uuid
from operator import itemgetter
from threading import Thread

import pynvml
import tiktoken
from tqdm import tqdm

shutdown = False

encoding = tiktoken.get_encoding("cl100k_base")


class Predictor(object):
    def predict(self, task_info, batch_data):
        task_info["start_time"] = get_curr_time()
        try:
            markdown, num_pages = self._predict(batch_data)
        except Exception as e:
            task_info["successful"] = False
            print(e)
            raise
        finally:
            task_info["end_time"] = get_curr_time()
        task_info["successful"] = True
        task_info["processed_pages"] = num_pages
        task_info["generated_tokens"] = len(encoding.encode(markdown))
        return markdown

    def _predict(self, batch_data):
        raise NotImplementedError

    def close(self):
        pass


class PaddleXPredictor(Predictor):
    def __init__(self, config_path, predict_kwargs=None, log_vlm_inputs=False, log_vlm_inputs_limit=20):
        from paddlex import create_pipeline

        super().__init__()
        self.pipeline = create_pipeline(config_path)
        self.predict_kwargs = predict_kwargs or {}
        if log_vlm_inputs:
            self._install_vlm_input_logger(log_vlm_inputs_limit)

    def _install_vlm_input_logger(self, limit):
        if not hasattr(self.pipeline, "_paddleocr_vl_run_vl_recognition_batches"):
            print("[benchmark] VLM input logging is unavailable for this PaddleX pipeline.")
            return

        original_run_vlm_batches = self.pipeline._paddleocr_vl_run_vl_recognition_batches

        def image_shape(image):
            if hasattr(image, "shape"):
                h, w = image.shape[:2]
                return int(w), int(h)
            if hasattr(image, "size"):
                w, h = image.size
                return int(w), int(h)
            return None, None

        def logged_run_vlm_batches(batch_dict_by_pixel, has_spotting, vlm_kwargs):
            entries = []
            for pixel_key, batch in batch_dict_by_pixel.items():
                images = batch.get("images", [])
                queries = batch.get("queries", [])
                for idx, image in enumerate(images):
                    width, height = image_shape(image)
                    query = queries[idx] if idx < len(queries) else ""
                    entries.append((pixel_key, width, height, query))

            print(f"[benchmark] VLM input batches: {len(batch_dict_by_pixel)}, requests: {len(entries)}", flush=True)
            for idx, (pixel_key, width, height, query) in enumerate(entries[:limit]):
                pixels = width * height if width is not None and height is not None else None
                print(
                    f"[benchmark] VLM input #{idx}: pixel_key={pixel_key}, size={width}x{height}, pixels={pixels}, query={query!r}",
                    flush=True,
                )
            if len(entries) > limit:
                print(f"[benchmark] VLM input log truncated: {len(entries) - limit} more requests", flush=True)

            return original_run_vlm_batches(batch_dict_by_pixel, has_spotting, vlm_kwargs)

        self.pipeline._paddleocr_vl_run_vl_recognition_batches = logged_run_vlm_batches

    def _predict(self, batch_data):
        results = list(self.pipeline.predict(batch_data, **self.predict_kwargs))
        return "\n\n".join(res._to_markdown(pretty=False)["markdown_texts"] for res in results), len(results)

    def close(self):
        self.pipeline.close()


def monitor_device(gpu_ids, gpu_metrics_list):
    try:
        pynvml.nvmlInit()
        handles = [pynvml.nvmlDeviceGetHandleByIndex(gpu_id) for gpu_id in gpu_ids]

        time.sleep(5)
        while not shutdown:
            try:
                gpu_util = 0
                mem_bytes = 0

                for handle in handles:
                    gpu_util += pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                    mem_bytes += pynvml.nvmlDeviceGetMemoryInfo(handle).used

                gpu_metrics_list.append(
                    {
                        "utilization": gpu_util,
                        "memory": mem_bytes,
                    }
                )
            except Exception as e:
                print(f"Error monitoring GPUs: {e}")

            time.sleep(0.5)

    except Exception as e:
        print(f"Error initializing the GPU monitor: {e}")
    finally:
        try:
            pynvml.nvmlShutdown()
        except:
            pass


def get_curr_time():
    return time.perf_counter()


def new_task_info():
    task_info = {}
    task_info["id"] = uuid.uuid4().hex
    return task_info


def create_and_submit_new_task(executor, requestor, task_info_dict, input_path):
    task_info = new_task_info()
    task = executor.submit(
        requestor.make_request,
        task_info,
        input_path,
    )
    task_info_dict[task] = task_info

    return task


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dirs", type=str, nargs="+", metavar="INPUT_DIR")
    parser.add_argument("-b", "--batch_size", type=int, default=1)
    parser.add_argument("-o", "--output_path", type=str, default="benchmark.json")
    parser.add_argument("--paddlex_config_path", type=str, default="PaddleOCR-VL.yaml")
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=[0])
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--min_pixels", type=int, default=None)
    parser.add_argument("--max_pixels", type=int, default=None)
    parser.add_argument("--ocr_min_pixels", type=int, default=None)
    parser.add_argument("--ocr_max_pixels", type=int, default=None)
    parser.add_argument("--table_min_pixels", type=int, default=None)
    parser.add_argument("--table_max_pixels", type=int, default=None)
    parser.add_argument("--chart_min_pixels", type=int, default=None)
    parser.add_argument("--chart_max_pixels", type=int, default=None)
    parser.add_argument("--formula_min_pixels", type=int, default=None)
    parser.add_argument("--formula_max_pixels", type=int, default=None)
    parser.add_argument("--seal_min_pixels", type=int, default=None)
    parser.add_argument("--seal_max_pixels", type=int, default=None)
    parser.add_argument("--log_vlm_inputs", action="store_true")
    parser.add_argument("--log_vlm_inputs_limit", type=int, default=20)
    args = parser.parse_args()

    task_info_list = []

    all_input_paths = []
    for input_dir in args.input_dirs:
        all_input_paths += glob.glob(os.path.join(input_dir, "*"))
    all_input_paths.sort()
    if len(all_input_paths) == 0:
        print("No valid data")
        sys.exit(1)

    predict_kwargs = {
        name: value
        for name, value in {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
        }.items()
        if value is not None
    }
    vlm_extra_args = {
        name: value
        for name, value in {
            "ocr_min_pixels": args.ocr_min_pixels,
            "ocr_max_pixels": args.ocr_max_pixels,
            "table_min_pixels": args.table_min_pixels,
            "table_max_pixels": args.table_max_pixels,
            "chart_min_pixels": args.chart_min_pixels,
            "chart_max_pixels": args.chart_max_pixels,
            "formula_min_pixels": args.formula_min_pixels,
            "formula_max_pixels": args.formula_max_pixels,
            "seal_min_pixels": args.seal_min_pixels,
            "seal_max_pixels": args.seal_max_pixels,
        }.items()
        if value is not None
    }
    if vlm_extra_args:
        predict_kwargs["vlm_extra_args"] = vlm_extra_args

    predictor = PaddleXPredictor(
        args.paddlex_config_path,
        predict_kwargs=predict_kwargs,
        log_vlm_inputs=args.log_vlm_inputs,
        log_vlm_inputs_limit=args.log_vlm_inputs_limit,
    )

    if args.batch_size < 1:
        print("Invalid batch size")
        sys.exit(2)

    gpu_metrics_list = []
    thread_device_monitor = Thread(
        target=monitor_device,
        args=(args.gpu_ids, gpu_metrics_list),
    )
    thread_device_monitor.start()

    try:
        start_time = get_curr_time()
        batch_data = []
        with open("generated_markdown.md", "w", encoding="utf-8") as f:
            for i, input_path in tqdm(enumerate(all_input_paths), total=len(all_input_paths)):
                batch_data.append(input_path)
                if len(batch_data) == args.batch_size or i == len(all_input_paths) - 1:
                    task_info = new_task_info()
                    try:
                        markdown = predictor.predict(task_info, batch_data)
                        f.write(markdown)
                        f.write("\n\n")
                    except Exception as e:
                        print(e)
                        continue
                    task_info_list.append(task_info)
                    batch_data.clear()
        end_time = get_curr_time()
    finally:
        shutdown = True
        thread_device_monitor.join()
        predictor.close()

    total_files = len(all_input_paths)
    throughput_file = total_files / (end_time - start_time)
    print(f"Throughput (file): {throughput_file:.4f} files per second")
    duration_list_batch = [info["end_time"] - info["start_time"] for info in task_info_list]
    avg_latency_batch = sum(duration_list_batch) / len(duration_list_batch) if duration_list_batch else None
    if avg_latency_batch is not None:
        print(f"Average latency (batch): {avg_latency_batch:.4f} seconds")
    else:
        print("Average latency (batch): n/a")

    successful_files = sum(map(lambda x: x["successful"], task_info_list))
    if successful_files:
        processed_pages = sum(info.get("processed_pages", 0) for info in task_info_list)
        throughput_page = processed_pages / (end_time - start_time)
        print(f"Processed pages: {processed_pages}")
        print(f"Throughput (page): {throughput_page:.4f} pages per second")
        generated_tokens = sum(info.get("generated_tokens", 0) for info in task_info_list)
        throughput_token = generated_tokens / (end_time - start_time)
        print(f"Generated tokens: {generated_tokens}")
        print(f"Throughput (token): {throughput_token:.1f} tokens per second")
    else:
        processed_pages = None
        throughput_page = None
        generated_tokens = None
        throughput_token = None

    if gpu_metrics_list:
        gpu_util_list = list(map(itemgetter("utilization"), gpu_metrics_list))
        print(
            f"GPU utilization (%): {max(gpu_util_list):.1f}, {min(gpu_util_list):.1f}, {sum(gpu_util_list) / len(gpu_util_list):.1f}"
        )
        gpu_mem_list = list(map(itemgetter("memory"), gpu_metrics_list))
        print(
            f"GPU memory usage (MB): {max(gpu_mem_list) / 1024**2:.1f}, {min(gpu_mem_list) / 1024**2:.1f}, {sum(gpu_mem_list) / len(gpu_mem_list) / 1024**2:.1f}"
        )

    dic = {
        "input_dirs": args.input_dirs,
        "batch_size": args.batch_size,
        "predict_kwargs": predict_kwargs,
        "total_files": total_files,
        "throughput_file": throughput_file,
        "avg_latency_batch": avg_latency_batch,
        "duration_list": duration_list_batch,
        "successful_files": successful_files,
        "processed_pages": processed_pages,
        "throughput_page": throughput_page,
        "generated_tokens": generated_tokens,
        "throughput_token": throughput_token,
        "gpu_metrics_list": gpu_metrics_list,
    }
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(
            dic,
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Config and results saved to {args.output_path}")
