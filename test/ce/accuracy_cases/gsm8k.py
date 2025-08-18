#!/bin/env python3
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python


import os
import re
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse, urlunparse

import openai
from datasets import load_dataset
from tqdm import tqdm

BASELINE = {
    "0.3B": 0.05,
    "21B": 0.49,
    "300B": 0.96,
}
baseline = BASELINE.get(os.environ.get("MODEL_SIZE"), None)
base_url = os.environ.get("URL", None)
atol = 0.03
if baseline is None:
    raise ValueError(
        f"Invalid MODEL_SIZE value '{os.environ.get('MODEL_SIZE')}', expected one of {list(BASELINE.keys())}"
    )
if base_url is None:
    raise ValueError(
        "Environment variable 'URL' is not set. "
        "Please specify the inference service address, e.g., 'http://localhost:8191/v1'."
    )


def strip_path_suffix(url: str, suffix: str = "chat/completions") -> str:
    """
    去除 URL 中的指定路径后缀（如 chat/completions）
    """
    parsed = urlparse(url)
    # 移除末尾的 suffix（注意确保只移除结尾部分）
    if parsed.path.endswith("/" + suffix):
        new_path = parsed.path[: -(len(suffix) + 1)]  # +1 是斜杠
    else:
        new_path = parsed.path
    # 重新构造 URL
    cleaned_url = urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            new_path.rstrip("/"),  # 去掉末尾的斜杠
            "",
            "",
            "",  # 忽略 params/query/fragment
        )
    )
    return cleaned_url


# ========== OpenAI 客户端配置 ==========
client = openai.OpenAI(
    api_key="DDDivano",
    # base_url="http://占位:8187/v1"
    base_url=strip_path_suffix(base_url),
)

model_name = "eb"
max_samples = 690
max_tokens = 12288
max_workers = 33

# ========== 加载数据集 ==========
dataset = load_dataset("parquet", data_files="gsm8k.parquet", split="train")
dataset = dataset.select(range(min(len(dataset), max_samples)))


# ========== 提取 GT 中 "#### 数字" 格式的最终答案 ==========
def extract_gt_answer(text):
    match = re.search(r"####\s*([\d,]+(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "").strip()
    return None


# ========== 提取模型输出中的“最后一句话”中的数字 ==========
def extract_model_answer(text):
    if not text:
        return None
    text = text.replace(",", "").replace("$", "")
    lines = text.strip().splitlines()
    last_line = lines[-1] if lines else text
    match = re.search(r"-?\d+(?:\.\d+)?", last_line)
    return match.group(0) if match else None


# ========== 数值比较函数 ==========
def is_answer_equal(pred, gt, tol=1e-6):
    if pred is None or gt is None:
        return False
    try:
        return abs(float(pred) - float(gt)) < tol
    except:
        return pred == gt


# ========== 构造 Prompt ==========
def build_prompt(sample):
    return f"以下是一个数学问题，请直接给出最终答案。一定要把最终答案数字在最后输出。\n\n问题：{sample['question']}\n\n答案："


# ========== 模型请求函数 ==========
def query_model(prompt):
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "你是一个数学专家，擅长严谨地解答数学问题。"},
                {"role": "user", "content": prompt},
            ],
            temperature=1.0,
            top_p=0.8,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Error] {e}, {str(traceback.format_exc())}"


# ========== 评估函数 ==========
def evaluate_sample(sample):
    prompt = build_prompt(sample)
    model_output = query_model(prompt)

    gt_value = extract_gt_answer(sample["answer"])
    pred_value = extract_model_answer(model_output)
    is_correct = is_answer_equal(pred_value, gt_value)

    result = {
        "question": sample["question"],
        "gt_answer": gt_value,
        "model_answer": pred_value,
        "raw_gt_answer": sample["answer"],
        "raw_model_output": model_output,
        "is_correct": is_correct,
    }

    return result


# ========== 主流程 ==========

acc = []
times = 3

for i in range(times):
    correct = 0
    total = 0
    results = []

    print(f"🚀 Starting evaluation with {max_workers} threads...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(evaluate_sample, sample) for sample in dataset]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
            result = future.result()
            results.append(result)
            total += 1
            if result["is_correct"]:
                correct += 1
            else:
                print("\n❌ Wrong prediction:")
                print(f"Q: {result['question']}")
                print(f"GT: {result['gt_answer']}")
                print(f"Model: {result['model_answer']}")
                print(f"Full GT: {result['raw_gt_answer']}")
                print(f"Model Output: {result['raw_model_output']}")

    # ========== 输出准确率 ==========
    accuracy = correct / total * 100 if total > 0 else 0.0
    print(f"\n🎯 Evaluation Complete: Accuracy = {accuracy:.2f}% ({correct}/{total})")
    acc.append(accuracy)

avg_acc = round(sum(acc) / times / 100, 4)  # 优化百分数
print(f"平均准确率：{avg_acc * 100:.2f}%")

assert (
    abs(avg_acc - baseline) <= atol
), f"模型准确率 {avg_acc:.2f} 与基准 {baseline:.2f} 相差 {abs(avg_acc - baseline):.2f}，超出容忍范围 {atol:.2f}"

# with open("eval_result_math.json", "w", encoding="utf-8") as f:
#     json.dump(results, f, indent=2, ensure_ascii=False)
