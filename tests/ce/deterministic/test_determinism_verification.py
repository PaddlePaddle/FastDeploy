"""
Determinism Feature Verification Test

Reference: test_batch_invariant.py. Verifies whether determinism works correctly.

Usage:
    # Step 1: Start server with determinism disabled
    bash ./tests/determinitic/start_fd.sh 0

    # Step 2: Run non-deterministic test (expected: results differ)
    python ./tests/determinitic/test_determinism_verification.py --phase non-deterministic

    # Step 3: Stop server
    bash fastdeploy/stop.sh

    # Step 4: Start server with determinism enabled and logging ON
    bash ./tests/determinitic/start_fd.sh 1 1

    # Step 5: Run deterministic test (expected: results consistent)
    python ./tests/determinitic/test_determinism_verification.py --phase deterministic

Arguments:
    --phase {deterministic,non-deterministic}
        Test mode
        - deterministic: determinism enabled with logging, expected MD5 consistency
        - non-deterministic: determinism disabled, expected different outputs

Note: The deterministic test requires FD_DETERMINISTIC_LOG_MODE=1 to extract MD5 values
      from logs for verification.
"""

import argparse
import asyncio
import hashlib
import random
import re
import sys

import aiohttp

API_URL = "http://localhost:8188/v1/chat/completions"
MODEL_NAME = "Qwen/Qwen2.5-7B"
LOG_FILE = "log/workerlog.0"

# Target prompt (we care about its determinism)
TARGET_PROMPT = "你好，请简单介绍一下自己。"

# Distractor prompts (different content, used to create batch interference)
DISTRACTOR_PROMPTS = [
    "今天天气怎么样？",
    "什么是人工智能？",
    "如何学习编程？",
    "什么是机器学习？",
    "Python 是什么？",
]

# Generation length for target prompt (fixed, longer)
TARGET_MAX_TOKENS = 128

# Generation length range for distractor prompts
DISTRACTOR_MAX_TOKENS_RANGE = (8, 32)


def parse_args():
    parser = argparse.ArgumentParser(description="Determinism feature verification test")
    parser.add_argument(
        "--phase",
        choices=["deterministic", "non-deterministic"],
        required=True,
        help="Test mode: deterministic (enabled) or non-deterministic (disabled)",
    )
    return parser.parse_args()


def extract_md5_from_log(log_file: str, request_id: str) -> list[str]:
    """Extract all decode step MD5 values for the specified request from log file."""
    md5_values = []
    try:
        with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
            pattern = rf"DETERMINISM-MD5-REQ\] chatcmpl-{request_id}_[0-9]+ \| decode"
            for line in f:
                if re.search(pattern, line):
                    match = re.search(r"hidden_states_md5=([a-f0-9]+)", line)
                    if match:
                        md5_values.append(match.group(1))
    except FileNotFoundError:
        pass
    return md5_values


async def send_request(session: aiohttp.ClientSession, prompt: str, request_id: str, max_tokens: int) -> str:
    """Send request and return response content."""
    request = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.8,
        "top_p": 0.9,
        "max_tokens": max_tokens,
        "request_id": request_id,
    }
    async with session.post(API_URL, json=request, timeout=300) as response:
        response.raise_for_status()
        result = await response.json()
        return result["choices"][0]["message"]["content"]


async def run_test_case(
    session: aiohttp.ClientSession, test_name: str, test_plan: list[tuple[str, str, bool]]
) -> list[tuple[str, str, bool]]:
    """
    Run a test case.

    Args:
        test_plan: List of (request_id, prompt, is_target) tuples.
        is_target: True indicates this is the target prompt to verify.

    Returns:
        List of (request_id, result, is_target) tuples.
    """
    target_count = sum(1 for _, _, is_target in test_plan if is_target)
    distractor_count = len(test_plan) - target_count
    print(f"\n[Test {test_name}] {len(test_plan)} requests")
    print(f"  Target prompts: {target_count}")
    print(f"  Distractor prompts: {distractor_count}")

    tasks = []
    for req_id, prompt, is_target in test_plan:
        max_tokens = TARGET_MAX_TOKENS if is_target else random.randint(*DISTRACTOR_MAX_TOKENS_RANGE)
        tasks.append(send_request(session, prompt, req_id, max_tokens))

    results = await asyncio.gather(*tasks)

    output = []
    for (req_id, _, is_target), result in zip(test_plan, results):
        marker = "[Target]" if is_target else "[Distractor]"
        print(f"  {marker} {req_id}: {result[:50]}...")
        output.append((req_id, result, is_target))

    return output


def compare_text_consistency(target_results: list[tuple[str, str, bool]], expect_consistent: bool = True) -> bool:
    """
    Compare target request text content against expected consistency.

    Args:
        target_results: List of (req_id, text, is_target) tuples (target only).
        expect_consistent: True expects consistency, False expects inconsistency.

    Returns:
        True if result matches expectation, False otherwise.
    """
    print("\n" + "=" * 80)
    if expect_consistent:
        print("Text Consistency Check (Expected: All target requests consistent)")
    else:
        print("Text Inconsistency Check (Expected: Different results exist)")
    print("=" * 80)

    if not target_results:
        print("[WARNING] No target request results found!")
        return False

    print(f"\n{len(target_results)} target request results:\n")

    # Calculate MD5 for each text to deduplicate
    unique_texts: dict[str, list[str]] = {}
    for req_id, text, _ in target_results:
        preview = text[:80] if text else "(empty)"
        print(f"  {req_id}: {preview}")
        text_md5 = hashlib.md5(text.encode("utf-8")).hexdigest()
        unique_texts.setdefault(text_md5, []).append(req_id)

    print("\n" + "-" * 80)
    is_consistent = len(unique_texts) == 1

    if is_consistent:
        print("Result: PASS All target requests consistent")
    else:
        print(f"Result: FAIL Found {len(unique_texts)} different results")
        for i, (md5, reqs) in enumerate(unique_texts.items(), 1):
            text = next(text for req_id, text, _ in target_results if req_id == reqs[0])
            preview = text[:50]
            print(f"  {i}. Result {preview!r}: {', '.join(reqs)}")

    print("=" * 80)

    passed = is_consistent == expect_consistent
    print(f"\nVerification: {'PASS' if passed else 'FAIL'}", end="")
    if expect_consistent:
        print(f" (expected consistent, actually {'consistent' if is_consistent else 'inconsistent'})")
    else:
        print(f" (expected inconsistent, actually {'inconsistent' if not is_consistent else 'consistent'})")

    return passed


def compare_md5_consistency(all_md5: dict[str, list[str]], expect_consistent: bool = True) -> bool:
    """
    Compare MD5 results and verify against expected consistency.

    Args:
        all_md5: Dict mapping request_id to [md5_1, md5_2, ...].
        expect_consistent: True expects consistency, False expects inconsistency.

    Returns:
        True if result matches expectation, False otherwise.
    """
    print("\n" + "=" * 80)
    print("MD5 Consistency Check (Expected: All target requests MD5 consistent)")
    print("=" * 80)

    if not all_md5:
        print("[WARNING] No MD5 values found!")
        return False

    # Collect first decode step MD5 for each request
    step_1_md5s = {req_id: md5s[0] for req_id, md5s in all_md5.items() if md5s}

    if not step_1_md5s:
        print("[WARNING] No decode step MD5 values found!")
        return False

    print(f"\n{len(step_1_md5s)} target requests Decode Step 1 MD5:\n")
    for req_id, md5 in step_1_md5s.items():
        print(f"  {req_id}: {md5}")

    unique_md5s = set(step_1_md5s.values())
    is_consistent = len(unique_md5s) == 1

    print("\n" + "-" * 80)
    if is_consistent:
        print(f"Result: PASS All target requests MD5 consistent ({list(unique_md5s)[0]})")
    else:
        print(f"Result: FAIL Found {len(unique_md5s)} different MD5 values")
        for i, md5 in enumerate(unique_md5s, 1):
            reqs = [req_id for req_id, m in step_1_md5s.items() if m == md5]
            print(f"  {i}. MD5={md5}: {', '.join(reqs)}")
    print("=" * 80)

    passed = is_consistent == expect_consistent
    print(f"\nVerification: {'PASS' if passed else 'FAIL'}", end="")
    if expect_consistent:
        print(f" (expected consistent, actually {'consistent' if is_consistent else 'inconsistent'})")
    else:
        print(f" (expected inconsistent, actually {'inconsistent' if not is_consistent else 'consistent'})")

    return passed


async def main() -> int:
    args = parse_args()
    is_deterministic = args.phase == "deterministic"

    print("=" * 80)
    print("Determinism Feature Verification Test")
    print("=" * 80)
    print(f"\nTest mode: {args.phase}")
    if is_deterministic:
        print("Expected: All target requests have consistent MD5 values")
        print("Verification: Extract MD5 values from logs for comparison")
    else:
        print("Expected: All target requests have different outputs")
        print("Verification: Compare generated text content")
    print("=" * 80)

    # Define test cases (consistent with test_batch_invariant.py)
    test_cases = [
        (
            "1: Single request (target only)",
            [
                ("target-1", TARGET_PROMPT, True),
            ],
        ),
        (
            "2: Two requests (1 target + 1 distractor, target at position 2)",
            [
                ("distract-2a", DISTRACTOR_PROMPTS[0], False),
                ("target-2", TARGET_PROMPT, True),
            ],
        ),
        (
            "3: Four requests (1 target + 3 distractors, target at position 3)",
            [
                ("distract-4a", DISTRACTOR_PROMPTS[0], False),
                ("distract-4b", DISTRACTOR_PROMPTS[1], False),
                ("target-4", TARGET_PROMPT, True),
                ("distract-4c", DISTRACTOR_PROMPTS[2], False),
            ],
        ),
        (
            "4: Six requests (1 target + 5 distractors, target at position 5)",
            [
                ("distract-6a", DISTRACTOR_PROMPTS[0], False),
                ("distract-6b", DISTRACTOR_PROMPTS[1], False),
                ("distract-6c", DISTRACTOR_PROMPTS[2], False),
                ("distract-6d", DISTRACTOR_PROMPTS[3], False),
                ("target-6", TARGET_PROMPT, True),
                ("distract-6e", DISTRACTOR_PROMPTS[4], False),
            ],
        ),
    ]

    async with aiohttp.ClientSession() as session:
        all_results = []
        for test_name, test_plan in test_cases:
            results = await run_test_case(session, test_name, test_plan)
            all_results.extend(results)
            await asyncio.sleep(1)

    # Extract target request results
    target_results = [(req_id, text, is_target) for req_id, text, is_target in all_results if is_target]
    target_request_ids = [req_id for req_id, _, _ in target_results]

    print("\n" + "=" * 80)
    print("All tests completed, starting verification...")
    print("=" * 80)

    if is_deterministic:
        # Deterministic mode: compare MD5
        all_md5 = {}
        for req_id in target_request_ids:
            md5_values = extract_md5_from_log(LOG_FILE, req_id)
            if md5_values:
                all_md5[req_id] = md5_values
                print(f"{req_id}: {len(md5_values)} decode steps")
            else:
                print(f"[WARNING] {req_id}: No MD5 logs found")

        if all_md5:
            passed = compare_md5_consistency(all_md5, expect_consistent=True)
        else:
            print("\n[WARNING] No MD5 logs found, fallback to text consistency check")
            passed = compare_text_consistency(target_results, expect_consistent=True)
    else:
        # Non-deterministic mode: compare text content
        passed = compare_text_consistency(target_results, expect_consistent=False)

    print("\n" + "=" * 80)
    if passed:
        print(f"PASS Test passed! {args.phase} mode verified successfully")
    else:
        print(f"FAIL Test failed! {args.phase} mode verification failed")
    print("=" * 80)

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
