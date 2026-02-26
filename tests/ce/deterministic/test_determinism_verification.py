"""
Determinism Feature Verification Test

Reference: test_batch_invariant.py. Verifies whether determinism works correctly.

Usage:
    # Step 1: Start server with determinism disabled
    bash ./tests/ce/deterministic/start_fd.sh 0

    # Step 2: Run non-deterministic test (expected: results differ)
    python ./tests/ce/deterministic/test_determinism_verification.py --phase non-deterministic

    # Step 3: Stop server
    bash fastdeploy/stop.sh

    # Step 4: Start server with determinism enabled and logging ON
    bash ./tests/ce/deterministic/start_fd.sh 1 1

    # Step 5: Run deterministic test (expected: results consistent)
    python ./tests/ce/deterministic/test_determinism_verification.py --phase deterministic

Arguments:
    --phase {deterministic,non-deterministic}
        Test mode
        - deterministic: determinism enabled with logging, expected MD5 consistency
        - non-deterministic: determinism disabled, expected different outputs
    --api-url       API endpoint URL (default: http://localhost:8188/v1/chat/completions)
    --model         Model name (default: Qwen/Qwen2.5-7B)
    --log-file      Server log file path (default: log/workerlog.0)
    --repeat        Number of repeat rounds for non-deterministic phase (default: 3)

Note: The deterministic test requires FD_DETERMINISTIC_LOG_MODE=1 to extract MD5 values
      from logs for verification.
"""

import argparse
import asyncio
import hashlib
import logging
import os
import random
import re
import sys

import aiohttp

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Defaults (overridable via CLI args or env vars)
DEFAULT_API_URL = "http://localhost:8188/v1/chat/completions"
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-7B"
DEFAULT_LOG_FILE = "log/workerlog.0"
DEFAULT_NON_DET_REPEAT = 3

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

# Health check settings
HEALTH_CHECK_INTERVAL = 5
HEALTH_CHECK_TIMEOUT = 300


def parse_args():
    parser = argparse.ArgumentParser(description="Determinism feature verification test")
    parser.add_argument(
        "--phase",
        choices=["deterministic", "non-deterministic"],
        required=True,
        help="Test mode: deterministic (enabled) or non-deterministic (disabled)",
    )
    parser.add_argument(
        "--api-url",
        default=os.environ.get("FD_TEST_API_URL", DEFAULT_API_URL),
        help=f"API endpoint URL (default: {DEFAULT_API_URL})",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("FD_TEST_MODEL", DEFAULT_MODEL_NAME),
        help=f"Model name (default: {DEFAULT_MODEL_NAME})",
    )
    parser.add_argument(
        "--log-file",
        default=os.environ.get("FD_TEST_LOG_FILE", DEFAULT_LOG_FILE),
        help=f"Server log file path (default: {DEFAULT_LOG_FILE})",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=int(os.environ.get("FD_TEST_REPEAT", DEFAULT_NON_DET_REPEAT)),
        help=f"Number of repeat rounds for non-deterministic phase (default: {DEFAULT_NON_DET_REPEAT})",
    )
    return parser.parse_args()


def extract_md5_from_log(log_file: str, request_id: str) -> list[str]:
    """Extract all decode step MD5 values for the specified request from log file."""
    md5_values = []
    try:
        with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
            pattern = rf"\[DETERMINISM-MD5-REQ\] {re.escape(request_id)} \| decode"
            for line in f:
                if re.search(pattern, line):
                    match = re.search(r"hidden_states_md5=([a-f0-9]+)", line)
                    if match:
                        md5_values.append(match.group(1))
    except FileNotFoundError:
        logger.warning("Log file not found: %s", log_file)
    return md5_values


async def wait_for_server(api_url: str) -> None:
    """Wait for the server to be ready by polling the API endpoint."""
    base_url = api_url.rsplit("/v1/", 1)[0]
    health_url = f"{base_url}/v1/models"
    timeout = aiohttp.ClientTimeout(total=10)

    logger.info("Waiting for server to be ready at %s ...", base_url)
    elapsed = 0
    while elapsed < HEALTH_CHECK_TIMEOUT:
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(health_url) as resp:
                    if resp.status == 200:
                        logger.info("Server is ready.")
                        return
        except (aiohttp.ClientError, asyncio.TimeoutError):
            pass
        await asyncio.sleep(HEALTH_CHECK_INTERVAL)
        elapsed += HEALTH_CHECK_INTERVAL
        logger.info("  Still waiting... (%ds/%ds)", elapsed, HEALTH_CHECK_TIMEOUT)

    raise RuntimeError(
        f"Server not ready after {HEALTH_CHECK_TIMEOUT}s. "
        f"Check that the server is running and accessible at {base_url}"
    )


async def send_request(
    session: aiohttp.ClientSession, api_url: str, prompt: str, request_id: str, max_tokens: int, model: str
) -> str:
    """Send request and return response content."""
    request = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.8,
        "top_p": 0.9,
        "max_tokens": max_tokens,
        "request_id": request_id,
    }
    timeout = aiohttp.ClientTimeout(total=300)
    async with session.post(api_url, json=request, timeout=timeout) as response:
        response.raise_for_status()
        result = await response.json()
        return result["choices"][0]["message"]["content"]


async def run_test_case(
    session: aiohttp.ClientSession,
    api_url: str,
    test_name: str,
    test_plan: list[tuple[str, str, bool]],
    model: str,
) -> list[tuple[str, str]]:
    """
    Run a test case.

    Args:
        api_url: API endpoint URL.
        test_plan: List of (request_id, prompt, is_target) tuples.
        model: Model name to use for the request.

    Returns:
        List of (request_id, result) tuples for target requests only.
    """
    target_count = sum(1 for _, _, t in test_plan if t)
    distractor_count = len(test_plan) - target_count
    logger.info(
        "[Test %s] %d requests (target=%d, distractor=%d)", test_name, len(test_plan), target_count, distractor_count
    )

    tasks = []
    for req_id, prompt, is_target in test_plan:
        max_tokens = TARGET_MAX_TOKENS if is_target else random.randint(*DISTRACTOR_MAX_TOKENS_RANGE)
        tasks.append(send_request(session, api_url, prompt, req_id, max_tokens, model))

    results = await asyncio.gather(*tasks)

    target_outputs = []
    for (req_id, _, is_target), result in zip(test_plan, results):
        marker = "[Target]" if is_target else "[Distractor]"
        logger.info("  %s %s: %s...", marker, req_id, result[:50])
        if is_target:
            target_outputs.append((req_id, result))

    return target_outputs


def _print_section(title: str) -> None:
    """Print a section banner."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def _check_consistency(
    items: dict[str, list[str]],
    label: str,
    expect_consistent: bool,
    detail_formatter=None,
) -> bool:
    """
    Unified consistency check logic.

    Args:
        items: Dict mapping unique_key -> list of request_ids sharing that key.
        label: Description label (e.g. "Text", "MD5 Step 1").
        expect_consistent: True expects all keys identical, False expects differences.
        detail_formatter: Optional callable(key) -> str for displaying details on mismatch.

    Returns:
        True if result matches expectation, False otherwise.
    """
    expected_desc = "consistent" if expect_consistent else "inconsistent"
    _print_section(f"{label} Consistency Check (Expected: {expected_desc})")

    if not items:
        logger.warning("No %s values found!", label)
        return False

    is_consistent = len(items) == 1

    print(f"\n  Unique values: {len(items)}")
    if is_consistent:
        key = next(iter(items))
        reqs = items[key]
        print(f"  All {len(reqs)} requests share the same value")
    else:
        for i, (key, reqs) in enumerate(items.items(), 1):
            detail = f" ({detail_formatter(key)})" if detail_formatter else ""
            print(f"  Group {i}: {', '.join(reqs)}{detail}")

    print("-" * 80)

    passed = is_consistent == expect_consistent
    actual_desc = "consistent" if is_consistent else "inconsistent"
    status = "PASS" if passed else "FAIL"
    print(f"  {status}: expected {expected_desc}, actual {actual_desc}")
    print("=" * 80)

    return passed


def compare_text_consistency(target_results: list[tuple[str, str]], expect_consistent: bool = True) -> bool:
    """Compare target request text content against expected consistency."""
    unique_texts: dict[str, list[str]] = {}
    text_map: dict[str, str] = {}
    for req_id, text in target_results:
        text_md5 = hashlib.md5(text.encode("utf-8")).hexdigest()
        unique_texts.setdefault(text_md5, []).append(req_id)
        if text_md5 not in text_map:
            text_map[text_md5] = text

    return _check_consistency(
        unique_texts,
        label="Text",
        expect_consistent=expect_consistent,
        detail_formatter=lambda key: repr(text_map[key][:50]),
    )


def compare_md5_consistency(all_md5: dict[str, list[str]], expect_consistent: bool = True) -> bool:
    """
    Compare MD5 results across ALL decode steps and verify against expected consistency.

    For each decode step, checks that all target requests produced identical hidden_states_md5.
    All steps must be consistent for the overall check to pass.
    """
    if not all_md5:
        logger.warning("No MD5 values found!")
        return False

    # Find the minimum number of decode steps across all requests
    min_steps = min(len(md5s) for md5s in all_md5.values())
    if min_steps == 0:
        logger.warning("Some requests have no decode step MD5 values!")
        return False

    req_ids = list(all_md5.keys())
    logger.info("Checking MD5 consistency across %d decode steps for %d requests", min_steps, len(req_ids))

    failed_steps = []

    for step in range(min_steps):
        step_md5s: dict[str, list[str]] = {}
        for req_id in req_ids:
            md5_val = all_md5[req_id][step]
            step_md5s.setdefault(md5_val, []).append(req_id)

        step_consistent = len(step_md5s) == 1

        if not step_consistent:
            failed_steps.append(step)

        # Print per-step result
        if step_consistent:
            md5_val = next(iter(step_md5s))
            logger.info("  Decode step %d: CONSISTENT (md5=%s)", step + 1, md5_val)
        else:
            logger.warning("  Decode step %d: INCONSISTENT (%d different values)", step + 1, len(step_md5s))
            for md5_val, reqs in step_md5s.items():
                logger.warning("    md5=%s: %s", md5_val, ", ".join(reqs))

    is_consistent = len(failed_steps) == 0

    _print_section(f"MD5 Consistency Check (all {min_steps} decode steps)")
    if is_consistent:
        print(f"  All {min_steps} decode steps are consistent across {len(req_ids)} requests")
    else:
        print(f"  {len(failed_steps)}/{min_steps} decode steps are INCONSISTENT")
        print(f"  Failed steps: {[s + 1 for s in failed_steps]}")
    print("-" * 80)

    passed = is_consistent == expect_consistent
    expected_desc = "consistent" if expect_consistent else "inconsistent"
    actual_desc = "consistent" if is_consistent else "inconsistent"
    status = "PASS" if passed else "FAIL"
    print(f"  {status}: expected {expected_desc}, actual {actual_desc}")
    print("=" * 80)

    return passed


# Test cases: (name, plan) where plan is [(request_id, prompt, is_target)]
TEST_CASES = [
    (
        "case1: Single request (target only)",
        [
            ("case1-target", TARGET_PROMPT, True),
        ],
    ),
    (
        "case2: Two requests (1 target + 1 distractor)",
        [
            ("case2-distract-a", DISTRACTOR_PROMPTS[0], False),
            ("case2-target", TARGET_PROMPT, True),
        ],
    ),
    (
        "case3: Four requests (1 target + 3 distractors)",
        [
            ("case3-distract-a", DISTRACTOR_PROMPTS[0], False),
            ("case3-distract-b", DISTRACTOR_PROMPTS[1], False),
            ("case3-target", TARGET_PROMPT, True),
            ("case3-distract-c", DISTRACTOR_PROMPTS[2], False),
        ],
    ),
    (
        "case4: Six requests (1 target + 5 distractors)",
        [
            ("case4-distract-a", DISTRACTOR_PROMPTS[0], False),
            ("case4-distract-b", DISTRACTOR_PROMPTS[1], False),
            ("case4-distract-c", DISTRACTOR_PROMPTS[2], False),
            ("case4-distract-d", DISTRACTOR_PROMPTS[3], False),
            ("case4-target", TARGET_PROMPT, True),
            ("case4-distract-e", DISTRACTOR_PROMPTS[4], False),
        ],
    ),
]


def _build_test_plan(test_cases, repeat: int = 1):
    """
    Build test plan with optional repetition.

    For repeat > 1, each test case is duplicated with round-suffixed request_ids.
    This increases sample size for non-deterministic testing.
    """
    if repeat <= 1:
        return test_cases

    expanded = []
    for case_name, plan in test_cases:
        for r in range(repeat):
            round_name = f"{case_name} (round {r + 1})"
            round_plan = [(f"{req_id}-r{r + 1}", prompt, is_target) for req_id, prompt, is_target in plan]
            expanded.append((round_name, round_plan))
    return expanded


async def main() -> int:
    args = parse_args()
    is_deterministic = args.phase == "deterministic"

    _print_section("Determinism Feature Verification Test")
    print(f"\n  Test mode: {args.phase}")
    print(f"  API URL: {args.api_url}")
    print(f"  Model: {args.model}")
    print(f"  Log file: {args.log_file}")
    if is_deterministic:
        print("  Expected: All target requests have consistent MD5 values")
    else:
        print(f"  Expected: Target requests produce different outputs (repeat={args.repeat})")
    print("=" * 80)

    # Wait for server to be ready
    await wait_for_server(args.api_url)

    # Build test plan (repeat for non-deterministic to reduce flaky probability)
    repeat = args.repeat if not is_deterministic else 1
    test_plan = _build_test_plan(TEST_CASES, repeat=repeat)

    async with aiohttp.ClientSession() as session:
        all_target_results: list[tuple[str, str]] = []
        for test_name, plan in test_plan:
            target_outputs = await run_test_case(session, args.api_url, test_name, plan, args.model)
            all_target_results.extend(target_outputs)
            await asyncio.sleep(1)

    target_request_ids = [req_id for req_id, _ in all_target_results]

    _print_section("All tests completed, starting verification...")

    if is_deterministic:
        # Deterministic mode: compare MD5 across all decode steps
        all_md5 = {}
        for req_id in target_request_ids:
            md5_values = extract_md5_from_log(args.log_file, req_id)
            if md5_values:
                all_md5[req_id] = md5_values
                logger.info("%s: %d decode steps found", req_id, len(md5_values))
            else:
                logger.warning("%s: No MD5 logs found", req_id)

        if all_md5:
            passed = compare_md5_consistency(all_md5, expect_consistent=True)
        else:
            logger.warning("No MD5 logs found, fallback to text consistency check")
            passed = compare_text_consistency(all_target_results, expect_consistent=True)
    else:
        # Non-deterministic mode: compare text content
        passed = compare_text_consistency(all_target_results, expect_consistent=False)

    _print_section("Final Result")
    if passed:
        print(f"  PASS: {args.phase} mode verified successfully")
    else:
        print(f"  FAIL: {args.phase} mode verification failed")
    print("=" * 80)

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
