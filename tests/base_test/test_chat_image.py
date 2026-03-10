# -*- coding: utf-8 -*-
"""
FD 服务 Chat Completion 全参数测试
重点：针对服务能力，不检查模型是否聪明。
usage:
    python -m pytest test_maoyan.py
"""
import json
import os
import re
from collections import Counter

import pytest
from utils import send_request, send_request_stream

# ===================== 基础配置 =====================
# 如用本地桩，保持不动；如指向真实 FD，直接改 URL
# URL = "http://10.95.237.204:1211/v1/chat/completions"
# URL = "http://10.63.64.38:9980/v1/chat/completions"
# URL = "http://10.174.137.88:8801/v1/chat/completions"
# 从环境变量读取，默认回退值（可选）
URL_HOST = os.getenv("URL_HOST", "10.174.136.93")
URL_PORT = os.getenv("URL_PORT", "8180")

URL = f"http://{URL_HOST}:{URL_PORT}/v1/chat/completions"
print(f"FD URL: {URL}")

MODEL = "default"  # 内部 FD 固定模型名
image_url_1 = "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example2.jpg"
image_url_2 = "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example1.jpg"


# ===================== 基础冒烟 =====================
def test_minimal_success():
    """最小必填集"""
    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url_1}},
                    {"type": "text", "text": "图中的文物属于哪个年代?"},
                ],
            }
        ],
    }
    r = send_request(URL, data)
    print(json.dumps(data, ensure_ascii=False))
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["object"] == "chat.completion"
    assert len(j["choices"]) == 1
    assert j["choices"][0]["message"]["content"]
    assert j["usage"]["prompt_tokens"] > 0
    assert j["usage"]["completion_tokens"] >= 0
    assert j["usage"]["total_tokens"] == j["usage"]["prompt_tokens"] + j["usage"]["completion_tokens"]


# ===================== messages 各种写法 =====================
@pytest.mark.parametrize("role", ["system", "user", "assistant"])
def test_single_message(role):
    data = {
        "model": MODEL,
        "messages": [
            {
                "role": role,
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url_1}},
                    {"type": "text", "text": "图中的文物属于哪个年代?"},
                ],
            }
        ],
    }
    j = send_request(URL, data).json()
    # 服务永远返回 assistant
    assert j["choices"][0]["message"]["role"] == "assistant"
    assert j["choices"][0]["message"]["content"]


def test_multi_turn():
    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url_1}},
                    {"type": "text", "text": "图中的文物属于哪个年代?"},
                ],
            },
            {"role": "assistant", "content": "我不清楚"},
            {"role": "user", "content": "你再看看"},
        ],
    }
    j = send_request(URL, data).json()
    # 验证多轮对话正常
    assert len(j["choices"]) == 1
    assert j["choices"][0]["message"]["content"]


def test_more_image():
    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url_1}},
                    {"type": "image_url", "image_url": {"url": image_url_2}},
                    {"type": "text", "text": "图片内容是什么?简短回复我"},
                ],
            },
            {"role": "assistant", "content": "我不清楚"},
            {"role": "user", "content": "你再看看"},
        ],
        "max_completion_tokens": 50,
    }
    j = send_request(URL, data).json()
    print(json.dumps(j, ensure_ascii=False))
    # 验证多图对话正常
    assert len(j["choices"]) == 1
    assert j["choices"][0]["message"]["content"]


def test_message_with_name():
    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url_1}},
                    {"type": "text", "text": "图中的文物属于哪个年代?"},
                ],
                "name": "bob",
            }
        ],
    }
    j = send_request(URL, data).json()
    # ===== 基础结构校验 =====
    assert "choices" in j
    assert isinstance(j["choices"], list)
    assert len(j["choices"]) == 1

    choice = j["choices"][0]

    # ===== message schema 校验 =====
    assert "message" in choice
    message = choice["message"]

    assert message["role"] == "assistant"
    assert "content" in message
    assert isinstance(message["content"], str)
    assert message["content"].strip() != ""

    # ===== finish_reason 校验（防止异常中断）=====
    assert "finish_reason" in choice
    assert choice["finish_reason"] in ("stop", "length")


# ===================== temperature =====================
@pytest.mark.parametrize("t", [0, 0.1, 0.5, 1, 1.5, 2])
def test_temperature_boundary(t):
    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url_1}},
                    {"type": "text", "text": "图中的文物属于哪个年代?"},
                ],
            }
        ],
        "temperature": t,
        "max_completion_tokens": 200,
    }
    j = send_request(URL, data).json()
    print(json.dumps(j, ensure_ascii=False))
    choice = j["choices"][0]
    assert j["choices"][0]["message"]["content"].strip() != ""
    # ===== 推理正常结束校验 =====
    assert choice["finish_reason"] in ("stop", "length")


# ===================== top_p =====================
@pytest.mark.parametrize("p", [0.01, 0.5, 1, 1.5])
def test_top_p_boundary(p):
    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url_1}},
                    {"type": "text", "text": "图中的文物属于哪个年代?"},
                ],
            }
        ],
        "top_p": p,
    }
    j = send_request(URL, data).json()
    print(json.dumps(j, ensure_ascii=False))
    if p <= 1:
        choice = j["choices"][0]
        assert j["choices"][0]["message"]["content"].strip() != ""
        # ===== 推理正常结束校验 =====
        assert choice["finish_reason"] in ("stop", "length")
    else:
        err = j["error"]
        assert err.get("param") == "top_p", f"param 字段错误: {err.get('param')}"
        assert err.get("message") == "Input should be less than or equal to 1", "错误提示不符合预期"


# ===================== seed + top_p 组合 =====================
# @pytest.mark.flaky(reruns=3, reruns_delay=1)
def test_seed_with_top_p():
    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url_1}},
                    {"type": "text", "text": "图中的文物属于哪个年代?"},
                ],
            }
        ],
        "seed": 24,
        "top_p": 0,
    }
    j1 = send_request(URL, data).json()
    j2 = send_request(URL, data).json()
    print(json.dumps(j1, ensure_ascii=False))
    print(json.dumps(j2, ensure_ascii=False))
    assert (
        j1["choices"][0]["message"]["content"] == j2["choices"][0]["message"]["content"]
    ), "top_p=0, 固定seed, 两次请求结果不一致"


# ===================== temperature + top_p 组合 =====================
@pytest.mark.parametrize("t,p", [(0.3, 0.9), (0.8, 0.3)])
def test_temp_with_top_p(t, p):
    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url_1}},
                    {"type": "text", "text": "图中的文物属于哪个年代?"},
                ],
            }
        ],
        "temperature": t,
        "top_p": p,
    }
    j = send_request(URL, data).json()
    print(json.dumps(j, ensure_ascii=False))
    assert j["choices"][0]["message"]["content"].strip() != ""
    # ===== 推理正常结束 =====
    assert j["choices"][0]["finish_reason"] in ("stop", "length")
    # 语义稳定性断言
    # =========================
    weak_semantic_keywords = [
        "无法",
        "不能确定",
        "难以判断",
        "年代",
        "时期",
        "历史",
        "图中",
        "文物",
        "年",
        "世纪",
        "朝",
        "代",
    ]
    dynasty_keywords = [
        "新石器",
        "旧石器",
        "夏",
        "商",
        "西周",
        "东周",
        "春秋",
        "战国",
        "秦",
        "西汉",
        "东汉",
        "汉代",
        "三国",
        "魏晋",
        "两晋",
        "南北朝",
        "隋",
        "唐",
        "五代十国",
        "宋",
        "北宋",
        "南宋",
        "辽",
        "金",
        "元",
        "明",
        "清",
        "近代",
        "民国",
        "现代",
    ]

    ALL_TIME_KEYWORDS = weak_semantic_keywords + dynasty_keywords
    assert any(k in j["choices"][0]["message"]["content"] for k in ALL_TIME_KEYWORDS), "回答完全偏离问题语境"


# ===================== n =====================
@pytest.mark.parametrize("n", [1, 2, 3])
def test_n_returns(n):
    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url_1}},
                    {"type": "text", "text": "图中的文物属于哪个年代?"},
                ],
            }
        ],
        "n": n,
    }
    j = send_request(URL, data).json()
    assert len(j["choices"]) == n
    for ch in j["choices"]:
        assert ch["message"]["content"]
        assert ch["index"] in range(n)


# ===================== stop =====================
@pytest.mark.parametrize("stop", ["的", ["。", "是"], ["，"]])
def test_stop_variants(stop):
    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url_1}},
                    {"type": "text", "text": "图中的文物属于哪个年代?"},
                ],
            }
        ],
        "stop": stop,
    }
    j = send_request(URL, data).json()
    if isinstance(stop, list):
        assert any(
            j["choices"][0]["message"]["content"].endswith(s) for s in stop
        ), f'输出未按 stop 截断: stop={stop}, content={j["choices"][0]["message"]["content"]!r}'
    else:
        assert j["choices"][0]["message"]["content"].endswith(
            stop
        ), f'输出未按 stop 截断: stop={stop}, content={j["choices"][0]["message"]["content"]!r}'


# ===================== max_completion_tokens =====================
@pytest.mark.parametrize("max_t", [1, 16, 64])
def test_max_completion_tokens(max_t):
    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url_1}},
                    {"type": "text", "text": "图中的文物属于哪个年代?"},
                ],
            }
        ],
        "max_completion_tokens": max_t,
    }
    j = send_request(URL, data).json()
    assert j["usage"]["completion_tokens"] <= max_t


# ===================== stream =====================
def test_stream_false():
    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url_1}},
                    {"type": "text", "text": "图中的文物属于哪个年代?"},
                ],
            }
        ],
        "stream": False,
    }
    j = send_request(URL, data).json()
    assert j["object"] == "chat.completion"
    assert isinstance(j["choices"], list)


# ===================== penalties =====================
@pytest.mark.parametrize("pp,fp", [(0, 0), (0.8, 0), (0, 0.8), (0.8, 0.8), (1, -1), (2, -2)])
def test_penalties(pp, fp):
    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url_1}},
                    {"type": "text", "text": "描述这个文物"},
                ],
            }
        ],
        "presence_penalty": pp,
        "frequency_penalty": fp,
        "max_completion_tokens": 50,
        # 明确开启采样，penalty 才有意义
        "temperature": 0.8,
        "top_p": 0.9,
    }

    j = send_request(URL, data).json()
    content = j["choices"][0]["message"]["content"]

    # ===== 协议 / 基础行为断言 =====
    assert isinstance(content, str)
    assert content.strip() != ""
    assert j["choices"][0]["finish_reason"] in ("stop", "length")

    # ===== 行为级指标：重复度 =====
    chars = [c for c in content if not c.isspace()]
    if not chars:
        pytest.skip("输出过短，无法评估重复度")

    counter = Counter(chars)
    max_freq = counter.most_common(1)[0][1]
    repeat_ratio = max_freq / len(chars)

    # ===== penalty 生效的“趋势断言” =====
    # 不和具体文本比，只限制“不能极端重复”
    if pp > 0 or fp > 0:
        assert repeat_ratio < 0.5, f"penalty={pp, fp} 下重复度仍然过高: {repeat_ratio}, content={content!r}"


# ===================== logprobs & top_logprobs =====================
@pytest.mark.parametrize("log,top", [(True, 0), (True, 5), (False, None)])
def test_logprobs(log, top):
    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url_1}},
                    {"type": "text", "text": "图中的文物属于哪个年代?"},
                ],
            }
        ],
        "logprobs": log,
        "top_logprobs": top,
    }
    j = send_request(URL, data).json()
    # print(json.dumps(j, ensure_ascii=False))
    ch = j["choices"][0]
    if log:
        assert ch["logprobs"] is not None
        assert "content" in ch["logprobs"]
        for tok_info in ch["logprobs"]["content"]:
            assert "token" in tok_info
            assert isinstance(tok_info["logprob"], (int, float))
            assert isinstance(tok_info["bytes"], list)
            assert len(tok_info["top_logprobs"]) == top
    else:
        assert ch["logprobs"] is None


# ===================== return_token_ids =====================
def test_return_token_ids():
    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url_1}},
                    {"type": "text", "text": "图中的文物属于哪个年代?"},
                ],
            }
        ],
        "return_token_ids": True,
    }
    j = send_request(URL, data).json()
    msg = j["choices"][0]["message"]
    assert isinstance(msg["prompt_token_ids"], list)
    assert isinstance(msg["completion_token_ids"], list)


# ===================== user =====================
def test_user_field():
    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url_1}},
                    {"type": "text", "text": "图中的文物属于哪个年代?"},
                ],
            }
        ],
        "user": "alice123",
    }
    j = send_request(URL, data).json()
    assert j["choices"][0]["message"]["content"]
    assert data["user"] in j["id"]


# ===================== service_tier =====================
@pytest.mark.parametrize("tier", ["auto", "default", "flex"])
def test_service_tier(tier):
    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url_1}},
                    {"type": "text", "text": "图中的文物属于哪个年代?"},
                ],
            }
        ],
        "service_tier": tier,
    }
    j = send_request(URL, data).json()
    assert j["choices"][0]["message"]["content"]


# ===================== metadata+min_tokens =====================
def test_metadata():
    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url_1}},
                    {"type": "text", "text": "图中的文物属于哪个年代?"},
                ],
            }
        ],
        "min_tokens": 500,
    }
    j = send_request(URL, data).json()
    # assert len(j["choices"][0]["message"]["content"]) >=500
    assert j["usage"]["completion_tokens"] >= 500


# ===================== store =====================
def test_store_true():
    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url_1}},
                    {"type": "text", "text": "图中的文物属于哪个年代?"},
                ],
            }
        ],
        "store": True,
    }
    j = send_request(URL, data).json()
    print(json.dumps(j, ensure_ascii=False))
    assert j["choices"][0]["message"]["content"]


# ===================== 大组合 =====================
def test_big_combo():
    req = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url_1}},
                    {"type": "text", "text": "图中的文物属于哪个年代?"},
                ],
            }
        ],
        "temperature": 0.8,
        "top_p": 0.95,
        "n": 2,
        "max_completion_tokens": 3,
        # "max_completion_tokens":3,
        "stop": ["do not"],  # 故意留一个几乎撞不到的 stop
        "presence_penalty": 1,
        "frequency_penalty": -1,
        "logprobs": True,
        "top_logprobs": 5,
        "return_token_ids": True,
        "user": "combo",
        "service_tier": "flex",
        "metadata": {"k": "v"},
        "store": True,
    }
    resp = send_request(URL, req).json()
    # print(json.dumps(resp, indent=2, ensure_ascii=False))

    # --------------- 0. 顶级骨架 ---------------
    assert resp.get("object") == "chat.completion"
    # assert resp.get("model") == MODEL
    assert isinstance(resp.get("id"), str) and resp["id"]
    assert isinstance(resp.get("created"), int) and resp["created"] > 0
    usage = resp.get("usage")
    assert isinstance(usage, dict)
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    # --------------- 1. choices 长度 ---------------
    choices = resp.get("choices")
    assert isinstance(choices, list) and len(choices) == req["n"]

    # --------------- 2. 每条 choice 必传字段 ---------------
    for idx, ch in enumerate(choices):
        assert ch.get("index") == idx
        assert ch.get("finish_reason") in {"stop", "length"}

        msg = ch.get("message")
        assert isinstance(msg, dict)
        assert msg.get("role") == "assistant"
        content = msg.get("content")
        # 可见字符断言：至少一个非空白字符
        assert isinstance(content, str) and re.search(r"\S", content, re.U), f"choice[{idx}] content 只有空白或为空"

        # --------------- 3. token 级特征 ---------------
        assert isinstance(msg.get("prompt_token_ids"), list)
        assert isinstance(msg.get("completion_token_ids"), list)
        assert len(msg["completion_token_ids"]) > 0
        # 不能超过用户给的硬上限
        assert len(msg["completion_token_ids"]) <= req["max_completion_tokens"]

        # --------------- 4. logprobs 深度校验 ---------------
        logprobs = ch.get("logprobs", {})
        content_lp = logprobs.get("content")
        assert isinstance(content_lp, list) and len(content_lp) == len(msg["completion_token_ids"])
        for tok_lp in content_lp:
            assert isinstance(tok_lp.get("token"), str) and tok_lp["token"]
            assert isinstance(tok_lp.get("logprob"), (int, float))
            assert isinstance(tok_lp.get("bytes"), list)
            top5 = tok_lp.get("top_logprobs")
            assert isinstance(top5, list) and len(top5) == req["top_logprobs"]
            for cand in top5:
                assert isinstance(cand.get("token"), str)
                assert isinstance(cand.get("logprob"), (int, float))

        # --------------- 5. stop 条件校验 ---------------
        # 如果 finish_reason==stop，则最后 token 必须落在 req["stop"] 之一
        if ch["finish_reason"] == "length":
            tokens = msg["completion_token_ids"]
            assert (
                len(tokens) == req["max_completion_tokens"]
            ), f"choice[{idx}] 宣称 length 结束，但 token 数未顶满上限"
            # last_token = content_lp[-1]["token"]
            # assert any(stop in last_token for stop in req["stop"]), \
            #     f"choice[{idx}] 宣称 stop，但最后一个 token '{last_token}' 不在 stop 列表里"

    # --------------- 6. 参数“被吃”校验 ---------------
    # 只要 penalty 绝对值>0，模型就应该在 logprobs 里体现出不同 token 的得分差异；
    # 这里用“最小熵”简单兜底：如果所有 top5 的 logprob 完全一致，说明 penalty 大概率没生效
    for ch in choices:
        lp = ch["logprobs"]["content"]
        scores = [t["logprob"] for t in lp]
        assert max(scores) != min(scores), "penalty 似乎未生效，所有 token 得分相同"

    # --------------- 7. 冗余字段 ---------------
    #
    assert req["user"] in resp["id"]


# ===================== 空 image =====================
def test_empty_messages():
    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": ""}}, {"type": "text", "text": ""}],
            }
        ],
    }
    j = send_request(URL, data).json()
    print(json.dumps(j, ensure_ascii=False, indent=2))
    assert j.get("error", {}).get("type") == "invalid_request_error"


# ===================== 超长 content =====================
def test_very_long_content():
    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url_1}},
                    {"type": "text", "text": "图中的文物属于哪个年代?" * 1000},
                ],
            }
        ],
        "max_completion_tokens": 100,
    }
    j = send_request(URL, data).json()
    print(json.dumps(j, ensure_ascii=False))
    assert j["choices"][0]["message"]["content"]


# ===================== 特殊字符 =====================
def test_special_chars():
    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url_1}},
                    {"type": "text", "text": "图中的文物属于哪个年代?😀\n\t\r"},
                ],
            }
        ],
    }
    j = send_request(URL, data).json()
    assert j["choices"][0]["message"]["content"].strip() != ""


# ------------------------------------------------------
# 1. 最简流式
# ------------------------------------------------------
def test_stream_minimal():
    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url_1}},
                    {"type": "text", "text": "图中的文物属于哪个年代?"},
                ],
            }
        ],
        "stream": True,
    }
    chunks, text, usage, _ = send_request_stream(URL, data)
    print(json.dumps(chunks, ensure_ascii=False))
    assert len(chunks) >= 2, "至少首包+末包"
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert chunks[-1]["choices"][0]["finish_reason"] in {"stop", "length"}
    assert text.strip(), "返回内容不能为空"


# ------------------------------------------------------
# 2. 流式 + temperature
# ------------------------------------------------------
@pytest.mark.parametrize("t", [0, 1.5, 2])
def test_stream_temperature(t):
    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url_1}},
                    {"type": "text", "text": "图中的文物属于哪个年代?"},
                ],
            }
        ],
        "stream": True,
        "temperature": t,
        "max_completion_tokens": 4,
    }
    _, text, _, _ = send_request_stream(URL, data)
    assert len(text.strip()) > 0


# ------------------------------------------------------
# 3. 流式 + top_p
# ------------------------------------------------------
@pytest.mark.parametrize("p", [0.1, 0.5, 1, 1.5])
def test_stream_top_p(p):
    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url_1}},
                    {"type": "text", "text": "图中的文物属于哪个年代?"},
                ],
            }
        ],
        "stream": True,
        "top_p": p,
        "max_completion_tokens": 4,
    }
    print(json.dumps(data, ensure_ascii=False))
    if p <= 1:
        chunks, text, _, _ = send_request_stream(URL, data)
        assert len(text.strip()) > 0
        print(json.dumps(chunks, ensure_ascii=False))
        assert len(text.strip()) > 0
    else:
        _, _, _, j = send_request_stream(URL, data)
        err = json.loads(j).get("error")
        assert err.get("param") == "top_p", f"param 字段错误: {err.get('param')}"
        assert err.get("message") == "Input should be less than or equal to 1", "错误提示不符合预期"


# ------------------------------------------------------
# 4. 流式 + n>1
# ------------------------------------------------------
def test_stream_n2():
    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url_1}},
                    {"type": "text", "text": "图中的文物属于哪个年代?"},
                ],
            }
        ],
        "stream": True,
        "n": 2,
        "max_completion_tokens": 6,
    }
    chunks, text, usage, _ = send_request_stream(URL, data)

    # n=2：至少出现过两个 index
    indexes = {c["choices"][0]["index"] for c in chunks}
    # print(json.dumps(chunks, ensure_ascii=False))
    # print(text)
    assert indexes == {0, 1}
    assert text[0] is not None
    assert text[1] is not None


# ------------------------------------------------------
# 5. 流式 + stop 序列
# ------------------------------------------------------
def test_stream_stop():
    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url_1}},
                    {"type": "text", "text": "图中的文物属于哪个年代?"},
                ],
            }
        ],
        "stream": True,
        "stop": ["的"],
    }
    _, text, _, _ = send_request_stream(URL, data)
    print(text)
    # 碰到 "的" 应提前结束
    assert text.endswith("的")


# ------------------------------------------------------
# 6. 流式 + max_completion_tokens 严格截断
# ------------------------------------------------------
def test_stream_max_completion_tokens():
    max_t = 5
    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url_1}},
                    {"type": "text", "text": "图中的文物属于哪个年代?"},
                ],
            }
        ],
        "stream": True,
        "max_completion_tokens": max_t,
        "return_token_ids": True,
    }
    chunks, _, usage, _ = send_request_stream(URL, data)

    # 1. 真实生成的 token 数
    real_tokens = 0
    for c in chunks:
        delta = c["choices"][0]["delta"]
        token_ids = delta.get("completion_token_ids")

        if token_ids:
            real_tokens += len(token_ids)
    assert real_tokens == max_t, f"expect {max_t} tokens, got {real_tokens}"

    # 2. 最后一包 finish_reason
    assert chunks[-1]["choices"][0]["finish_reason"] == "length"

    # 3. 如果后端返回了 usage，也一起检查
    if usage and "completion_tokens" in usage:
        assert usage["completion_tokens"] == max_t


# ------------------------------------------------------
# 7. 流式 + penalties
# ------------------------------------------------------


@pytest.mark.parametrize("pp,fp", [(0, 0), (0.8, 0), (0, 0.8), (0.8, 0.8), (1, -1), (2, -2)])
def test_stream_penalties(pp, fp):
    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url_1}},
                    {"type": "text", "text": "描述这个文物"},
                ],
            }
        ],
        "stream": True,
        "presence_penalty": pp,
        "frequency_penalty": fp,
        "max_completion_tokens": 50,
        # 明确开启采样，penalty 才有意义
        "temperature": 0.8,
        "top_p": 0.9,
    }
    _, text, _, _ = send_request_stream(URL, data)

    # ===== 协议 / 基础行为断言 =====
    assert text.strip() != ""

    # ===== 行为级指标：重复度 =====
    chars = [c for c in text if not c.isspace()]
    if not chars:
        pytest.skip("输出过短，无法评估重复度")

    counter = Counter(chars)
    max_freq = counter.most_common(1)[0][1]
    repeat_ratio = max_freq / len(chars)

    # ===== penalty 生效的“趋势断言” =====
    # 不和具体文本比，只限制“不能极端重复”
    if pp > 0 or fp > 0:
        assert repeat_ratio < 0.5, f"penalty={pp, fp} 下重复度仍然过高: {repeat_ratio}, content={text!r}"


# ------------------------------------------------------
# 8. 流式 + logprobs
# ------------------------------------------------------
def test_stream_logprobs():
    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url_1}},
                    {"type": "text", "text": "图中的文物属于哪个年代?"},
                ],
            }
        ],
        "stream": True,
        "logprobs": True,
        "top_logprobs": 3,
        "max_completion_tokens": 6,
    }
    chunks, _, _, _ = send_request_stream(URL, data)
    # 每个非空 delta 都应带 logprobs
    for chk in chunks:
        delta = chk["choices"][0]["delta"]
        if delta.get("content"):
            assert chk["choices"][0]["logprobs"]["content"]
            for tok in chk["choices"][0]["logprobs"]["content"]:
                assert "token" in tok
                assert isinstance(tok["logprob"], (int, float))
                assert len(tok["top_logprobs"]) == 3


# ------------------------------------------------------
# 9. 流式 + user / metadata / store /min_tokens
# ------------------------------------------------------
def test_stream_meta():
    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url_1}},
                    {"type": "text", "text": "图中的文物好看吗?"},
                ],
            }
        ],
        "stream": True,
        "user": "stream-tester",
        "min_tokens": 500,
        "store": True,
        "max_completion_tokens": 600,
        "stream_options": {"include_usage": True},
    }
    chunks, text, usage, _ = send_request_stream(URL, data)
    print(chunks)
    assert usage["completion_tokens"] > data["min_tokens"]
    assert data["user"] in chunks[0]["id"]


# ------------------------------------------------------
# 10. 流式 + return_token_ids
# ------------------------------------------------------
def test_stream_token_ids():
    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url_1}},
                    {"type": "text", "text": "图中的文物属于哪个年代?"},
                ],
            }
        ],
        "stream": True,
        "return_token_ids": True,
        "max_completion_tokens": 4,
    }
    chunks, _, _, _ = send_request_stream(URL, data)
    print(json.dumps(chunks, ensure_ascii=False))
    # 首包应出现 prompt_token_ids
    first = chunks[0]
    assert isinstance(first["choices"][0]["delta"]["prompt_token_ids"], list)
    # 剩余包都包含 completion_token_ids
    for chunk in chunks[1:]:
        assert isinstance(chunk["choices"][0]["delta"]["completion_token_ids"], list)
        assert len(chunk["choices"][0]["delta"]["completion_token_ids"]) >= 1


# ------------------------------------------------------
# 11. 流式超长内容
# ------------------------------------------------------
def test_stream_long():
    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url_1}},
                    {"type": "text", "text": "图中的文物属于哪个年代?" * 500},
                ],
            }
        ],
        "stream": True,
        "max_completion_tokens": 8,
    }
    _, text, _, _ = send_request_stream(URL, data)
    assert len(text.strip()) > 0


# ------------------------------------------------------
# 12. 流式特殊字符
# ------------------------------------------------------
def test_stream_special():
    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url_1}},
                    {"type": "text", "text": "图中的文物属于哪个年代?😀\n\t\r"},
                ],
            }
        ],
        "stream": True,
        "max_completion_tokens": 6,
    }
    _, text, _, _ = send_request_stream(URL, data)
    assert text.strip()  # 正常回复即可


# ------------------------------------------------------
# 13. 流式 + service_tier
# ------------------------------------------------------
@pytest.mark.parametrize("tier", ["auto", "default", "flex"])
def test_stream_service_tier(tier):
    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url_1}},
                    {"type": "text", "text": "图中的文物属于哪个年代?"},
                ],
            }
        ],
        "stream": True,
        "service_tier": tier,
        "max_completion_tokens": 4,
    }
    _, text, _, _ = send_request_stream(URL, data)
    assert len(text.strip()) > 0


# ------------------------------------------------------
# 14. 流式大组合
# ------------------------------------------------------
def test_stream_big_combo():
    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url_1}},
                    {"type": "text", "text": "图中的文物属于哪个年代?"},
                ],
            }
        ],
        "stream": True,
        "temperature": 0.7,
        "top_p": 0.95,
        "n": 1,
        "max_completion_tokens": 10,
        "stop": ["bye"],
        "presence_penalty": 1,
        "frequency_penalty": 1,
        "logprobs": True,
        "top_logprobs": 5,
        "return_token_ids": True,
        "user": "combo",
        "service_tier": "flex",
        "metadata": {"k": "v"},
        "store": True,
    }
    chunks, text, usage, _ = send_request_stream(URL, data)
    # print(json.dumps(chunks, ensure_ascii=False))
    # 真实生成的 token 数
    real_tokens = 0
    for c in chunks:
        delta = c["choices"][0]["delta"]
        token_ids = delta.get("completion_token_ids")
        if token_ids:
            real_tokens += len(token_ids)
    assert len(text.strip()) > 0
    # 生成的token数符合预期
    assert real_tokens <= data["max_completion_tokens"]
    # 首包 token_ids
    assert "role" in chunks[0]["choices"][0]["delta"] and chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert "prompt_token_ids" in chunks[0]["choices"][0]["delta"] and isinstance(
        chunks[0]["choices"][0]["delta"]["prompt_token_ids"], list
    )
    # 每步 logprobs
    for chk in chunks[1:]:
        if chk["choices"][0]["delta"].get("content"):
            assert chk["choices"][0]["logprobs"]["content"]
            for tok in chk["choices"][0]["logprobs"]["content"]:
                assert "token" in tok
                assert isinstance(tok["logprob"], (int, float))
                assert len(tok["top_logprobs"]) in [data["top_logprobs"], data["top_logprobs"] + 1]
    # 只要 stop 列表非空，就校验最终文本里确实没有出现任何一个 stop 串
    if data["stop"]:
        for s in data["stop"]:
            assert s not in text, f"stop sequence '{s}' should not appear in output"
    # 末包 finish_reason 合法值
    finish = chunks[-1]["choices"][0]["finish_reason"]
    assert finish in {"stop", "length"}, f"unexpected finish_reason: {finish}"


# ===================== 补充：异常参数校验 =====================
@pytest.mark.parametrize(
    "invalid_payload",
    [
        {"top_p": -1},  # 负值非法
        {"presence_penalty": 3.0},  # 超过上限 2.0
    ],
)
def test_invalid_parameter_values(invalid_payload):
    """验证非法参数值应返回 400 或对应的错误结构"""
    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url_1}},
                    {"type": "text", "text": "图中的文物属于哪个年代?"},
                ],
            }
        ],
        **invalid_payload,
    }
    r = send_request(URL, data)
    print(r.json())
    print(data)
    # 服务能力测试：非法输入不应返回 200
    assert r.status_code in [400, 422], f"Expected error for {invalid_payload}, but got {r.status_code}"


# ===================== 补充：Response Format (JSON Mode) =====================
def test_response_format_json():
    """验证 json_object 模式（通用协议支持）"""
    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url_1}},
                    {"type": "text", "text": "图中的文物属于哪个年代?"},
                ],
            }
        ],
        "response_format": {"type": "json_object"},
    }
    j = send_request(URL, data)
    # 只要返回 200，说明服务能力（Gateway/Wrapper）是通的
    assert j.status_code == 200


# ===================== 补充：Seed 一致性初步校验 =====================
def test_seed_consistency():
    """校验 seed 参数是否存在且不崩溃 (由于模型确定性难保证，仅校验接口接受该参数)"""
    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url_1}},
                    {"type": "text", "text": "图中的文物属于哪个年代?"},
                ],
            }
        ],
        "seed": 12345,
        "temperature": 0,
    }
    r1 = send_request(URL, data).json()
    r2 = send_request(URL, data).json()

    assert "id" in r1 and "id" in r2
    # 如果服务支持 system_fingerprint，建议加上校验
    # if "system_fingerprint" in r1:
    #     assert r1["system_fingerprint"] == r2["system_fingerprint"]


# ===================== 补充：Tool Calls (插件/工具调用) 协议骨架 =====================
def test_tool_call_schema():
    """
    即使后端模型不支持实际调用，也要测试接口是否能正确解析 tools 定义
    这是目前 OpenAI 协议中最复杂的部分
    """
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {"location": {"type": "string"}}},
            },
        }
    ]
    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url_1}},
                    {"type": "text", "text": "图中的文物属于哪个年代?"},
                ],
            }
        ],
        "tools": tools,
        "tool_choice": "auto",
    }
    r = send_request(URL, data)
    assert r.status_code == 200
    j = r.json()
    # 校验返回结构中包含 message，且可能包含 tool_calls 字段（视模型能力而定）
    assert "choices" in j


# ===================== 补充：流式停止位深度校验 =====================
def test_stream_cancel_simulation():
    """
    模拟客户端中途断开连接。
    虽然 requests 很难直接模拟 socket 断开，但可以验证读取前几包后停止迭代。
    该用例主要用于手动观察后端日志是否会正常 stop 推理。
    """
    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url_1}},
                    {"type": "text", "text": "图中的文物属于哪个年代?"},
                ],
            }
        ],
        "stream": True,
    }
    resp = send_request(URL, data, stream=True)
    count = 0
    for line in resp.iter_lines():
        if count > 5:  # 只读 5 包就手动关掉连接
            resp.close()
            break
        count += 1
    assert count > 0


# ===================== 用法示例：pytest -q =====================
if __name__ == "__main__":
    pytest.main([__file__, "-q"])
