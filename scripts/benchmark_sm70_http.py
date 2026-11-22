# SPDX-License-Identifier: Apache-2.0
# Author: 数萌AI服务器 (winintel)
"""Measure independent streaming requests against an OpenAI compatible server."""

import argparse
import ast
import concurrent.futures
import json
import threading
import time
import urllib.request
from pathlib import Path

FIRST_TOKEN_TIMEOUT_SECONDS = 180

CODING_TASKS = [
    (
        "lru",
        (
            "Implement class LRUCache(capacity) with get(key) returning -1 for missing "
            "keys and put(key, value). Updating and getting refresh recency. Evict the "
            "least recently used item. Reject capacity <= 0 with ValueError. O(1) ops."
        ),
        """
c = LRUCache(2)
c.put("a", 1); c.put("b", 2)
assert c.get("a") == 1
c.put("c", 3)
assert c.get("b") == -1
c.put("a", 4); c.put("d", 5)
assert c.get("a") == 4 and c.get("c") == -1
try:
    LRUCache(0)
except ValueError:
    pass
else:
    raise AssertionError("invalid capacity accepted")
""",
    ),
    (
        "csv",
        (
            "Implement parse_records(text) for CSV with a header name,age,email. "
            "Return a list of dictionaries with age converted to int. Strip field "
            "whitespace. Skip empty rows. Support quoted commas and CRLF. Reject "
            "missing required headers, missing field values, non-integer ages, or "
            "negative ages with ValueError. Additional columns may be ignored."
        ),
        """
r = parse_records('name,age,email\\r\\n"Smith, Alice", 23 , a@x.com\\r\\n\\r\\n')
assert r == [{"name": "Smith, Alice", "age": 23, "email": "a@x.com"}]
assert parse_records("name,age,email\\n") == []
for s in ["name,age\\nA,2", "name,age,email\\nA,-1,a@x",
          "name,age,email\\nA,no,a@x", "name,age,email\\nA,2,"]:
    try:
        parse_records(s)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid CSV accepted")
""",
    ),
    (
        "dag",
        (
            "Implement topological_sort(graph) where graph maps nodes to lists of "
            "outgoing neighbors. Include nodes appearing only as neighbors. Return "
            "a list with every node once and sources before destinations. Detect "
            "cycles including self-loops and raise ValueError. Support an empty "
            "graph. Use O(V+E) time and avoid recursion depth limits."
        ),
        """
g = {"a": ["b", "c"], "b": ["d"], "c": ["d"], "z": []}
r = topological_sort(g)
assert len(r) == 5 and set(r) == {"a", "b", "c", "d", "z"}
assert all(r.index(u) < r.index(v) for u, vs in g.items() for v in vs)
assert topological_sort({}) == []
assert len(topological_sort({i: [i+1] for i in range(3000)})) == 3001
for g in [{"a": ["a"]}, {"a": ["b"], "b": ["a"]}]:
    try:
        topological_sort(g)
    except ValueError:
        pass
    else:
        raise AssertionError("cycle accepted")
""",
    ),
]


def coding_suite(args, tokenizer):
    import subprocess
    import sys

    tasks = [
        task
        for task in CODING_TASKS
        if args.coding_tasks is None or task[0] in args.coding_tasks
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": (
                        "Write Python 3 code using only the standard library. "
                        "Output only executable code, without Markdown or prose. "
                        "Include docstrings and a comprehensive unittest suite "
                        "of at least eight tests. " + task
                    ),
                }
            ],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for _, task, _ in tasks
    ]
    warmup_prompt = prompts[0]
    if args.coding_context_tokens:
        marker = "WININTEL_CODING_CONTEXT"
        frame = tokenizer.apply_chat_template(
            [{"role": "user", "content": marker}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prefix = tokenizer.encode(frame.split(marker)[0], add_special_tokens=False)
        filler = tokenizer.encode(
            "# Archived reference code; the implementation task follows below.\n"
            "def archived_identity(value):\n    return value\n\n",
            add_special_tokens=False,
        )
        for index, prompt in enumerate(prompts):
            padding = args.coding_context_tokens - len(prompt)
            if padding < 0:
                raise ValueError("Coding context is shorter than the task prompt")
            background = (filler * (padding // len(filler) + 1))[:padding]
            prompts[index] = prompt[: len(prefix)] + background + prompt[len(prefix) :]
            assert len(prompts[index]) == args.coding_context_tokens
    request(args.base_url, args.model, warmup_prompt, 4)
    report = {
        "base_url": args.base_url,
        "model": args.model,
        "workload": "English Python implementation and unit tests",
        "measurement": "Client stream; exclude first chunk from decode speed",
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "thinking": False,
        "ignore_eos": False,
        "coding_tasks": [task[0] for task in tasks],
        "coding_context_tokens": args.coding_context_tokens,
        "first_token_timeout_seconds": FIRST_TOKEN_TIMEOUT_SECONDS,
        "results": [],
    }
    for concurrency in args.concurrency:
        for trial in range(args.rounds):
            groups = (
                [[i] for i in range(len(tasks))]
                if concurrency == 1
                else [[i % len(tasks) for i in range(concurrency)]]
            )
            for indices in groups:
                barrier = threading.Barrier(concurrency)
                with concurrent.futures.ThreadPoolExecutor(concurrency) as pool:
                    rows = list(
                        pool.map(
                            lambda i, barrier=barrier: request(
                                args.base_url,
                                args.model,
                                prompts[i],
                                args.max_tokens,
                                barrier,
                                ignore_eos=False,
                            ),
                            indices,
                        )
                    )
                for session, (i, row) in enumerate(zip(indices, rows)):
                    name, _, checks = tasks[i]
                    code = row["output"].strip()
                    if code.startswith("```"):
                        code = code.split("\n", 1)[1].rsplit("```", 1)[0]
                    row["task"] = name
                    row["session"] = session + 1
                    try:
                        ast.parse(code)
                        # Import without invoking the generated unittest main.
                        harness = (
                            "import types, unittest\n"
                            "module=types.ModuleType('generated')\n"
                            "scope=module.__dict__\n"
                            f"exec(compile({code!r}, 'generated.py', 'exec'), scope)\n"
                            f"exec({checks!r}, scope)\n"
                            "suite=unittest.defaultTestLoader."
                            "loadTestsFromModule(module)\n"
                            "assert suite.countTestCases() >= 8\n"
                            "result=unittest.TextTestRunner().run(suite)\n"
                            "assert result.wasSuccessful()\n"
                        )
                        completed = subprocess.run(
                            [sys.executable, "-I", "-c", harness],
                            capture_output=True,
                            text=True,
                            timeout=15,
                        )
                        row["behavior_passed"] = completed.returncode == 0
                        row["validation_error"] = completed.stderr[-2000:]
                    except (SyntaxError, subprocess.TimeoutExpired) as exc:
                        row["behavior_passed"] = False
                        row["validation_error"] = str(exc)
                elapsed = max(r["finished"] for r in rows) - min(
                    r["started"] for r in rows
                )
                decode_elapsed = max(r["last_token"] for r in rows) - min(
                    r["first_token"] for r in rows
                )
                result = {
                    "concurrency": concurrency,
                    "round": trial + 1,
                    "aggregate_output_tokens_per_second": (
                        sum(r["usage"]["completion_tokens"] for r in rows) / elapsed
                    ),
                    "aggregate_decode_tokens_per_second": (
                        sum(r["decode_tokens"] for r in rows) / decode_elapsed
                    ),
                    "requests": rows,
                }
                report["results"].append(result)
                report["best_by_concurrency"] = {}
                for value in args.concurrency:
                    valid = [
                        group
                        for group in report["results"]
                        if group["concurrency"] == value
                        and all(row["behavior_passed"] for row in group["requests"])
                    ]
                    if valid:
                        best = max(
                            valid,
                            key=lambda group: group[
                                "aggregate_decode_tokens_per_second"
                            ],
                        )
                        report["best_by_concurrency"][str(value)] = {
                            "best_round": best["round"],
                            "best_decode_tokens_per_second": best[
                                "aggregate_decode_tokens_per_second"
                            ],
                            "mean_decode_tokens_per_second": sum(
                                group["aggregate_decode_tokens_per_second"]
                                for group in valid
                            )
                            / len(valid),
                            "passed_rounds": len(valid),
                        }
                args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False))
                print(
                    concurrency,
                    trial + 1,
                    [
                        (
                            r["task"],
                            round(r["decode_tokens_per_second"], 2),
                            r["usage"]["completion_tokens"],
                            r["behavior_passed"],
                        )
                        for r in rows
                    ],
                    "total",
                    round(result["aggregate_decode_tokens_per_second"], 2),
                    flush=True,
                )
    if not all(
        row["behavior_passed"]
        for group in report["results"]
        for row in group["requests"]
    ):
        raise SystemExit("Generated code or its unit tests failed")


def context_suite(args, tokenizer):
    """Check distant exact quotations in an explicitly counted long context."""
    clauses = [
        "合同编号A731：甲方应于2027年3月19日前支付尾款人民币137万元。",
        "合同编号B842：逾期付款的，每日按未付款金额的万分之七支付违约金。",
        "合同编号C953：因不可抗力导致迟延交付的，乙方不承担迟延交付违约责任。",
    ]
    marker = "WININTEL_CONTEXT_BODY"
    frame = tokenizer.apply_chat_template(
        [{"role": "user", "content": marker}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prefix, suffix = frame.split(marker)
    prefix = tokenizer.encode(prefix, add_special_tokens=False)
    suffix = tokenizer.encode(suffix, add_special_tokens=False)
    filler = tokenizer.encode(
        "背景记录：旧合同已终止。往年付款日在10月，金额为50万元，"
        "旧违约金为千分之五；这些资料不适用于当前合同。\n",
        add_special_tokens=False,
    )
    report = {
        "base_url": args.base_url,
        "model": args.model,
        "workload": "Long-context exact retrieval at 5%, 50%, and 95%",
        "context_limit_includes_output": True,
        "results": [],
    }
    for target_tokens in args.context_lengths:
        for concurrency in args.concurrency:
            prompts = []
            for session in range(concurrency):
                instruction = tokenizer.encode(
                    f"\n会话编号{session + 1}。请依次逐字引用当前合同A731、"
                    "B842、C953的三个完整原文。每行必须从“合同编号”开始，"
                    "包含编号、冒号和后面的完整句子。每条一行，"
                    "不加说明或引号。",
                    add_special_tokens=False,
                )
                body_size = target_tokens - len(prefix) - len(suffix) - len(instruction)
                body = (filler * (body_size // len(filler) + 1))[:body_size]
                for fraction, clause in zip((0.05, 0.5, 0.95), clauses):
                    needle = tokenizer.encode(
                        "\n当前合同原文：" + clause + "\n", add_special_tokens=False
                    )
                    start = int(body_size * fraction)
                    body[start : start + len(needle)] = needle
                prompt = prefix + body + instruction + suffix
                assert len(prompt) == target_tokens
                prompts.append(prompt)
            for trial in range(args.rounds):
                barrier = threading.Barrier(concurrency)
                with concurrent.futures.ThreadPoolExecutor(concurrency) as pool:
                    rows = list(
                        pool.map(
                            lambda prompt, barrier=barrier: request(
                                args.base_url,
                                args.model,
                                prompt,
                                args.max_tokens,
                                barrier,
                                ignore_eos=False,
                            ),
                            prompts,
                        )
                    )
                for session, row in enumerate(rows):
                    row["session"] = session + 1
                    row["exact_quote_passed"] = row["output"].strip() == "\n".join(
                        clauses
                    )
                report["results"].append(
                    {
                        "prompt_tokens": target_tokens,
                        "concurrency": concurrency,
                        "round": trial + 1,
                        "requests": rows,
                        "all_passed": all(row["exact_quote_passed"] for row in rows),
                    }
                )
                args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False))
                print(
                    target_tokens,
                    concurrency,
                    trial + 1,
                    [
                        (
                            row["session"],
                            round(row["ttft_seconds"], 2),
                            row["exact_quote_passed"],
                        )
                        for row in rows
                    ],
                    flush=True,
                )
    if not all(group["all_passed"] for group in report["results"]):
        raise SystemExit("Long-context quotations failed")


def request(base_url, model, prompt, max_tokens, barrier=None, ignore_eos=True):
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": ignore_eos,
        "stream": True,
        "stream_options": {"include_usage": True},
        "logprobs": 0,
    }
    if barrier is not None:
        barrier.wait(timeout=30)
    started = time.perf_counter()
    first = last = None
    first_count = total_count = 0
    text, tokens, usage = "", [], None
    http_request = urllib.request.Request(
        base_url.rstrip("/") + "/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(
        http_request, timeout=FIRST_TOKEN_TIMEOUT_SECONDS
    ) as response:
        stream_socket = response.fp.raw._sock
        while True:
            if first is None:
                remaining = FIRST_TOKEN_TIMEOUT_SECONDS - (
                    time.perf_counter() - started
                )
                if remaining <= 0:
                    raise TimeoutError("First token exceeded the waiting limit")
                stream_socket.settimeout(remaining)
            else:
                stream_socket.settimeout(180)
            line = response.readline()
            if not line:
                break
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                break
            event = json.loads(data)
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                chunk_tokens = (choice.get("logprobs") or {}).get("tokens", [])
                if not chunk_tokens:
                    continue
                now = time.perf_counter()
                if first is None:
                    first, first_count = now, len(chunk_tokens)
                last = now
                total_count += len(chunk_tokens)
                tokens.extend(chunk_tokens)
                text += choice.get("text", "")
    finished = time.perf_counter()
    if first is None or last is None or usage is None:
        raise RuntimeError("Server did not return token logprobs and final usage")
    if usage["completion_tokens"] != total_count or (
        ignore_eos and total_count != max_tokens
    ):
        raise RuntimeError(f"Incomplete token stream: {total_count}, {usage}")
    decode_seconds = last - first
    return {
        "prompt_token_ids": prompt,
        "started": started,
        "first_token": first,
        "last_token": last,
        "finished": finished,
        "ttft_seconds": first - started,
        "decode_seconds": decode_seconds,
        "decode_tokens": total_count - first_count,
        "decode_tokens_per_second": (
            (total_count - first_count) / decode_seconds if decode_seconds else None
        ),
        "prompt_tokens_per_ttft_second": usage["prompt_tokens"] / (first - started),
        "usage": usage,
        "output": text,
        "tokens": tokens,
    }


def main():
    global FIRST_TOKEN_TIMEOUT_SECONDS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8027/v1")
    parser.add_argument("--model", default="winintel-qwen27")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--long-prompt-tokens", type=int, default=2048)
    parser.add_argument(
        "--coding-context-tokens",
        type=int,
        default=0,
        help="Pad English code prompts to this exact input token count",
    )
    parser.add_argument(
        "--coding-tasks",
        nargs="+",
        choices=("lru", "csv", "dag"),
        help="Select code tasks; use lru for the historical fastest workload",
    )
    parser.add_argument("--first-token-timeout-seconds", type=float, default=180)
    parser.add_argument(
        "--workload",
        choices=("quotation", "english-code", "context-retrieval"),
        default="quotation",
    )
    parser.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        default=[8192, 65536, 131072, 262016],
    )
    parser.add_argument(
        "--quotation-suite",
        action="store_true",
        help="Also require six exact source extractions, including long distractors",
    )
    args = parser.parse_args()
    if any(value < 1 for value in args.concurrency):
        parser.error("--concurrency values must be positive")
    if args.coding_context_tokens < 0:
        parser.error("--coding-context-tokens must be nonnegative")
    if not 0 < args.first_token_timeout_seconds <= 180:
        parser.error("--first-token-timeout-seconds must be in (0, 180]")
    FIRST_TOKEN_TIMEOUT_SECONDS = args.first_token_timeout_seconds
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if args.workload == "english-code":
        coding_suite(args, tokenizer)
        return
    if args.workload == "context-retrieval":
        context_suite(args, tokenizer)
        return
    questions = [
        "请用中文回答：人工智能服务器为什么需要大显存？",
        "用中文解释张量并行是什么，给一个简单例子。",
        (
            "原文：甲方应于2026年12月1日前支付尾款人民币100万元。"
            "只输出上述原文，不得改变任何字或数字。"
        ),
    ]
    short = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": q}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for q in questions
    ]
    paragraph = tokenizer.encode(
        "资料：数萌AI服务器支持私有化部署，数据保存在企业内部。",
        add_special_tokens=False,
    )
    long = [
        (paragraph * (args.long_prompt_tokens // len(paragraph) + 1))[
            : args.long_prompt_tokens - len(p)
        ]
        + p
        for p in short
    ]
    request(args.base_url, args.model, short[0], 4)
    report = {
        "base_url": args.base_url,
        "model": args.model,
        "measurement": "Client streaming; TTFT includes prefill and transport",
        "results": [],
    }
    expected_quote = "甲方应于2026年12月1日前支付尾款人民币100万元。"
    quality = request(args.base_url, args.model, short[2], 128, ignore_eos=False)
    report["quality"] = {
        "scope": "One exact Chinese quotation, greedy decoding with EOS enabled",
        "expected": expected_quote,
        "exact_quote_passed": quality["output"].strip() == expected_quote,
        "request": quality,
    }
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    if args.quotation_suite:
        clauses = [
            ("付款截止日", "甲方应于2026年12月1日前支付尾款人民币100万元。"),
            ("违约金", "逾期付款的，每日按未付款金额的万分之三支付违约金。"),
            ("例外条件", "因不可抗力导致迟延交付的，乙方不承担迟延交付违约责任。"),
        ]
        checks = []
        for length in ("short", "long"):
            for target, expected in clauses:
                question = (
                    "材料原文：\n"
                    "双方于2026年11月19日签署本合同。\n"
                    f"{expected}\n"
                    "其他事项由双方另行书面约定。\n"
                    f"请逐字引用材料中关于{target}的完整句子。"
                    "只输出该句，不添加引号、说明或其他内容。"
                )
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": question}],
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                if length == "long":
                    # Background stays inside the user message, before the source.
                    background = (
                        "背景资料（不是本合同条款）：往年付款日在10月，"
                        "往年违约金为千分之五，往年交付安排不同。\n"
                    )
                    while True:
                        candidate = tokenizer.apply_chat_template(
                            [{"role": "user", "content": background + question}],
                            tokenize=True,
                            add_generation_prompt=True,
                            enable_thinking=False,
                        )
                        if len(candidate) >= args.long_prompt_tokens:
                            prompt = candidate
                            break
                        background += (
                            "背景记录：旧合同已终止，其日期、金额和责任条款"
                            "不适用于本合同。\n"
                        )
                result = request(
                    args.base_url, args.model, prompt, 128, ignore_eos=False
                )
                checks.append(
                    {
                        "prompt_length": length,
                        "target": target,
                        "expected": expected,
                        "exact_quote_passed": result["output"].strip() == expected,
                        "request": result,
                    }
                )
                report["quotation_suite"] = {
                    "scope": (
                        "Six synthetic exact source extractions; "
                        "not general legal-review acceptance"
                    ),
                    "all_passed": all(row["exact_quote_passed"] for row in checks),
                    "results": checks,
                }
                args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False))
                print(length, target, checks[-1]["exact_quote_passed"], flush=True)
    for length, prompts in (("short", short), ("long", long)):
        for concurrency in args.concurrency:
            for trial in range(args.rounds):
                barrier = threading.Barrier(concurrency)
                with concurrent.futures.ThreadPoolExecutor(concurrency) as pool:
                    futures = [
                        pool.submit(
                            request,
                            args.base_url,
                            args.model,
                            p,
                            args.max_tokens,
                            barrier,
                        )
                        for p in (prompts[i % len(prompts)] for i in range(concurrency))
                    ]
                    rows = [f.result() for f in futures]
                elapsed = max(r["finished"] for r in rows) - min(
                    r["started"] for r in rows
                )
                decode_elapsed = max(r["last_token"] for r in rows) - min(
                    r["first_token"] for r in rows
                )
                result = {
                    "prompt_length": length,
                    "concurrency": concurrency,
                    "round": trial + 1,
                    "total_seconds": elapsed,
                    "aggregate_output_tokens_per_second": (
                        sum(r["usage"]["completion_tokens"] for r in rows) / elapsed
                    ),
                    "aggregate_decode_tokens_per_second": (
                        sum(r["decode_tokens"] for r in rows) / decode_elapsed
                    ),
                    "requests": rows,
                }
                report["results"].append(result)
                args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False))
                print(
                    length,
                    concurrency,
                    trial + 1,
                    result["aggregate_decode_tokens_per_second"],
                    flush=True,
                )
    if args.quotation_suite and not (
        report["quality"]["exact_quote_passed"]
        and report["quotation_suite"]["all_passed"]
    ):
        raise SystemExit("Exact source quotation checks failed")


if __name__ == "__main__":
    main()
