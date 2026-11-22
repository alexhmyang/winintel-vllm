# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 WinIntel contributors.
# Author: 数萌AI服务器 (winintel)
# Email: service@winintel.com
# Website: https://winintel.com
"""Check content of the four fixed prompts in evaluate_sm70_model.py."""

import argparse
import ast
import json
from pathlib import Path


def check_function(text: str) -> dict:
    """Execute only a plain two-argument comparison function."""
    try:
        source = text.split("```python\n", 1)[1].split("```", 1)[0]
        tree = ast.parse(source)
        allowed = (
            ast.Module,
            ast.FunctionDef,
            ast.arguments,
            ast.arg,
            ast.If,
            ast.Return,
            ast.Compare,
            ast.Gt,
            ast.Lt,
            ast.GtE,
            ast.LtE,
            ast.Name,
            ast.Load,
            ast.Expr,
            ast.Constant,
            ast.Call,
        )
        if any(not isinstance(node, allowed) for node in ast.walk(tree)):
            raise ValueError("Function contains unsupported syntax")
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
            raise ValueError("Expected one function")
        function = tree.body[0]
        parameters = function.args
        if (
            len(parameters.args) != 2
            or parameters.posonlyargs
            or parameters.kwonlyargs
            or parameters.vararg
            or parameters.kwarg
            or parameters.defaults
            or function.decorator_list
            or function.returns
            or any(argument.annotation for argument in parameters.args)
        ):
            raise ValueError("Expected two plain positional arguments")
        names = {argument.arg for argument in parameters.args}
        if any(
            isinstance(node, ast.Name) and node.id not in names | {"max"}
            for node in ast.walk(tree)
        ):
            raise ValueError("Function references names outside its arguments")
        if any(
            isinstance(node, ast.Call)
            and (
                not isinstance(node.func, ast.Name)
                or node.func.id != "max"
                or len(node.args) != 2
                or node.keywords
                or any(
                    not isinstance(argument, ast.Name) or argument.id not in names
                    for argument in node.args
                )
            )
            for node in ast.walk(tree)
        ):
            raise ValueError("Only max of the two arguments is supported")
        namespace = {"__builtins__": {"max": max}}
        exec(compile(tree, "<checked-model-function>", "exec"), namespace)
        values = [(5, 10), (10, 5), (5, 5), (-9, -2), (1.25, -3)]
        outcomes = [
            namespace[function.name](left, right) == max(left, right)
            for left, right in values
        ]
        return {"passed": all(outcomes), "inputs": values, "outcomes": outcomes}
    except (IndexError, SyntaxError, ValueError, TypeError, NameError) as exc:
        return {"passed": False, "error": str(exc)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    comparison = json.loads(args.comparison.read_text(encoding="utf-8"))
    rows = comparison["results"]
    if (
        len(rows) != 4
        or rows[0]["question"] != "What is 17 plus 25? Answer with only the number."
        or not rows[1]["question"].startswith("材料原文：合同约定货款总额")
        or rows[2]["question"]
        != "Write a Python function that returns the larger of two numbers."
        or not rows[3]["question"].startswith("Context: ")
        or not rows[3]["question"].endswith("What is the final number?")
    ):
        parser.error("Expected the evaluator's four fixed prompts")
    results = {}
    for engine in ("reference", "winintel"):
        text = [row[f"{engine}_text"] for row in rows]
        results[engine] = {
            "arithmetic_contains_correct_answer": "42" in text[0],
            "payment_date_and_source_phrase": "付款截止日为2026年10月1日" in text[1],
            "comparison_function": check_function(text[2]),
            "long_prompt_contains_correct_answer": "42" in text[3],
        }
    args.report.write_text(
        json.dumps(
            {
                "source": str(args.comparison),
                "scope": "Four fixed prompt content checks; not general model quality",
                "strict_token_parity_passed": all(row["equal_tokens"] for row in rows),
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(args.report.read_text(encoding="utf-8"))
    if any(
        not checks["arithmetic_contains_correct_answer"]
        or not checks["payment_date_and_source_phrase"]
        or not checks["comparison_function"]["passed"]
        or not checks["long_prompt_contains_correct_answer"]
        for checks in results.values()
    ):
        raise SystemExit("Fixed-prompt content checks failed")


if __name__ == "__main__":
    main()
