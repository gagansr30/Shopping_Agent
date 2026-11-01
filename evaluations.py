"""Live evaluations for tool selection and shopping response quality."""

import json

from langchain_core.messages import AIMessage

from shopping_agent import llm, run_shopping_agent


TOOL_CALL_CASES = [
    {
        "prompt": "Find organic honey under $20.",
        "tool": "search_products",
        "expected_args": {"is_organic": True, "max_price": 20},
        "judge": True,
    },
    {
        "prompt": "Show me almonds costing no more than $12.",
        "tool": "search_products",
        "expected_args": {"max_price": 12},
        "judge": True,
    },
    {
        "prompt": "What have I ordered before?",
        "tool": "get_order_history",
        "expected_args": {},
        "judge": False,
    },
]


def _tool_calls(messages: list) -> list[dict]:
    calls = []
    for message in messages:
        if isinstance(message, AIMessage):
            calls.extend(message.tool_calls or [])
        elif isinstance(message, dict):
            calls.extend(message.get("tool_calls", []))
    return calls


def _judge_response(prompt: str, response: str, tool_results: list[str]) -> dict:
    request = {
        "user_request": prompt,
        "assistant_response": response,
        "authoritative_tool_results": tool_results,
    }
    judged = llm.invoke(
        [
            (
                "system",
                "You are an evaluator for a shopping assistant. Score each criterion from "
                "1 (poor) to 5 (excellent): relevance, correctness (product names, prices, "
                "ratings, and claims match the authoritative tool results), and "
                "format_compliance (matching product results use the required numbered-list "
                "format '#N. Name (ID:X) — $price ★rating — organic/non-organic'). "
                "Do not reward invented products. Return only a JSON object with numeric "
                "keys relevance, correctness, format_compliance, plus a short rationale.",
            ),
            ("human", json.dumps(request)),
        ]
    )
    parsed = json.loads(str(judged.content))
    for key in ("relevance", "correctness", "format_compliance"):
        if not isinstance(parsed.get(key), (int, float)) or not 1 <= parsed[key] <= 5:
            raise AssertionError(f"Judge returned an invalid {key} score: {parsed!r}")
    return parsed


def main() -> None:
    for index, case in enumerate(TOOL_CALL_CASES, start=1):
        result = run_shopping_agent(
            [{"role": "user", "content": case["prompt"]}],
            user_id=f"shopping-agent-eval-{index}",
        )
        calls = _tool_calls(result["messages"])
        matching = [call for call in calls if call.get("name") == case["tool"]]
        assert matching, (
            f"Case {index}: expected call to {case['tool']}; "
            f"observed {[call.get('name') for call in calls]}"
        )
        actual_args = matching[0].get("args", {})
        for name, expected in case["expected_args"].items():
            actual = actual_args.get(name)
            if isinstance(expected, (int, float)) and not isinstance(expected, bool):
                matches = actual is not None and float(actual) == float(expected)
            else:
                matches = actual == expected
            assert matches, (
                f"Case {index}: expected {name}={expected!r}, got {actual!r}"
            )

        print(f"PASS tool-call case {index}: {case['tool']} {actual_args}")
        if case["judge"]:
            response = str(result["messages"][-1].content)
            tool_results = [
                str(message.content)
                for message in result["messages"]
                if getattr(message, "type", "") == "tool"
            ]
            scores = _judge_response(case["prompt"], response, tool_results)
            print(f"  response scores: {json.dumps(scores)}")


if __name__ == "__main__":
    main()