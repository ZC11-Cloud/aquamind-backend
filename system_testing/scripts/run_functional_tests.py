import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from common import (
    ROOT_DIR,
    auth_header,
    build_results_path,
    load_metrics_config,
    login_and_get_token,
    register_test_user,
    safe_json,
    timed_request,
    write_json,
)


CASES_PATH = ROOT_DIR / "data" / "functional_test_cases.json"
LOW_COST_RAG_SENTINEL_CASE_IDS = {"F002"}


def _load_cases() -> list[dict[str, Any]]:
    with CASES_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _create_conversation(base_url: str, token: str, timeout_sec: int, title: str) -> int:
    resp, _ = timed_request(
        method="POST",
        url=f"{base_url}/qa/conversations",
        timeout=timeout_sec,
        headers=auth_header(token),
        json_payload={"title": title},
    )
    if resp.status_code != 201:
        raise RuntimeError(f"创建测试会话失败: {resp.status_code}, {resp.text}")
    payload = safe_json(resp)
    conv_id = payload.get("id")
    if not conv_id:
        raise RuntimeError(f"会话响应缺少 id: {payload}")
    return int(conv_id)


def _finalize_payload(case_payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if case_payload is None:
        return None
    payload = dict(case_payload)
    repeat_count = payload.pop("_repeat_content", None)
    if repeat_count and "content" in payload:
        payload["content"] = str(payload["content"]) * int(repeat_count)
    return payload


def _status_ok(case: dict[str, Any], status_code: int) -> bool:
    if "expected_status_any" in case:
        return status_code in case["expected_status_any"]
    return status_code == case.get("expected_status", 200)


def _is_rag_message_case(case: dict[str, Any]) -> bool:
    payload = case.get("payload")
    if not isinstance(payload, dict):
        return False
    return (
        case.get("method") == "POST"
        and str(case.get("endpoint", "")).endswith("/messages")
        and bool(payload.get("use_rag"))
    )


def _parse_case_ids(raw: str | None) -> set[str]:
    if not raw:
        return set()
    return {part.strip() for part in raw.split(",") if part.strip()}


def _select_cases(
    cases: list[dict[str, Any]],
    profile: str,
    include_ids: set[str],
    exclude_ids: set[str],
    skip_rag_cases: bool,
    max_rag_cases: int | None,
) -> list[dict[str, Any]]:
    selected = []
    for case in cases:
        case_id = str(case.get("id", "")).strip()
        if include_ids and case_id not in include_ids:
            continue
        if case_id in exclude_ids:
            continue
        selected.append(case)

    if profile == "low-cost" and not include_ids:
        compact_selected = []
        for case in selected:
            case_id = str(case.get("id", "")).strip()
            if not _is_rag_message_case(case) or case_id in LOW_COST_RAG_SENTINEL_CASE_IDS:
                compact_selected.append(case)
        selected = compact_selected

    if skip_rag_cases:
        selected = [case for case in selected if not _is_rag_message_case(case)]

    if max_rag_cases is not None and max_rag_cases >= 0:
        rag_seen = 0
        limited = []
        for case in selected:
            if _is_rag_message_case(case):
                if rag_seen >= max_rag_cases:
                    continue
                rag_seen += 1
            limited.append(case)
        selected = limited

    return selected


def _run_case(
    base_url: str,
    timeout_sec: int,
    conversation_id: int,
    token: str,
    case: dict[str, Any],
) -> dict[str, Any]:
    endpoint = case["endpoint"].replace("{{conversation_id}}", str(conversation_id))
    url = f"{base_url}{endpoint}"
    mode = case.get("auth_mode", "normal")
    headers = auth_header(token, auth_mode=mode)
    payload = _finalize_payload(case.get("payload"))
    raw_body = case.get("raw_body")

    try:
        resp, latency_ms = timed_request(
            method=case["method"],
            url=url,
            timeout=timeout_sec,
            headers=headers,
            json_payload=payload,
            raw_body=raw_body,
        )
        body = safe_json(resp)
        status_pass = _status_ok(case, resp.status_code)

        content_text = ""
        citations = []
        if isinstance(body, dict):
            content_text = str(body.get("content", "")).strip()
            citations = body.get("citations") or []
        keyword_pass = True
        keywords = case.get("expect_keywords_any", [])
        if keywords and isinstance(body, dict):
            target_text = str(body.get("content", ""))
            keyword_pass = any(k in target_text for k in keywords)

        non_empty_pass = True
        if case.get("expect_non_empty_answer"):
            non_empty_pass = bool(content_text)

        generation_score = 0
        if status_pass and non_empty_pass and keyword_pass:
            generation_score = 2
        elif status_pass and non_empty_pass:
            generation_score = 1

        retrieval_hit = False
        if payload and payload.get("use_rag") and status_pass:
            retrieval_hit = isinstance(citations, list) and len(citations) > 0

        passed = status_pass and keyword_pass and non_empty_pass
        fail_reasons = []
        if not status_pass:
            fail_reasons.append(f"状态码不符({resp.status_code})")
        if not keyword_pass:
            fail_reasons.append("关键字未命中")
        if not non_empty_pass:
            fail_reasons.append("答案为空")

        is_rag_case = bool(payload and payload.get("use_rag")) and endpoint.endswith("/messages")

        return {
            "id": case["id"],
            "name": case["name"],
            "category": case["category"],
            "method": case["method"],
            "endpoint": endpoint,
            "status_code": resp.status_code,
            "expected": case.get("expected_status_any", case.get("expected_status")),
            "latency_ms": round(latency_ms, 2),
            "passed": passed,
            "is_rag_case": is_rag_case,
            "retrieval_hit": retrieval_hit,
            "generation_score": generation_score,
            "fail_reasons": fail_reasons,
            "response_body": body,
        }
    except Exception as e:  # noqa: BLE001
        return {
            "id": case["id"],
            "name": case["name"],
            "category": case["category"],
            "method": case["method"],
            "endpoint": endpoint,
            "status_code": None,
            "expected": case.get("expected_status_any", case.get("expected_status")),
            "latency_ms": None,
            "passed": False,
            "is_rag_case": False,
            "retrieval_hit": False,
            "generation_score": 0,
            "fail_reasons": [f"请求异常: {str(e)}"],
            "response_body": None,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 RAG 系统功能测试")
    parser.add_argument("--base-url", type=str, default=None, help="API 基地址")
    parser.add_argument("--timeout", type=int, default=None, help="请求超时时间（秒）")
    parser.add_argument(
        "--profile",
        choices=["full", "low-cost"],
        default="full",
        help="测试配置档位：full 为全量，low-cost 为低成本档",
    )
    parser.add_argument(
        "--include-case-ids",
        type=str,
        default="",
        help="仅执行指定用例ID（逗号分隔），例如 F001,F002,F035",
    )
    parser.add_argument(
        "--exclude-case-ids",
        type=str,
        default="",
        help="排除指定用例ID（逗号分隔）",
    )
    parser.add_argument(
        "--skip-rag-cases",
        action="store_true",
        help="跳过所有 use_rag=True 的消息用例",
    )
    parser.add_argument(
        "--max-rag-cases",
        type=int,
        default=None,
        help="最多执行多少个 RAG 消息用例（按文件顺序）",
    )
    args = parser.parse_args()

    cfg = load_metrics_config()
    base_url = args.base_url or cfg["environment"]["base_url"]
    timeout_sec = args.timeout or cfg["functional"]["timeout_seconds"]
    account = cfg["test_account"]

    register_test_user(base_url, account, timeout=timeout_sec)
    token = login_and_get_token(base_url, account, timeout=timeout_sec)
    conversation_id = _create_conversation(
        base_url=base_url,
        token=token,
        timeout_sec=timeout_sec,
        title=cfg["functional"]["conversation_title"],
    )

    all_cases = _load_cases()
    include_ids = _parse_case_ids(args.include_case_ids)
    exclude_ids = _parse_case_ids(args.exclude_case_ids)
    cases = _select_cases(
        cases=all_cases,
        profile=args.profile,
        include_ids=include_ids,
        exclude_ids=exclude_ids,
        skip_rag_cases=args.skip_rag_cases,
        max_rag_cases=args.max_rag_cases,
    )
    if not cases:
        raise RuntimeError("筛选后无可执行测试用例，请检查参数配置。")

    rag_case_count = sum(1 for case in cases if _is_rag_message_case(case))
    print(
        f"[功能测试] profile={args.profile}, 用例数={len(cases)}, "
        f"RAG消息用例数={rag_case_count}"
    )
    results = [
        _run_case(
            base_url=base_url,
            timeout_sec=timeout_sec,
            conversation_id=conversation_id,
            token=token,
            case=case,
        )
        for case in cases
    ]

    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    pass_rate = passed / total if total else 0.0
    latency_values = [r["latency_ms"] for r in results if r["latency_ms"] is not None]
    avg_latency_ms = statistics.mean(latency_values) if latency_values else 0
    p95_latency_ms = 0
    if len(latency_values) >= 2:
        sorted_vals = sorted(latency_values)
        idx = min(len(sorted_vals) - 1, int(len(sorted_vals) * 0.95))
        p95_latency_ms = sorted_vals[idx]

    rag_cases = [r for r in results if r.get("is_rag_case")]
    retrieval_hits = sum(1 for r in rag_cases if r.get("retrieval_hit"))
    retrieval_hit_rate = (retrieval_hits / len(rag_cases)) if rag_cases else 0.0
    generation_score_sum = sum(int(r.get("generation_score", 0)) for r in rag_cases)
    generation_accuracy = (generation_score_sum / (2 * len(rag_cases))) if rag_cases else 0.0

    summary = {
        "base_url": base_url,
        "conversation_id": conversation_id,
        "total_cases": total,
        "passed_cases": passed,
        "pass_rate": round(pass_rate, 4),
        "avg_latency_ms": round(avg_latency_ms, 2),
        "p95_latency_ms": round(p95_latency_ms, 2),
        "rag_case_count": len(rag_cases),
        "retrieval_hit_rate": round(retrieval_hit_rate, 4),
        "generation_accuracy": round(generation_accuracy, 4),
        "thresholds": cfg["acceptance"],
    }

    output = {"summary": summary, "cases": results}
    out_path = build_results_path("functional", "functional_report")
    write_json(out_path, output)

    print(f"[功能测试完成] 输出文件: {out_path}")
    print(f"通过率: {passed}/{total} = {pass_rate:.2%}")
    print(f"平均响应: {avg_latency_ms:.2f} ms, P95: {p95_latency_ms:.2f} ms")
    print(f"检索命中率: {retrieval_hit_rate:.2%}, 生成正确率: {generation_accuracy:.2%}")


if __name__ == "__main__":
    main()
