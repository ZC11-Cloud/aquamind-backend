import argparse
import random
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import psutil
import requests

from common import (
    auth_header,
    build_results_path,
    load_metrics_config,
    login_and_get_token,
    register_test_user,
    safe_json,
    timed_request,
    write_json,
)


class SharedStats:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.success = 0
        self.failure = 0
        self.errors: list[str] = []
        self.latency_ms: list[float] = []
        self.samples: list[dict[str, Any]] = []

    def add_success(self, latency_ms: float) -> None:
        with self.lock:
            self.success += 1
            self.latency_ms.append(latency_ms)

    def add_failure(self, error_msg: str) -> None:
        with self.lock:
            self.failure += 1
            self.errors.append(error_msg)

    def add_sample(self, payload: dict[str, Any]) -> None:
        with self.lock:
            self.samples.append(payload)


def create_conversation(base_url: str, token: str, timeout_sec: int, idx: int) -> int:
    resp, _ = timed_request(
        method="POST",
        url=f"{base_url}/qa/conversations",
        timeout=timeout_sec,
        headers=auth_header(token),
        json_payload={"title": f"稳定性测试会话_{idx}"},
    )
    if resp.status_code != 201:
        raise RuntimeError(f"创建会话失败: {resp.status_code}, {resp.text}")
    payload = safe_json(resp)
    conv_id = payload.get("id")
    if not conv_id:
        raise RuntimeError("创建会话响应缺少 id")
    return int(conv_id)


def worker_loop(
    base_url: str,
    token: str,
    conversation_id: int,
    timeout_sec: int,
    stop_time: datetime,
    stats: SharedStats,
    rag_ratio: float,
    request_interval_sec: float,
) -> None:
    headers = auth_header(token)
    while datetime.now() < stop_time:
        try:
            use_rag = random.random() < rag_ratio
            resp, latency_ms = timed_request(
                method="POST",
                url=f"{base_url}/qa/conversations/{conversation_id}/messages",
                timeout=timeout_sec,
                headers=headers,
                json_payload={
                    "content": (
                        "请解释RAG检索增强的作用"
                        if use_rag
                        else "请简述RAG系统为什么需要引用来源"
                    ),
                    "use_rag": use_rag,
                },
            )
            if resp.status_code == 201:
                stats.add_success(latency_ms)
            else:
                stats.add_failure(f"status={resp.status_code}")
        except requests.RequestException as e:
            stats.add_failure(f"request_exception={str(e)}")
        except Exception as e:  # noqa: BLE001
            stats.add_failure(f"unknown_exception={str(e)}")
        time.sleep(request_interval_sec)


def sample_system(stats: SharedStats) -> None:
    stats.add_sample(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "cpu_percent": psutil.cpu_percent(interval=None),
            "memory_percent": psutil.virtual_memory().percent,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 RAG 稳定性测试")
    parser.add_argument("--base-url", type=str, default=None, help="API 基地址")
    parser.add_argument("--duration-hours", type=float, default=None, help="持续时长（小时）")
    parser.add_argument("--concurrency", type=int, default=None, help="并发工作线程数")
    parser.add_argument("--timeout", type=int, default=60, help="单请求超时（秒）")
    parser.add_argument("--sample-interval-min", type=int, default=None, help="采样间隔（分钟）")
    parser.add_argument(
        "--rag-ratio",
        type=float,
        default=1.0,
        help="消息中 use_rag=True 的比例，范围 0~1",
    )
    parser.add_argument(
        "--request-interval-sec",
        type=float,
        default=0.2,
        help="每个线程两次请求之间的间隔秒数",
    )
    parser.add_argument(
        "--low-cost",
        action="store_true",
        help="低成本模式（等价于 rag_ratio=0.1 且 request_interval_sec=1.0）",
    )
    args = parser.parse_args()

    cfg = load_metrics_config()
    base_url = args.base_url or cfg["environment"]["base_url"]
    duration_hours = args.duration_hours or cfg["stability"]["duration_hours"]
    concurrency = args.concurrency or cfg["stability"]["concurrency"]
    timeout_sec = args.timeout
    sample_interval_min = args.sample_interval_min or cfg["stability"]["sample_interval_minutes"]
    rag_ratio = args.rag_ratio
    request_interval_sec = args.request_interval_sec
    if args.low_cost:
        rag_ratio = 0.1
        request_interval_sec = 1.0

    if not 0 <= rag_ratio <= 1:
        raise ValueError("rag_ratio 必须在 0 到 1 之间")
    if request_interval_sec < 0:
        raise ValueError("request_interval_sec 不能小于 0")
    account = cfg["test_account"]

    register_test_user(base_url, account, timeout=timeout_sec)
    token = login_and_get_token(base_url, account, timeout=timeout_sec)
    conv_ids = [
        create_conversation(base_url, token, timeout_sec, idx)
        for idx in range(concurrency)
    ]

    stats = SharedStats()
    end_time = datetime.now() + timedelta(hours=float(duration_hours))
    next_sample_time = datetime.now()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                worker_loop,
                base_url,
                token,
                conv_ids[idx],
                timeout_sec,
                end_time,
                stats,
                rag_ratio,
                request_interval_sec,
            )
            for idx in range(concurrency)
        ]

        while datetime.now() < end_time:
            if datetime.now() >= next_sample_time:
                sample_system(stats)
                next_sample_time = datetime.now() + timedelta(minutes=sample_interval_min)
            time.sleep(1)

        for f in futures:
            f.result()

    total_requests = stats.success + stats.failure
    error_rate = (stats.failure / total_requests) if total_requests else 0.0
    avg_latency_ms = statistics.mean(stats.latency_ms) if stats.latency_ms else 0.0

    p95_latency_ms = 0.0
    if len(stats.latency_ms) >= 2:
        sorted_vals = sorted(stats.latency_ms)
        idx = min(len(sorted_vals) - 1, int(len(sorted_vals) * 0.95))
        p95_latency_ms = sorted_vals[idx]

    summary = {
        "base_url": base_url,
        "duration_hours": duration_hours,
        "concurrency": concurrency,
        "rag_ratio": rag_ratio,
        "request_interval_sec": request_interval_sec,
        "total_requests": total_requests,
        "success_requests": stats.success,
        "failed_requests": stats.failure,
        "error_rate": round(error_rate, 4),
        "avg_latency_ms": round(avg_latency_ms, 2),
        "p95_latency_ms": round(p95_latency_ms, 2),
        "crash_count": 0,
        "sample_count": len(stats.samples),
        "errors_preview": stats.errors[:20],
    }

    out_path = build_results_path("stability", "stability_report")
    write_json(out_path, {"summary": summary, "samples": stats.samples})

    print(f"[稳定性测试完成] 输出文件: {out_path}")
    print(
        f"总请求={total_requests}, 错误率={error_rate:.2%}, "
        f"平均响应={avg_latency_ms:.2f}ms, P95={p95_latency_ms:.2f}ms"
    )


if __name__ == "__main__":
    main()
