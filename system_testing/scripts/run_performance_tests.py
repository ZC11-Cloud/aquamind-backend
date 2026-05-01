import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from shutil import which

from common import ROOT_DIR, ensure_dir, load_metrics_config


def _run_once(
    locust_file: Path,
    users: int,
    spawn_rate: int,
    duration_minutes: int,
    base_url: str,
    output_prefix: Path,
    chat_mode: str,
) -> int:
    locust_cli = which("locust")
    if locust_cli:
        cmd = [locust_cli]
    else:
        # 回退到模块方式，兼容仅安装了 Python 包但未写入 locust 可执行文件的环境
        cmd = [sys.executable, "-m", "locust"]

    cmd.extend(
        [
        "-f",
        str(locust_file),
        "--headless",
        "-u",
        str(users),
        "-r",
        str(spawn_rate),
        "-t",
        f"{duration_minutes}m",
        "--host",
        base_url,
        "--csv",
        str(output_prefix),
        ]
    )
    print("执行命令:", " ".join(cmd))
    env = {**os.environ, "AQUAMIND_PERF_CHAT_MODE": chat_mode}
    try:
        completed = subprocess.run(cmd, check=False, env=env)
        return int(completed.returncode)
    except FileNotFoundError as e:
        raise RuntimeError(
            "未找到 Locust 可执行程序。请先在当前环境安装：pip install locust"
        ) from e


def _read_aggregated_stats(stats_file: Path) -> dict:
    if not stats_file.exists():
        return {}
    rows = []
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            with stats_file.open("r", encoding=enc, errors="ignore") as f:
                rows = list(csv.DictReader(f))
            break
        except UnicodeDecodeError:
            rows = []
            continue
    if not rows:
        return {}
    row = next(
        (
            item
            for item in rows
            if item.get("Type") == "Aggregated" or item.get("Name") == "Aggregated"
        ),
        None,
    )
    if not row:
        return {}
    request_count = int(float(row.get("Request Count") or 0))
    failure_count = int(float(row.get("Failure Count") or 0))
    failure_rate = (failure_count / request_count) if request_count else 0.0
    return {
        "request_count": request_count,
        "failure_count": failure_count,
        "failure_rate": round(failure_rate, 4),
        "avg_response_time_ms": float(row.get("Average Response Time") or 0),
        "p95_response_time_ms": float(row.get("95%") or 0),
        "rps": float(row.get("Requests/s") or 0),
    }


def _read_failure_breakdown(failures_file: Path) -> list[dict]:
    if not failures_file.exists():
        return []
    rows = []
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            with failures_file.open("r", encoding=enc, errors="ignore") as f:
                rows = list(csv.DictReader(f))
            break
        except UnicodeDecodeError:
            rows = []
            continue
    breakdown = []
    for row in rows:
        occurrences = int(float(row.get("Occurrences") or 0))
        if occurrences <= 0:
            continue
        breakdown.append(
            {
                "method": row.get("Method"),
                "name": row.get("Name"),
                "error": row.get("Error"),
                "occurrences": occurrences,
            }
        )
    breakdown.sort(key=lambda x: x["occurrences"], reverse=True)
    return breakdown


def main() -> None:
    parser = argparse.ArgumentParser(description="自动执行 10/30/50 并发性能测试")
    parser.add_argument("--base-url", type=str, default=None, help="API 基地址")
    parser.add_argument("--duration-min", type=int, default=None, help="每档持续分钟")
    parser.add_argument(
        "--chat-mode",
        type=str,
        choices=["mixed", "rag_only", "normal_only"],
        default="mixed",
        help="压测聊天模式：mixed/rag_only/normal_only",
    )
    args = parser.parse_args()

    cfg = load_metrics_config()
    base_url = args.base_url or cfg["environment"]["base_url"]
    duration_minutes = args.duration_min or cfg["performance"]["duration_minutes"]
    user_levels = cfg["performance"]["user_levels"]
    spawn_rates = cfg["performance"]["spawn_rate"]

    locust_file = ROOT_DIR / "performance" / "locustfile.py"
    out_dir = ensure_dir(ROOT_DIR / "results" / "performance")
    run_reports: list[dict] = []

    for users, spawn_rate in zip(user_levels, spawn_rates):
        prefix = out_dir / f"c{users}"
        return_code = _run_once(
            locust_file=locust_file,
            users=users,
            spawn_rate=spawn_rate,
            duration_minutes=duration_minutes,
            base_url=base_url,
            output_prefix=prefix,
            chat_mode=args.chat_mode,
        )
        run_reports.append(
            {
                "users": users,
                "spawn_rate": spawn_rate,
                "duration_minutes": duration_minutes,
                "chat_mode": args.chat_mode,
                "csv_prefix": str(prefix),
                "return_code": return_code,
                "status": "success" if return_code == 0 else "has_errors",
                "aggregated_metrics": _read_aggregated_stats(Path(f"{prefix}_stats.csv")),
                "failure_breakdown": _read_failure_breakdown(Path(f"{prefix}_failures.csv")),
            }
        )
        if return_code != 0:
            print(
                f"[警告] 并发档位 users={users} 运行结束但存在错误（exit={return_code}），"
                "已继续执行后续档位。请在 *_stats.csv 和 *_failures.csv 中查看细节。"
            )

    report_path = out_dir / f"run_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump({"runs": run_reports}, f, ensure_ascii=False, indent=2)

    print("[性能测试完成] 结果目录:", out_dir)
    print("[性能测试报告]:", report_path)


if __name__ == "__main__":
    main()
