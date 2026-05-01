import csv
import json
from pathlib import Path
from typing import Any

from common import ROOT_DIR, ensure_dir, load_metrics_config, now_str, write_text


RESULTS_DIR = ROOT_DIR / "results"


def _latest_json(path: Path, prefix: str) -> Path | None:
    files = sorted(path.glob(f"{prefix}_*.json"))
    return files[-1] if files else None


def _read_json(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_locust_stats() -> list[dict[str, Any]]:
    perf_dir = RESULTS_DIR / "performance"
    if not perf_dir.exists():
        return []

    records: list[dict[str, Any]] = []
    for stats_file in sorted(perf_dir.glob("*_stats.csv")):
        # Locust 的 csv 默认按名称包含请求行和聚合行
        with stats_file.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        aggregate_row = next(
            (r for r in rows if r.get("Type") == "Aggregated" or r.get("Name") == "Aggregated"),
            None,
        )
        if not aggregate_row:
            continue
        records.append(
            {
                "file": stats_file.name,
                "request_count": aggregate_row.get("Request Count"),
                "failure_count": aggregate_row.get("Failure Count"),
                "median_response_time": aggregate_row.get("Median Response Time"),
                "avg_response_time": aggregate_row.get("Average Response Time"),
                "p95_response_time": aggregate_row.get("95%"),
                "rps": aggregate_row.get("Requests/s"),
            }
        )
    return records


def _load_latest_run_report() -> dict[str, Any]:
    perf_dir = RESULTS_DIR / "performance"
    if not perf_dir.exists():
        return {}
    reports = sorted(perf_dir.glob("run_report_*.json"))
    if not reports:
        return {}
    with reports[-1].open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_baseline_and_latest_run_reports() -> tuple[dict[str, Any], dict[str, Any]]:
    perf_dir = RESULTS_DIR / "performance"
    if not perf_dir.exists():
        return {}, {}
    reports = sorted(perf_dir.glob("run_report_*.json"))
    if len(reports) < 2:
        latest = _read_json(reports[-1]) if reports else {}
        return {}, latest
    # 对比首轮结果与最新回归结果，体现优化收益
    return _read_json(reports[0]), _read_json(reports[-1])


def _build_markdown(
    cfg: dict[str, Any],
    functional: dict[str, Any],
    stability: dict[str, Any],
    perf_rows: list[dict[str, Any]],
    run_report: dict[str, Any],
    previous_run_report: dict[str, Any],
) -> str:
    acceptance = cfg["acceptance"]
    func_summary = functional.get("summary", {})
    sta_summary = stability.get("summary", {})

    perf_section = "无性能测试CSV数据，请先执行Locust压测命令。\n"
    if perf_rows:
        lines = [
            "| 文件 | 请求数 | 失败数 | 平均响应(ms) | P95(ms) | RPS |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for r in perf_rows:
            lines.append(
                f"| {r['file']} | {r['request_count']} | {r['failure_count']} | "
                f"{r['avg_response_time']} | {r['p95_response_time']} | {r['rps']} |"
            )
        perf_section = "\n".join(lines)

    run_report_section = "无档位运行报告，请先执行 run_performance_tests.py。\n"
    runs = run_report.get("runs") if isinstance(run_report, dict) else None
    if isinstance(runs, list) and runs:
        lines = [
            "| 并发用户 | 退出状态 | 请求数 | 失败数 | 失败率 | Avg(ms) | P95(ms) |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for run in runs:
            metrics = run.get("aggregated_metrics") or {}
            lines.append(
                f"| {run.get('users', 'N/A')} | {run.get('status', 'N/A')} | "
                f"{metrics.get('request_count', 'N/A')} | {metrics.get('failure_count', 'N/A')} | "
                f"{metrics.get('failure_rate', 'N/A')} | {metrics.get('avg_response_time_ms', 'N/A')} | "
                f"{metrics.get('p95_response_time_ms', 'N/A')} |"
            )
        run_report_section = "\n".join(lines)

    comparison_section = "无可用的前后对比报告（至少需要两次 run_report）。\n"
    old_runs = previous_run_report.get("runs") if isinstance(previous_run_report, dict) else None
    if isinstance(old_runs, list) and isinstance(runs, list) and old_runs and runs:
        old_map = {int(item.get("users", -1)): item for item in old_runs}
        new_map = {int(item.get("users", -1)): item for item in runs}
        lines = [
            "| 并发用户 | 改进前状态 | 改进后状态 | 变化 |",
            "| ---: | --- | --- | --- |",
        ]
        for users in sorted(set(old_map.keys()) | set(new_map.keys())):
            old_status = old_map.get(users, {}).get("status", "N/A")
            new_status = new_map.get(users, {}).get("status", "N/A")
            delta = "持平"
            if old_status != "success" and new_status == "success":
                delta = "改善"
            elif old_status == "success" and new_status != "success":
                delta = "退化"
            lines.append(f"| {users} | {old_status} | {new_status} | {delta} |")
        comparison_section = "\n".join(lines)

    return f"""# 系统测试结果汇总（自动生成）

生成时间：{now_str()}

## 1. 验收阈值

- 功能通过率 >= {acceptance['functional_pass_rate_min']:.0%}
- 平均响应时间 <= {acceptance['avg_latency_seconds_max']} s
- P95响应时间 <= {acceptance['p95_latency_seconds_max']} s
- 稳定性错误率 <= {acceptance['stability_error_rate_max']:.0%}
- 崩溃次数 <= {acceptance['stability_crash_count_max']}

## 2. 功能测试结果

- 总用例数：{func_summary.get('total_cases', 'N/A')}
- 通过用例数：{func_summary.get('passed_cases', 'N/A')}
- 用例通过率：{func_summary.get('pass_rate', 'N/A')}
- 平均响应时间：{func_summary.get('avg_latency_ms', 'N/A')} ms
- P95响应时间：{func_summary.get('p95_latency_ms', 'N/A')} ms
- RAG检索命中率：{func_summary.get('retrieval_hit_rate', 'N/A')}
- 生成正确率：{func_summary.get('generation_accuracy', 'N/A')}

## 3. 性能测试结果（Locust）

{perf_section}

### 3.1 档位运行报告（run_report）

{run_report_section}

### 3.2 改进前后对比（首轮 vs 最新 run_report）

{comparison_section}

## 4. 稳定性测试结果

- 运行时长（小时）：{sta_summary.get('duration_hours', 'N/A')}
- 并发数：{sta_summary.get('concurrency', 'N/A')}
- 总请求数：{sta_summary.get('total_requests', 'N/A')}
- 错误率：{sta_summary.get('error_rate', 'N/A')}
- 平均响应时间：{sta_summary.get('avg_latency_ms', 'N/A')} ms
- P95响应时间：{sta_summary.get('p95_latency_ms', 'N/A')} ms
- 崩溃次数：{sta_summary.get('crash_count', 'N/A')}

## 5. 结论模板

可将以下句式拷贝至论文第5章：

1) 功能测试共执行 N 条用例，通过 M 条，通过率为 X%，系统核心功能可用。  
2) 在 Y 并发下平均响应时间为 A 秒，P95 为 B 秒，错误率为 C%，满足（或未满足）预设性能目标。  
3) 持续运行 T 小时后系统无崩溃（或出现 K 次崩溃），表明系统具有（或仍需提升）稳定性。  
"""


def main() -> None:
    cfg = load_metrics_config()
    functional_file = _latest_json(RESULTS_DIR / "functional", "functional_report")
    stability_file = _latest_json(RESULTS_DIR / "stability", "stability_report")
    functional_data = _read_json(functional_file)
    stability_data = _read_json(stability_file)
    perf_rows = _load_locust_stats()
    previous_run_report, run_report = _load_baseline_and_latest_run_reports()

    summary_text = _build_markdown(
        cfg, functional_data, stability_data, perf_rows, run_report, previous_run_report
    )
    out_dir = ensure_dir(RESULTS_DIR / "summary")
    out_file = out_dir / "thesis_test_summary.md"
    write_text(out_file, summary_text)
    print(f"[汇总完成] 输出文件: {out_file}")


if __name__ == "__main__":
    main()
