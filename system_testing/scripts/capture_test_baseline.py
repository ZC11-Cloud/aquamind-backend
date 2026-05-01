import argparse
import platform
import socket
from datetime import datetime

import psutil

from common import build_results_path, load_metrics_config, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="采集测试基线环境与验收阈值")
    parser.add_argument("--base-url", type=str, default=None, help="API 基地址（可选覆盖）")
    args = parser.parse_args()

    cfg = load_metrics_config()
    if args.base_url:
        cfg["environment"]["base_url"] = args.base_url

    baseline = {
        "captured_at": datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "system": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "processor": platform.processor(),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "memory_total_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        },
        "environment_config": cfg["environment"],
        "acceptance_thresholds": cfg["acceptance"],
    }

    out_path = build_results_path("baseline", "baseline_env")
    write_json(out_path, baseline)
    print(f"[基线采集完成] 输出文件: {out_path}")


if __name__ == "__main__":
    main()
