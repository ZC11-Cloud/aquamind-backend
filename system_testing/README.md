# RAG系统测试实操包

本目录用于直接落地论文中的系统测试章节，覆盖功能测试、性能测试、稳定性测试以及结果汇总。

## 1. 目录结构

- `config/metrics.yaml`：测试阈值、环境参数、测试账号配置。
- `data/rag_eval_dataset.csv`：100条问答测试集（含2/1/0评分规则字段）。
- `data/functional_test_cases.json`：36条功能测试用例（正常/边界/异常）。
- `scripts/capture_test_baseline.py`：采集测试环境与验收阈值快照。
- `scripts/run_functional_tests.py`：执行功能测试并输出统计结果。
- `performance/locustfile.py`：性能压测脚本（10/30/50并发）。
- `scripts/run_performance_tests.py`：自动执行10/30/50并发性能压测。
- `scripts/run_stability_test.py`：稳定性测试脚本（默认8小时，可调）。
- `scripts/summarize_results.py`：聚合功能/性能/稳定性结果，生成论文结果草稿。
- `thesis/chapter5_template.md`：可直接改写进论文的第5章模板。

## 2. 前置准备

1. 启动后端服务（默认 `http://127.0.0.1:8000`）。
2. 准备测试账号（脚本会自动尝试注册，不存在则创建）。
3. 确认 `.env` 已配置数据库和大模型密钥。
4. 安装额外测试依赖：

```bash
pip install locust
```

## 3. 执行顺序

### 3.0 采集基线

```bash
python system_testing/scripts/capture_test_baseline.py --base-url http://127.0.0.1:8000
```

输出目录：`system_testing/results/baseline/`

### 3.1 功能测试

```bash
python system_testing/scripts/run_functional_tests.py --base-url http://127.0.0.1:8000
```

输出目录：`system_testing/results/functional/`

### 3.2 性能测试（Locust）

自动执行三档并发（建议每档10-15分钟）：

```bash
python system_testing/scripts/run_performance_tests.py --base-url http://127.0.0.1:8000 --duration-min 10
```

手动执行命令（可选）：

```bash
locust -f system_testing/performance/locustfile.py --headless -u 10 -r 2 -t 10m --csv=system_testing/results/performance/c10
locust -f system_testing/performance/locustfile.py --headless -u 30 -r 5 -t 10m --csv=system_testing/results/performance/c30
locust -f system_testing/performance/locustfile.py --headless -u 50 -r 8 -t 10m --csv=system_testing/results/performance/c50
```

### 3.3 稳定性测试

默认 8 小时、20 并发、每 30 分钟采样一次：

```bash
python system_testing/scripts/run_stability_test.py --base-url http://127.0.0.1:8000
```

输出目录：`system_testing/results/stability/`

### 3.4 结果汇总

```bash
python system_testing/scripts/summarize_results.py
```

输出文件：`system_testing/results/summary/thesis_test_summary.md`

## 4. 论文验收阈值（默认）

- 功能用例通过率 >= 95%
- 平均响应时间 <= 2.5s
- P95 响应时间 <= 4.5s
- 稳定性测试错误率 <= 1%
- 稳定性测试期间服务崩溃次数 = 0

阈值可在 `config/metrics.yaml` 中按实际情况调整，并在论文中注明原因。
