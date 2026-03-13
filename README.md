# ASR Benchmark V2 Scaffold

This folder is a fast scaffold that demonstrates a cleaner architecture for:

- dataset contract
- model interface
- inference evaluation pipeline
- reporting outputs

It intentionally uses backend-shaped mock adapters (`funasr_mock` and `sherpa_mock`) so you can review the structure before integrating real ASR engines.
It also includes detailed trace prints with file and line info for learning the flow quickly.

## Quick Start

```bash
cd /home/gjt/data/workspace/hmi_gate/asr_bench_v2_scaffold
python3 run_demo.py
```

By default it runs all `configs/runs/demo*.yaml` (currently FunASR + Sherpa).

Expected outputs:

- `outputs/demo_run/run_summary.json`
- `outputs/demo_run/sample_results.jsonl`
- `outputs/demo_sherpa_run/run_summary.json`
- `outputs/demo_sherpa_run/sample_results.jsonl`

Run Sherpa-style streaming mock:

```bash
python3 run_demo.py --config configs/runs/demo_sherpa_run.yaml
```

Run one specific config:

```bash
python3 run_demo.py --config configs/runs/demo_funasr_run.yaml
```

Enable debug mode for normalizer/operators/evaluator:

```bash
python3 run_demo.py --debug
```

## Web Demo

Recommended (no external dependency):

```bash
cd /home/gjt/data/workspace/hmi_gate/asr_bench_v2_scaffold
python3 web/simple_server.py
```

Open `http://127.0.0.1:5078` to trigger pipeline runs from a page.
All function traces are printed in terminal with source location.
网页评测支持多选数据集，一次请求可批量执行并更新横向对比索引。

Optional Flask version (if Flask is installed):

```bash
python3 web/app.py
```

## 日志打印规则

- 同类信息同色: `流程` 与 `收益` 使用固定类别颜色。
- 同脚本同色: 来自同一 `py` 文件的“文件列/函数列”使用同一颜色。
- 固定列宽: `类型 | 文件:行号 | 函数名 | 作用 | 细节` 对齐，便于快速扫描。
- 全中文描述: 作用说明和收益说明默认输出中文文案。
- 颜色开关: 默认在终端启用颜色；可用 `FORCE_COLOR=1` 强制启用，`NO_COLOR=1` 关闭颜色。

## Why this design is better

- One standard sample schema is used across data collection and evaluation.
- Model adapters implement one stable `BaseASREngine` interface.
- Evaluator logic is model-agnostic and easy to test.
- Reports are generated from per-sample structured results, enabling future badcase tooling.
- Trace logs clearly map each runtime step to concrete source files and line numbers.

## 配置组织（降低第三模型接入成本）

- `configs/models/` 只放模型差异配置。
- `configs/datasets/` 只放数据来源配置。
- `configs/eval/` 只放解码与归一化配置。
- `configs/runs/` 用 `_base` 组合上面三类配置，并给出 `run_name/output`。
- 接入第三个模型通常只需:
  1. 新增一个 `models/<backend>_mock.py` 并在 `models/registry.py` 注册。
  2. 新增一个 `configs/models/<backend>.yaml`（模型参数）。
  3. 新增一个 `configs/runs/demo_<backend>_run.yaml`（组合配置）。

## Directory Layout

```text
asr_bench_v2_scaffold/
  asr_framework/
    contracts.py
    dataset.py
    normalizer.py
    operators.py
    evaluator.py
    report.py
    pipeline.py
    models/
      base.py
      funasr_mock.py
      sherpa_mock.py
      registry.py
  configs/
    datasets/default_dataset.yaml
    eval/default_eval.yaml
    models/funasr_mock.yaml
    models/sherpa_mock.yaml
    runs/demo_funasr_run.yaml
    runs/demo_sherpa_run.yaml
    demo_config.yaml
    demo_sherpa_config.yaml
  datasets/
    demo/
      metadata.jsonl
  outputs/
    _compare/
      run_summaries.jsonl
      latest_table.json
      latest_table.csv
  run_demo.py
  web/
    app.py
    simple_server.py
    templates/index.html
```

## 输出设计（便于网页横向对比）

- 每次运行仍保留模型详细产物：`outputs/<run_dir>/sample_results.jsonl` + `run_summary.json`。
- 同时自动更新全局对比索引：
  - `outputs/_compare/run_summaries.jsonl`：所有历史运行追加日志。
  - `outputs/_compare/latest_table.json`：按 `model_id + dataset_id` 保留最新记录，适合网页直接加载。
  - `outputs/_compare/latest_table.csv`：同上，便于离线分析或导入表格工具。
# asr_benchmark
