import argparse
import glob
import os

from asr_framework.debug_trace import trace_step
from asr_framework.pipeline import run_pipeline


def _discover_demo_configs(config_dir: str):
    trace_step("自动发现所有 demo 配置文件", f"config_dir={config_dir}")
    files = sorted(glob.glob(os.path.join(config_dir, "demo*.yaml")))
    if not files:
        files = sorted(glob.glob(os.path.join(config_dir, "demo*.yml")))
    if not files:
        raise FileNotFoundError(f"未找到 demo YAML 配置: {config_dir}/demo*.yaml")
    return files


def main() -> None:
    trace_step("Parse CLI arguments and start demo pipeline")
    parser = argparse.ArgumentParser(description="Run ASR Benchmark V2 scaffold demo")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to one config YAML; if omitted, run all demo*.yaml configs",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for normalizer and operators",
    )
    args = parser.parse_args()

    if args.config:
        trace_step("Run pipeline with chosen config", f"config={args.config}")
        run_pipeline(args.config, debug=args.debug)
        return

    root_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(root_dir, "configs", "runs")
    if not os.path.exists(config_dir):
        config_dir = os.path.join(root_dir, "configs")
    configs = _discover_demo_configs(config_dir)
    trace_step("未显式指定配置，按顺序执行全部 demo 配置", f"count={len(configs)}")
    for cfg in configs:
        trace_step("开始执行 demo 配置", f"config={cfg}")
        run_pipeline(cfg, debug=args.debug)


if __name__ == "__main__":
    trace_step("Entry point invoked for run_demo.py")
    main()
