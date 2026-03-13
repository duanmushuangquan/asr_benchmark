import os
from typing import Any, Dict

from asr_framework.contracts import DecodeOptions, RunSummary
from asr_framework.dataset import load_samples_from_jsonl
from asr_framework.debug_trace import trace_benefit, trace_step
from asr_framework.evaluator import Evaluator
from asr_framework.models.registry import create_engine
from asr_framework.normalizer import NormalizerConfig, TextNormalizer
from asr_framework.operators import aggregate_results
from asr_framework.report import ReportWriter


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def run_pipeline(config_path: str, debug: bool = False) -> RunSummary:
    trace_step("Orchestrate full benchmark pipeline", f"config_path={config_path}")
    config = _load_config(config_path)
    config_dir = os.path.dirname(os.path.abspath(config_path))

    dataset_cfg = config.get("dataset", {})
    model_cfg = config.get("model", {})
    decode_cfg = config.get("decode", {})
    normalizer_cfg = config.get("normalizer", {})
    output_cfg = config.get("output", {})
    debug_mode = bool(config.get("debug", False) or debug)
    trace_step("流水线调试模式", f"debug={debug_mode}")

    metadata_path = _resolve_path(config_dir, dataset_cfg.get("metadata_path", ""))
    output_dir = _resolve_path(config_dir, output_cfg.get("dir", "../outputs/default_run"))
    run_name = str(config.get("run_name", "default_run"))
    dataset_id = str(dataset_cfg.get("dataset_id", "dataset"))

    trace_step("Resolved key IO paths", f"metadata_path={metadata_path} output_dir={output_dir}")
    samples = load_samples_from_jsonl(metadata_path)
    hotwords_base = list(decode_cfg.get("hotwords", []))
    if not hotwords_base and decode_cfg.get("hotwords_yaml"):
        hotwords_yaml = _resolve_path(config_dir, decode_cfg.get("hotwords_yaml"))
        hotwords_base = _load_hotwords_from_yaml(hotwords_yaml)
        trace_step("从热词配置文件加载默认热词", f"hotwords_yaml={hotwords_yaml} count={len(hotwords_base)}")
    decode_options = DecodeOptions(
        language=decode_cfg.get("language"),
        hotwords=hotwords_base,
        extra=dict(decode_cfg.get("extra", {})),
    )

    normalizer = TextNormalizer(
        NormalizerConfig(
            to_lower=bool(normalizer_cfg.get("to_lower", True)),
            remove_punctuation=bool(normalizer_cfg.get("remove_punctuation", True)),
            strip_spaces=bool(normalizer_cfg.get("strip_spaces", True)),
            debug=debug_mode,
        )
    )

    engine = create_engine(model_cfg)
    trace_benefit(
        "Pipeline never imports concrete model adapters directly. It depends only on registry + BaseASREngine contract.",
        "asr_framework/models/base.py:7",
    )
    engine.load()
    try:
        evaluator = Evaluator(engine, normalizer, debug=debug_mode)
        sample_results = evaluator.evaluate(samples, decode_options)
    finally:
        engine.close()

    aggregate = aggregate_results(sample_results, debug=debug_mode)
    summary = RunSummary(
        run_name=run_name,
        dataset_id=dataset_id,
        model_id=engine.model_id,
        total_samples=len(samples),
        succeeded_samples=len(sample_results),
        failed_samples=max(0, len(samples) - len(sample_results)),
        aggregate_metrics=aggregate,
        capabilities=engine.capabilities().__dict__,
    )

    writer = ReportWriter(output_dir)
    sample_path = writer.write_sample_results(sample_results)
    summary_path = writer.write_run_summary(summary)
    compare_paths = writer.write_comparison_index(
        summary,
        config_path=os.path.abspath(config_path),
        sample_results_path=sample_path,
        run_summary_path=summary_path,
    )
    writer.print_run_summary(summary)
    print(f"[输出文件] sample_results={sample_path}")
    print(f"[输出文件] run_summary={summary_path}")
    print(f"[输出文件] compare_jsonl={compare_paths['run_summaries_jsonl']}")
    print(f"[输出文件] compare_latest_json={compare_paths['latest_table_json']}")
    print(f"[输出文件] compare_latest_csv={compare_paths['latest_table_csv']}")
    trace_step("Pipeline completed", f"summary_model={summary.model_id}")
    return summary


def _load_config(path: str, visited=None) -> Dict[str, Any]:
    trace_step("Load config from disk", f"path={path}")
    if visited is None:
        visited = set()
    abs_path = os.path.abspath(path)
    if abs_path in visited:
        raise ValueError(f"检测到配置循环引用: {abs_path}")
    visited.add(abs_path)

    config = _parse_config_file(abs_path)

    base_items = config.get("_base", config.get("_base_", []))
    if isinstance(base_items, str):
        base_items = [base_items]

    merged: Dict[str, Any] = {}
    config_dir = os.path.dirname(abs_path)
    for item in base_items:
        base_path = item if os.path.isabs(item) else os.path.join(config_dir, item)
        trace_step("加载基础配置并执行递归合并", f"base_path={base_path}")
        merged = _deep_merge(merged, _load_config(base_path, visited=visited.copy()))

    current = dict(config)
    current.pop("_base", None)
    current.pop("_base_", None)
    return _deep_merge(merged, current)


def _parse_config_file(path: str) -> Dict[str, Any]:
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    if ext in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore

            data = yaml.safe_load(raw)
            if data is None:
                return {}
            if not isinstance(data, dict):
                raise ValueError(f"配置文件根节点必须是字典: {path}")
            return data
        except ImportError as exc:
            raise RuntimeError(
                "读取 YAML 配置需要安装 pyyaml: pip install pyyaml"
            ) from exc

    if ext == ".json":
        import json

        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError(f"配置文件根节点必须是字典: {path}")
        return data

    raise ValueError(f"不支持的配置后缀: {path}")


def _resolve_path(base_dir: str, path: str) -> str:
    trace_step("Resolve relative/absolute path", f"base_dir={base_dir} path={path}")
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(base_dir, path))


def _load_hotwords_from_yaml(path: str):
    trace_step("读取 hotwords 配置文件", f"path={path}")
    if not os.path.exists(path):
        return []
    try:
        import yaml  # type: ignore

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        values = data.get("hotwords", [])
        return [str(x).strip() for x in values if str(x).strip()]
    except Exception:
        # Fallback parser for simple yaml list style:
        # hotwords:
        #   - word1
        #   - word2
        words = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if text.startswith("-"):
                    value = text[1:].strip().strip('"').strip("'")
                    if value:
                        words.append(value)
        return words
