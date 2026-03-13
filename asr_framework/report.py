import csv
import json
import os
from datetime import datetime
from dataclasses import asdict
from typing import Dict, List

from asr_framework.contracts import RunSummary, SampleEvaluation
from asr_framework.debug_trace import trace_step


class ReportWriter:
    def __init__(self, output_dir: str):
        trace_step("Initialize report writer and ensure output directory", f"output_dir={output_dir}")
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def write_sample_results(self, results: List[SampleEvaluation], filename: str = "sample_results.jsonl") -> str:
        trace_step("Persist sample-level evaluation results", f"filename={filename} count={len(results)}")
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            for item in results:
                f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
        return path

    def write_run_summary(self, summary: RunSummary, filename: str = "run_summary.json") -> str:
        trace_step("Persist run-level summary", f"filename={filename}")
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(summary), f, ensure_ascii=False, indent=2)
        return path

    def write_comparison_index(
        self,
        summary: RunSummary,
        *,
        config_path: str,
        sample_results_path: str,
        run_summary_path: str,
    ) -> Dict[str, str]:
        outputs_root = _find_outputs_root(self.output_dir)
        compare_dir = os.path.join(outputs_root, "_compare")
        os.makedirs(compare_dir, exist_ok=True)

        now = datetime.now().isoformat(timespec="seconds")
        row = {
            "timestamp": now,
            "run_name": summary.run_name,
            "dataset_id": summary.dataset_id,
            "model_id": summary.model_id,
            "total_samples": summary.total_samples,
            "succeeded_samples": summary.succeeded_samples,
            "failed_samples": summary.failed_samples,
            "wer": summary.aggregate_metrics.get("wer"),
            "cer": summary.aggregate_metrics.get("cer"),
            "avg_latency_sec": summary.aggregate_metrics.get("avg_latency_sec"),
            "p50_latency_sec": summary.aggregate_metrics.get("p50_latency_sec"),
            "p90_latency_sec": summary.aggregate_metrics.get("p90_latency_sec"),
            "streaming": summary.capabilities.get("streaming"),
            "supports_hotwords": summary.capabilities.get("supports_hotwords"),
            "supports_first_token_latency": summary.capabilities.get("supports_first_token_latency"),
            "config_path": config_path,
            "sample_results_path": sample_results_path,
            "run_summary_path": run_summary_path,
        }

        append_path = os.path.join(compare_dir, "run_summaries.jsonl")
        with open(append_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        latest_map: Dict[str, Dict] = {}
        all_rows: List[Dict] = []
        with open(append_path, "r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue
                item = json.loads(text)
                all_rows.append(item)
                key = f"{item.get('model_id')}|{item.get('dataset_id')}"
                if key not in latest_map or item.get("timestamp", "") >= latest_map[key].get("timestamp", ""):
                    latest_map[key] = item

        latest_rows = sorted(
            latest_map.values(),
            key=lambda x: (
                x.get("wer") if x.get("wer") is not None else 999.0,
                x.get("cer") if x.get("cer") is not None else 999.0,
                x.get("avg_latency_sec") if x.get("avg_latency_sec") is not None else 999.0,
            ),
        )

        latest_json = os.path.join(compare_dir, "latest_table.json")
        with open(latest_json, "w", encoding="utf-8") as f:
            json.dump({"rows": latest_rows, "count": len(latest_rows)}, f, ensure_ascii=False, indent=2)

        latest_csv = os.path.join(compare_dir, "latest_table.csv")
        headers = [
            "timestamp",
            "model_id",
            "dataset_id",
            "run_name",
            "wer",
            "cer",
            "avg_latency_sec",
            "p50_latency_sec",
            "p90_latency_sec",
            "total_samples",
            "succeeded_samples",
            "failed_samples",
            "streaming",
            "supports_hotwords",
            "supports_first_token_latency",
            "config_path",
            "run_summary_path",
        ]
        with open(latest_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for item in latest_rows:
                writer.writerow({k: item.get(k) for k in headers})

        trace_step(
            "更新横向对比索引",
            f"jsonl={append_path} latest_json={latest_json} latest_csv={latest_csv} total_records={len(all_rows)}",
        )
        return {
            "run_summaries_jsonl": append_path,
            "latest_table_json": latest_json,
            "latest_table_csv": latest_csv,
        }

    @staticmethod
    def print_run_summary(summary: RunSummary) -> None:
        trace_step("Print concise run summary for quick inspection")
        print("[运行摘要] run_name=", summary.run_name)
        print("[运行摘要] dataset_id=", summary.dataset_id)
        print("[运行摘要] model_id=", summary.model_id)
        print(
            "[运行摘要] samples=",
            summary.total_samples,
            "succeeded=",
            summary.succeeded_samples,
            "failed=",
            summary.failed_samples,
        )
        print("[运行摘要] aggregate_metrics=", summary.aggregate_metrics)
        print("[运行摘要] capabilities=", summary.capabilities)


def _find_outputs_root(path: str) -> str:
    current = os.path.abspath(path)
    while True:
        if os.path.basename(current) == "outputs":
            return current
        parent = os.path.dirname(current)
        if parent == current:
            return os.path.abspath(os.path.join(path, ".."))
        current = parent
