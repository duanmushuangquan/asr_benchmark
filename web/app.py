import glob
import os
import sys
from dataclasses import asdict
from typing import List

try:
    from flask import Flask, jsonify, render_template, request

    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from asr_framework.debug_trace import trace_benefit, trace_step
from asr_framework.pipeline import run_pipeline

app = Flask(__name__, template_folder=os.path.join(CURRENT_DIR, "templates")) if FLASK_AVAILABLE else None


def list_config_files() -> List[str]:
    trace_step("Discover runnable config files for the web UI")
    pattern = os.path.join(PROJECT_ROOT, "configs", "runs", "demo*.yaml")
    files = sorted(glob.glob(pattern))
    if not files:
        files = sorted(glob.glob(os.path.join(PROJECT_ROOT, "configs", "runs", "demo*.yml")))
    if not files:
        files = sorted(glob.glob(os.path.join(PROJECT_ROOT, "configs", "demo*.yaml")))
    return [os.path.relpath(path, PROJECT_ROOT) for path in files]


def list_run_summaries() -> List[str]:
    trace_step("Discover generated run summaries")
    pattern = os.path.join(PROJECT_ROOT, "outputs", "*", "run_summary.json")
    files = sorted(glob.glob(pattern), reverse=True)
    return [os.path.relpath(path, PROJECT_ROOT) for path in files]


if FLASK_AVAILABLE:

    @app.route("/", methods=["GET"])
    def index():
        trace_step("Render framework demo web page")
        configs = list_config_files()
        summaries = list_run_summaries()
        trace_benefit(
            "Web layer only triggers pipeline entrypoint, so UI and evaluation engine stay cleanly separated.",
            "asr_framework/pipeline.py:14",
        )
        return render_template("index.html", configs=configs, summaries=summaries)


    @app.route("/run", methods=["POST"])
    def run_eval():
        trace_step("Receive web request and run evaluation pipeline")
        selected_config = request.form.get("config_path", "configs/runs/demo_funasr_run.yaml")
        config_abs = os.path.join(PROJECT_ROOT, selected_config)
        trace_step("Execute pipeline from web route", f"config_abs={config_abs}")
        try:
            summary = run_pipeline(config_abs)
            return jsonify({"success": True, "summary": asdict(summary)})
        except Exception as exc:
            trace_step("Pipeline failed inside web route", f"error={exc}")
            return jsonify({"success": False, "error": str(exc)}), 500


if __name__ == "__main__":
    trace_step("Start Flask development server for scaffold web UI")
    if not FLASK_AVAILABLE:
        print("[依赖缺失] 当前环境未安装 Flask，请使用: python3 web/simple_server.py")
        raise SystemExit(1)
    app.run(host="0.0.0.0", port=5077, debug=True)
