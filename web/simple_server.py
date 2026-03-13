import glob
import json
import os
import random
import sys
import time
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from asr_framework.debug_trace import trace_benefit, trace_step
from asr_framework.pipeline import run_pipeline

CONFIG_RUNS_DIR = os.path.join(PROJECT_ROOT, "configs", "runs")
DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets")
HOTWORDS_PATH = os.path.join(PROJECT_ROOT, "configs", "hotwords.yaml")
COMPARE_DIR = os.path.join(PROJECT_ROOT, "outputs", "_compare")


def list_run_configs():
    trace_step("扫描可运行配置文件")
    files = sorted(glob.glob(os.path.join(CONFIG_RUNS_DIR, "demo*.yaml")))
    if not files:
        files = sorted(glob.glob(os.path.join(CONFIG_RUNS_DIR, "demo*.yml")))
    if not files:
        files = sorted(glob.glob(os.path.join(PROJECT_ROOT, "configs", "demo*.yaml")))
    return [os.path.relpath(path, PROJECT_ROOT) for path in files]


def list_dataset_names():
    trace_step("扫描数据集目录")
    os.makedirs(DATASETS_DIR, exist_ok=True)
    names = sorted(
        [d for d in os.listdir(DATASETS_DIR) if os.path.isdir(os.path.join(DATASETS_DIR, d))]
    )
    return names


def load_hotword_bundle():
    trace_step("读取热词配置", f"path={HOTWORDS_PATH}")
    if not os.path.exists(HOTWORDS_PATH):
        return {"hotwords": [], "sherpa_fst": []}

    try:
        import yaml  # type: ignore

        with open(HOTWORDS_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return {
            "hotwords": list(data.get("hotwords", [])),
            "sherpa_fst": list(data.get("sherpa_fst", [])),
        }
    except Exception:
        hotwords = []
        with open(HOTWORDS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if text.startswith("-"):
                    hotwords.append(text[1:].strip().strip('"').strip("'"))
        return {"hotwords": hotwords, "sherpa_fst": []}


def save_hotword_bundle(bundle):
    trace_step("写入热词配置", f"path={HOTWORDS_PATH}")
    os.makedirs(os.path.dirname(HOTWORDS_PATH), exist_ok=True)
    try:
        import yaml  # type: ignore

        with open(HOTWORDS_PATH, "w", encoding="utf-8") as f:
            yaml.safe_dump(bundle, f, allow_unicode=True, sort_keys=False)
    except Exception:
        lines = ["hotwords:\n"]
        for word in bundle.get("hotwords", []):
            lines.append(f'  - "{word}"\n')
        lines.append("\n")
        lines.append("sherpa_fst:\n")
        for item in bundle.get("sherpa_fst", []):
            name = str(item.get("name", ""))
            pinyin = str(item.get("pinyin", "pending"))
            lines.append(f'  - name: "{name}"\n')
            lines.append(f'    pinyin: "{pinyin}"\n')
        with open(HOTWORDS_PATH, "w", encoding="utf-8") as f:
            f.writelines(lines)


def ensure_hotword_registered(word: str):
    value = (word or "").strip()
    if not value or value.lower() == "no":
        return
    bundle = load_hotword_bundle()
    if value in bundle["hotwords"]:
        return
    bundle["hotwords"].append(value)
    bundle["sherpa_fst"].append({"name": value, "pinyin": "pending"})
    save_hotword_bundle(bundle)
    trace_benefit(
        "采集页输入新热词后自动沉淀到 configs/hotwords.yaml，后续可用于筛选子数据集和导出专用词表。",
        "web/simple_server.py:88",
    )


def _safe_dataset_name(name: str) -> str:
    value = (name or "").strip()
    if not value:
        return "demo"
    return "".join(ch for ch in value if ch.isalnum() or ch in ("_", "-", "."))


def _sanitize_dataset_name(name: str) -> str:
    value = (name or "").strip()
    if not value:
        return ""
    return "".join(ch for ch in value if ch.isalnum() or ch in ("_", "-", "."))


def _build_sample_record(form):
    dataset_name = _safe_dataset_name(form.get("dataset_name", ["demo"])[0])
    text = str(form.get("text", [""])[0]).strip()
    language = str(form.get("language", ["zh"])[0]).strip() or "zh"
    hotword = str(form.get("hotword", ["no"])[0]).strip() or "no"
    task_scene = str(form.get("task_scene", ["ask"])[0]).strip() or "ask"
    speaker_type = str(form.get("speaker_type", ["single"])[0]).strip() or "single"
    speech_style = str(form.get("speech_style", ["normal"])[0]).strip() or "normal"

    now = int(time.time())
    sample_id = f"smp_{now}_{random.randint(100, 999)}"
    record = {
        "sample_id": sample_id,
        "audio_path": f"audio/{sample_id}.wav",
        "text": text,
        "timestamp": now,
        "language": language,
        "hotword": hotword,
        "task_scene": task_scene,
        "speaker_type": speaker_type,
        "speech_style": speech_style,
        "device": str(form.get("device", ["unknown"])[0]),
        "distance": str(form.get("distance", ["unknown"])[0]),
        "noise": str(form.get("noise", ["unknown"])[0]),
        "dataset_name": dataset_name,
    }
    return dataset_name, sample_id, record


def save_sample_record(form):
    trace_step("保存采集样本到数据集")
    dataset_name, sample_id, record = _build_sample_record(form)
    if not record["text"]:
        raise ValueError("text 不能为空")

    dataset_dir = os.path.join(DATASETS_DIR, dataset_name)
    audio_dir = os.path.join(dataset_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    audio_file = os.path.join(audio_dir, f"{sample_id}.wav")
    # 虚拟流程: 先落空 wav 占位文件，后续可替换真实录音。
    with open(audio_file, "wb") as f:
        f.write(b"")

    metadata_path = os.path.join(dataset_dir, "metadata.jsonl")
    with open(metadata_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    ensure_hotword_registered(record["hotword"])
    return {
        "dataset_name": dataset_name,
        "sample_id": sample_id,
        "metadata_path": metadata_path,
        "audio_path": record["audio_path"],
    }


def list_records(dataset_name: str = ""):
    trace_step("读取数据集记录用于管理页面", f"dataset_name={dataset_name}")
    target = _safe_dataset_name(dataset_name) if dataset_name else ""
    records = []
    for ds_name in list_dataset_names():
        if target and ds_name != target:
            continue
        path = os.path.join(DATASETS_DIR, ds_name, "metadata.jsonl")
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                item["_dataset"] = ds_name
                records.append(item)
    records.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
    return records


def run_virtual_eval(config_rel: str, dataset_name: str = "", debug: bool = False):
    trace_step("按网页请求触发虚拟评测流程", f"config={config_rel} dataset_name={dataset_name}")
    config_abs = os.path.join(PROJECT_ROOT, config_rel)
    if not os.path.exists(config_abs):
        raise FileNotFoundError(f"配置文件不存在: {config_abs}")

    dataset_name = _sanitize_dataset_name(dataset_name)
    if dataset_name:
        temp_name = f"_web_run_{dataset_name}.yaml"
        temp_path = os.path.join(CONFIG_RUNS_DIR, temp_name)
        override = {
            "_base": os.path.relpath(config_abs, CONFIG_RUNS_DIR),
            "dataset": {
                "dataset_id": dataset_name,
                "metadata_path": f"../../datasets/{dataset_name}/metadata.jsonl",
            },
            "run_name": f"web_run_{dataset_name}",
            "output": {
                "dir": f"../../outputs/web_runs/{os.path.splitext(os.path.basename(config_abs))[0]}/{dataset_name}"
            },
        }
        _write_yaml_config(temp_path, override)
        summary = run_pipeline(temp_path, debug=debug)
    else:
        summary = run_pipeline(config_abs, debug=debug)
    return asdict(summary)


def run_virtual_eval_multi(config_rel: str, dataset_names, debug: bool = False):
    clean = []
    for raw in dataset_names:
        name = _sanitize_dataset_name(raw)
        if not name:
            continue
        if name not in clean:
            clean.append(name)
    if not clean:
        payload = {"summaries": [run_virtual_eval(config_rel, "", debug=debug)]}
        payload["compare_latest"] = _load_compare_latest()
        payload["dataset_count"] = 0
        return payload

    summaries = []
    for ds in clean:
        summaries.append(run_virtual_eval(config_rel, ds, debug=debug))

    return {"summaries": summaries, "compare_latest": _load_compare_latest(), "dataset_count": len(clean)}


def _load_compare_latest():
    latest_json = os.path.join(COMPARE_DIR, "latest_table.json")
    if os.path.exists(latest_json):
        with open(latest_json, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"rows": [], "count": 0}


def _write_yaml_config(path: str, payload: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        import yaml  # type: ignore

        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)
        return
    except Exception:
        # JSON is valid YAML 1.2; fallback keeps web flow available.
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, indent=2))


def _json_response(handler, payload, status=200):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _render_collect_page():
    datasets = list_dataset_names()
    hotwords = load_hotword_bundle().get("hotwords", [])
    configs = list_run_configs()
    dataset_options = "\n".join([f'<option value="{d}">{d}</option>' for d in datasets])
    hotword_options = '\n'.join([f'<option value="{w}">{w}</option>' for w in hotwords])
    config_options = "\n".join([f'<option value="{c}">{c}</option>' for c in configs])
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>ASR V2 采集与评测</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f7fb; color: #1f2d3d; }}
    .card {{ background: #fff; border: 1px solid #dce5f2; border-radius: 10px; padding: 14px; margin-bottom: 14px; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
    input, select, textarea, button {{ width: 100%; box-sizing: border-box; padding: 8px; margin-top: 6px; }}
    button {{ background: #2f74c0; color: #fff; border: none; border-radius: 8px; cursor: pointer; }}
    .btn2 {{ background: #27ae60; }}
    pre {{ background: #edf3fc; padding: 10px; border-radius: 6px; }}
    a {{ color: #2f74c0; text-decoration: none; }}
  </style>
</head>
<body>
  <div class="card">
    <h2>采集页面（虚拟流程）</h2>
    <p>保存 metadata 到 datasets，再直接触发 run_demo 同源评测流程。</p>
    <a href="/manage">进入管理页面</a>
  </div>

  <div class="card">
    <div class="grid">
      <div>
        <label>数据集目录（已有）</label>
        <select id="dataset_existing">{dataset_options}</select>
      </div>
      <div>
        <label>新数据集名称（可选）</label>
        <input id="dataset_new" placeholder="例如: dataset_v2" />
      </div>
      <div>
        <label>语言</label>
        <select id="language">
          <option value="zh">中文</option>
          <option value="en">英文</option>
          <option value="mixed">中英混合</option>
        </select>
      </div>
      <div>
        <label>hotword（可选 no）</label>
        <input id="hotword" list="hotword_list" placeholder="no 或输入热词" />
        <datalist id="hotword_list">
          <option value="no">no</option>
          {hotword_options}
        </datalist>
      </div>
      <div>
        <label>任务场景</label>
        <select id="task_scene">
          <option value="grasp">抓取场景</option>
          <option value="nav">导航场景</option>
          <option value="ask">问询交流</option>
        </select>
      </div>
      <div>
        <label>说话人</label>
        <select id="speaker_type">
          <option value="single">单人</option>
          <option value="multi">多人</option>
          <option value="other">其他</option>
        </select>
      </div>
      <div>
        <label>语音特点</label>
        <select id="speech_style">
          <option value="normal">正常句式</option>
          <option value="long_hard">长难句</option>
          <option value="wake_word">唤醒词</option>
        </select>
      </div>
      <div>
        <label>设备</label>
        <input id="device" value="web_virtual_mic" />
      </div>
      <div>
        <label>距离</label>
        <input id="distance" value="near" />
      </div>
      <div>
        <label>噪音</label>
        <input id="noise" value="quiet" />
      </div>
    </div>
    <label>真值文本</label>
    <textarea id="text" rows="3" placeholder="输入真值文本"></textarea>
    <button onclick="saveSample()">保存到数据集</button>
    <pre id="save_out">等待保存...</pre>
  </div>

  <div class="card">
    <h3>触发评测</h3>
    <div class="grid">
      <div>
        <label>运行配置</label>
        <select id="run_config">{config_options}</select>
      </div>
      <div>
        <label>评测数据集（可多选，空表示配置内默认）</label>
        <select id="run_datasets" multiple size="6">{dataset_options}</select>
      </div>
    </div>
    <p>提示: 按住 Ctrl/Command 可多选数据集。</p>
    <button class="btn2" onclick="runEval()">执行评测</button>
    <pre id="run_out">等待执行...</pre>
  </div>

  <div class="card">
    <h3>横向对比（最新）</h3>
    <button onclick="loadCompare()">刷新对比表</button>
    <div id="compare_out">暂无对比数据</div>
  </div>

  <script>
    async function saveSample() {{
      const datasetNew = document.getElementById("dataset_new").value.trim();
      const datasetExisting = document.getElementById("dataset_existing").value;
      const datasetName = datasetNew || datasetExisting || "demo";
      const body = new URLSearchParams();
      body.set("dataset_name", datasetName);
      body.set("language", document.getElementById("language").value);
      body.set("hotword", document.getElementById("hotword").value || "no");
      body.set("task_scene", document.getElementById("task_scene").value);
      body.set("speaker_type", document.getElementById("speaker_type").value);
      body.set("speech_style", document.getElementById("speech_style").value);
      body.set("device", document.getElementById("device").value);
      body.set("distance", document.getElementById("distance").value);
      body.set("noise", document.getElementById("noise").value);
      body.set("text", document.getElementById("text").value);
      const resp = await fetch("/api/save_sample", {{
        method: "POST",
        headers: {{ "Content-Type": "application/x-www-form-urlencoded" }},
        body
      }});
      const data = await resp.json();
      document.getElementById("save_out").textContent = JSON.stringify(data, null, 2);
    }}

    async function runEval() {{
      const body = new URLSearchParams();
      body.set("config_path", document.getElementById("run_config").value);
      const dsSelect = document.getElementById("run_datasets");
      const selected = Array.from(dsSelect.selectedOptions).map(x => x.value).filter(Boolean);
      for (const ds of selected) {{
        body.append("dataset_names", ds);
      }}
      const resp = await fetch("/api/run_eval", {{
        method: "POST",
        headers: {{ "Content-Type": "application/x-www-form-urlencoded" }},
        body
      }});
      const data = await resp.json();
      document.getElementById("run_out").textContent = JSON.stringify(data, null, 2);
      if (data.success && data.compare_latest) {{
        renderCompare(data.compare_latest);
      }}
    }}

    async function loadCompare() {{
      const resp = await fetch("/api/compare_latest", {{ method: "POST" }});
      const data = await resp.json();
      if (data.success) {{
        renderCompare(data.compare_latest);
      }}
    }}

    function renderCompare(payload) {{
      const rows = (payload && payload.rows) ? payload.rows : [];
      if (!rows.length) {{
        document.getElementById("compare_out").textContent = "暂无对比数据";
        return;
      }}
      const head = "<tr><th>时间</th><th>模型</th><th>数据集</th><th>WER</th><th>CER</th><th>平均时延</th><th>样本数</th></tr>";
      const body = rows.map(r => {{
        return `<tr>
          <td>${{r.timestamp || ""}}</td>
          <td>${{r.model_id || ""}}</td>
          <td>${{r.dataset_id || ""}}</td>
          <td>${{(r.wer ?? "").toString()}}</td>
          <td>${{(r.cer ?? "").toString()}}</td>
          <td>${{(r.avg_latency_sec ?? "").toString()}}</td>
          <td>${{(r.total_samples ?? "").toString()}}</td>
        </tr>`;
      }}).join("");
      document.getElementById("compare_out").innerHTML = `<table border="1" cellpadding="6" cellspacing="0" style="width:100%;border-collapse:collapse;"><thead>${{head}}</thead><tbody>${{body}}</tbody></table>`;
    }}
  </script>
</body>
</html>
"""


def _render_manage_page(query):
    selected_dataset = query.get("dataset", [""])[0].strip()
    records = list_records(selected_dataset)
    datasets = list_dataset_names()
    dataset_options = ['<option value="">全部</option>'] + [
        f'<option value="{d}" {"selected" if d == selected_dataset else ""}>{d}</option>' for d in datasets
    ]
    rows = []
    for item in records[:200]:
        rows.append(
            "<tr>"
            f"<td>{item.get('sample_id', '')}</td>"
            f"<td>{item.get('_dataset', '')}</td>"
            f"<td>{item.get('language', '')}</td>"
            f"<td>{item.get('task_scene', '')}</td>"
            f"<td>{item.get('speaker_type', '')}</td>"
            f"<td>{item.get('speech_style', '')}</td>"
            f"<td>{item.get('hotword', '')}</td>"
            f"<td>{item.get('text', '')}</td>"
            "</tr>"
        )
    table_rows = "\n".join(rows) or "<tr><td colspan='8'>暂无数据</td></tr>"
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>ASR V2 数据管理</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f7fb; }}
    .card {{ background: #fff; border: 1px solid #dce5f2; border-radius: 10px; padding: 14px; margin-bottom: 14px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border-bottom: 1px solid #e8eef7; padding: 8px; text-align: left; }}
  </style>
</head>
<body>
  <div class="card">
    <h2>数据集管理页面</h2>
    <a href="/">返回采集页面</a>
    <form method="GET" action="/manage">
      <label>数据集筛选</label>
      <select name="dataset">{''.join(dataset_options)}</select>
      <button type="submit">筛选</button>
    </form>
    <p>记录总数: {len(records)}</p>
  </div>
  <div class="card">
    <table>
      <thead>
        <tr>
          <th>ID</th><th>数据集</th><th>语言</th><th>任务场景</th>
          <th>说话人</th><th>语音特点</th><th>热词</th><th>真值文本</th>
        </tr>
      </thead>
      <tbody>{table_rows}</tbody>
    </table>
  </div>
</body>
</html>
"""


class DemoHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        trace_step("处理 GET 请求", f"path={parsed.path}")
        if parsed.path in ("/", "/collect"):
            body = _render_collect_page().encode("utf-8")
        elif parsed.path == "/manage":
            body = _render_manage_page(query).encode("utf-8")
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        parsed = urlparse(self.path)
        trace_step("处理 POST 请求", f"path={parsed.path}")
        content_len = int(self.headers.get("Content-Length", "0"))
        payload = self.rfile.read(content_len).decode("utf-8")
        form = parse_qs(payload)
        try:
            if parsed.path == "/api/save_sample":
                saved = save_sample_record(form)
                _json_response(self, {"success": True, "saved": saved})
                return
            if parsed.path == "/api/run_eval":
                config_rel = form.get("config_path", ["configs/runs/demo_funasr_run.yaml"])[0]
                dataset_names = form.get("dataset_names", [])
                if not dataset_names:
                    old_single = form.get("dataset_name", [""])[0].strip()
                    if old_single:
                        dataset_names = [old_single]
                debug = str(form.get("debug", ["false"])[0]).lower() in ("1", "true", "yes")
                payload = run_virtual_eval_multi(config_rel, dataset_names, debug=debug)
                _json_response(self, {"success": True, **payload})
                return
            if parsed.path == "/api/compare_latest":
                _json_response(self, {"success": True, "compare_latest": _load_compare_latest()})
                return
            _json_response(self, {"success": False, "error": "unknown endpoint"}, status=404)
        except Exception as exc:
            trace_step("网页接口执行失败", f"error={exc}")
            _json_response(self, {"success": False, "error": str(exc)}, status=500)


def main() -> None:
    trace_step("启动一体化采集与评测网页服务")
    address = ("0.0.0.0", 5078)
    server = ThreadingHTTPServer(address, DemoHandler)
    print(f"[服务启动] 采集与评测服务已启动: http://127.0.0.1:{address[1]}")
    trace_benefit(
        "采集、管理、评测触发都在同一个 web 入口，形成端到端虚拟闭环，便于先验证流程再接真实录音链路。",
        "web/simple_server.py:429",
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
