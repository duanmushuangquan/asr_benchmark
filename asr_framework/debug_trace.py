import hashlib
import inspect
import os
import sys
from datetime import datetime

# 规则:
# 1) 同类日志固定类型颜色: 流程=青色, 收益=洋红色
# 2) 同一个脚本固定文件颜色: 由文件路径 hash 到调色板
# 3) 固定列宽输出: 类型/文件行号/函数名对齐, 便于快速扫读

_FILE_COLOR_POOL = [
    "\033[38;5;75m",   # blue
    "\033[38;5;81m",   # cyan
    "\033[38;5;114m",  # green
    "\033[38;5;179m",  # yellow
    "\033[38;5;180m",  # light yellow
    "\033[38;5;141m",  # purple
    "\033[38;5;207m",  # pink
]
_TYPE_COLOR = {
    "流程": "\033[36m",
    "收益": "\033[35m",
}
_RESET = "\033[0m"
_USE_COLOR = (
    os.getenv("NO_COLOR") is None
    and (os.getenv("FORCE_COLOR") == "1" or sys.stdout.isatty())
)

_CN_MAP = {
    "Entry point invoked for run_demo.py": "进入 run_demo 主入口",
    "Parse CLI arguments and start demo pipeline": "解析命令行参数并启动演示流水线",
    "Run pipeline with chosen config": "按指定配置执行评测流水线",
    "Load dataset samples from JSONL metadata": "从 JSONL 元数据加载样本",
    "Parse one metadata line into AudioSample": "将单行元数据解析为 AudioSample",
    "Dataset loading complete": "数据集加载完成",
    "Initialize text normalizer with explicit config": "按配置初始化文本归一化器",
    "Normalize text for fair metric comparison": "对文本做归一化，保证指标可比",
    "Text normalization complete": "文本归一化完成",
    "Compute edit distance core for WER/CER": "计算 WER/CER 的编辑距离核心",
    "Calculate WER": "计算 WER",
    "WER calculation complete": "WER 计算完成",
    "Calculate CER": "计算 CER",
    "CER calculation complete": "CER 计算完成",
    "Calculate hotword metrics": "计算热词相关指标",
    "Hotword metrics complete": "热词指标计算完成",
    "Aggregate sample-level metrics into run summary": "将样本级指标聚合为运行级摘要",
    "Aggregation complete": "指标聚合完成",
    "Compute percentile helper": "计算分位数",
    "Percentile helper complete": "分位数计算完成",
    "Initialize report writer and ensure output directory": "初始化报告写入器并确保输出目录存在",
    "Persist sample-level evaluation results": "写出样本级评测结果",
    "Persist run-level summary": "写出运行级摘要",
    "Print concise run summary for quick inspection": "打印运行摘要便于快速查看",
    "Orchestrate full benchmark pipeline": "编排执行完整 Benchmark 流水线",
    "Resolved key IO paths": "解析关键输入输出路径",
    "Pipeline completed": "流水线执行完成",
    "Load JSON config from disk": "从磁盘读取 JSON 配置",
    "Resolve relative/absolute path": "解析相对/绝对路径",
    "Initialize evaluator with pluggable model engine and normalizer": "初始化评测器（可插拔模型+归一化器）",
    "Run sample-by-sample evaluation loop": "执行逐样本评测循环",
    "Evaluate one sample": "评测单个样本",
    "Evaluation loop complete": "评测循环完成",
    "Expose stable model_id for reporting and comparison": "暴露稳定 model_id 供报告和对比使用",
    "Expose backend capabilities to evaluator/report": "暴露模型能力给评测和报告层",
    "Simulate model load": "模拟模型加载",
    "Simulate one inference call with stable interface": "通过统一接口模拟一次推理",
    "Simulate model resource release": "模拟模型资源释放",
    "Generate deterministic mock prediction for repeatable demo": "生成可复现的 mock 预测结果",
    "Create model engine from registry": "通过注册表创建模型引擎",
    "Initialize base engine with backend-specific config": "使用后端配置初始化基础模型引擎",
    "Default no-op load in base class": "基类默认空实现 load",
    "Default no-op close in base class": "基类默认空实现 close",
    "Discover runnable config files for the web UI": "扫描网页可运行配置文件",
    "Discover generated run summaries": "扫描已生成的运行摘要",
    "Render framework demo web page": "渲染框架演示网页",
    "Receive web request and run evaluation pipeline": "接收网页请求并运行评测流水线",
    "Execute pipeline from web route": "从网页路由执行流水线",
    "Pipeline failed inside web route": "网页路由中流水线执行失败",
    "Start Flask development server for scaffold web UI": "启动 Flask 演示服务",
    "List config files for web selector": "列出网页下拉可选配置",
    "List recent run summary files": "列出最近运行摘要文件",
    "Render HTML page dynamically": "动态渲染 HTML 页面",
    "Handle GET request": "处理 GET 请求",
    "Handle POST request": "处理 POST 请求",
    "Execute pipeline from stdlib web server": "从标准库网页服务执行流水线",
    "Pipeline failed in stdlib web server": "标准库网页服务中流水线执行失败",
    "Start stdlib HTTP server for framework web demo": "启动标准库 HTTP 演示服务",
    "All downstream modules consume the same AudioSample contract, so data layer is decoupled from model/evaluator details.": "下游统一消费 AudioSample 契约，数据层与模型/评测层解耦。",
    "Registry pattern removes if/else in pipeline and makes adding new adapters a one-line registration change.": "注册表模式消除流水线中的 if/else，新模型只需一行注册。",
    "Pipeline never imports concrete model adapters directly. It depends only on registry + BaseASREngine contract.": "流水线不直接依赖具体模型实现，只依赖注册表和 BaseASREngine 契约。",
    "Evaluator never checks model type. Any new backend works as long as it implements BaseASREngine.transcribe.": "评测器不关心模型类型，只要实现 BaseASREngine.transcribe 就可接入。",
    "Because every backend returns InferenceResult, evaluator can stay unchanged when swapping models.": "所有后端统一返回 InferenceResult，替换模型时评测器无需改动。",
    "Aggregation is independent of model backend because evaluator emits a stable SampleEvaluation contract.": "聚合层与模型后端无关，因为评测器输出稳定的 SampleEvaluation 契约。",
    "Web layer only triggers pipeline entrypoint, so UI and evaluation engine stay cleanly separated.": "网页层只调用流水线入口，UI 与评测引擎保持清晰分离。",
    "Web page only calls run_pipeline, so orchestration logic has one single implementation for CLI and Web.": "网页仅调用 run_pipeline，CLI 和 Web 共用唯一编排实现。",
}


def _to_cn(text: str) -> str:
    return _CN_MAP.get(text, text)


def _fit_tail(text: str, width: int) -> str:
    if len(text) <= width:
        return text.ljust(width)
    return "…" + text[-(width - 1):]


def _fit_head(text: str, width: int) -> str:
    if len(text) <= width:
        return text.ljust(width)
    return text[: width - 1] + "…"


def _color_text(value: str, color: str) -> str:
    if not _USE_COLOR:
        return value
    return f"{color}{value}{_RESET}"


def _file_color(filename: str) -> str:
    digest = hashlib.md5(filename.encode("utf-8")).hexdigest()
    idx = int(digest[:2], 16) % len(_FILE_COLOR_POOL)
    return _FILE_COLOR_POOL[idx]


def _emit(kind: str, message: str, detail: str, code_ref: str) -> None:
    caller = inspect.currentframe().f_back.f_back
    info = inspect.getframeinfo(caller)
    filename = os.path.relpath(info.filename, start=os.getcwd())
    loc = f"{filename}:{info.lineno}"
    func = info.function

    time_col = datetime.now().strftime("%H:%M:%S")
    kind_col = _fit_head(kind, 4)
    loc_col = _fit_tail(loc, 48)
    func_col = _fit_head(func, 22)

    kind_view = _color_text(kind_col, _TYPE_COLOR.get(kind, ""))
    loc_view = _color_text(loc_col, _file_color(filename))
    func_view = _color_text(func_col, _file_color(filename))

    line = f"[{time_col}] [{kind_view}] [{loc_view}] [{func_view}] 作用: {_to_cn(message)}"
    if detail:
        line += f" | 细节: {detail}"
    if code_ref:
        line += f" | 对应代码: {code_ref}"
    print(line)


def trace_step(purpose: str, detail: str = "") -> None:
    _emit("流程", purpose, detail, "")


def trace_benefit(benefit: str, code_ref: str) -> None:
    _emit("收益", benefit, "", code_ref)
