import hashlib
import re
import time

from asr_framework.contracts import (
    AudioSample,
    DecodeOptions,
    InferenceResult,
    ModelCapabilities,
    RuntimeStats,
)
from asr_framework.debug_trace import trace_benefit, trace_step
from asr_framework.models.base import BaseASREngine


class FunasrMockEngine(BaseASREngine):
    """Mock of FunASR offline ONNX adapter with realistic config/runtime shape."""

    @property
    def model_id(self) -> str:
        trace_step("暴露稳定 model_id 供报告和对比使用")
        return str(self.config.get("model_id", "funasr_mock"))

    def capabilities(self) -> ModelCapabilities:
        trace_step("暴露模型能力给评测和报告层")
        return ModelCapabilities(
            streaming=False,
            supports_hotwords=True,
            supports_first_token_latency=False,
        )

    def load(self) -> None:
        trace_step("模拟 FunASR 离线模型加载")
        model_dir = self.config.get("model_dir", self.config.get("path", "funasr/paraformer-large"))
        batch_size = int(self.config.get("batch_size", 1))
        quantize = bool(self.config.get("quantize", False))
        device = str(self.config.get("device", "cpu"))
        print(
            "[模型加载][FunASR] "
            f"model_id={self.model_id} model_dir={model_dir} "
            f"device={device} batch_size={batch_size} quantize={quantize}"
        )

    def transcribe(self, sample: AudioSample, options: DecodeOptions) -> InferenceResult:
        trace_step("模拟 FunASR 离线一次整句推理", f"sample_id={sample.sample_id}")
        print(
            "[模型推理][FunASR] "
            f"sample_id={sample.sample_id} audio_path={sample.audio_path} "
            f"language={options.language} hotwords={options.hotwords}"
        )
        start = time.time()
        prediction = self._simulate_decode(sample.reference_text, sample.sample_id, options.hotwords)
        total_latency, decode_latency = self._estimate_latency(sample.reference_text, sample.sample_id)
        end = time.time()
        overhead = max(0.0, round(end - start, 4))

        runtime = RuntimeStats(
            total_latency_sec=round(total_latency + overhead, 4),
            decode_latency_sec=round(decode_latency + overhead * 0.7, 4),
            first_token_latency_sec=None,
            extra={
                "backend": "funasr_onnx_offline_mock",
                "stream_mode": "offline_whole_utterance",
            },
        )
        trace_benefit(
            "离线后端只需实现统一 transcribe()，评测与聚合逻辑不感知 FunASR 细节。",
            "asr_framework/models/base.py:23",
        )
        return InferenceResult(text=prediction, runtime=runtime, raw_output={"mock": "funasr"})

    def close(self) -> None:
        trace_step("模拟 FunASR 模型资源释放")
        print(f"[模型释放][FunASR] model_id={self.model_id}")

    @staticmethod
    def _simulate_decode(reference_text: str, sample_id: str, hotwords) -> str:
        trace_step("生成贴近离线 ASR 风格的可复现输出", f"sample_id={sample_id}")
        text = (reference_text or "").strip()
        if not text:
            return ""

        digest = hashlib.md5(sample_id.encode("utf-8")).hexdigest()
        bucket = int(digest[:2], 16) % 4

        if " " in text:
            tokens = text.split()
            if bucket == 1 and len(tokens) > 2:
                tokens = tokens[:-1]
            elif bucket == 2 and len(tokens) > 1:
                tokens[1] = tokens[1] + "x"
            elif bucket == 3 and hotwords:
                tokens.append(str(hotwords[0]))
            return " ".join(tokens)

        normalized = re.sub(r"[，。！？,.!?]", "", text)
        if bucket == 1 and len(normalized) > 3:
            normalized = normalized[:-1]
        elif bucket == 2 and len(normalized) > 2:
            normalized = normalized[:1] + "误" + normalized[2:]
        elif bucket == 3 and hotwords:
            normalized = normalized + str(hotwords[0])
        return normalized

    def _estimate_latency(self, reference_text: str, sample_id: str):
        trace_step("估算离线推理时延分布（总时延/解码时延）")
        batch_size = int(self.config.get("batch_size", 1))
        base = 0.05 + len(reference_text) * 0.0028 + batch_size * 0.004
        digest = hashlib.md5((sample_id + "lat").encode("utf-8")).hexdigest()
        jitter = (int(digest[:2], 16) % 9) * 0.001
        total = round(base + jitter, 4)
        decode = round(total * 0.86, 4)
        return total, decode
