import hashlib
import math
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


class SherpaMockEngine(BaseASREngine):
    """Mock of Sherpa-ONNX streaming adapter with first-token latency simulation."""

    @property
    def model_id(self) -> str:
        trace_step("暴露稳定 model_id 供报告和对比使用")
        return str(self.config.get("model_id", "sherpa_mock"))

    def capabilities(self) -> ModelCapabilities:
        trace_step("暴露模型能力给评测和报告层")
        return ModelCapabilities(
            streaming=True,
            supports_hotwords=True,
            supports_first_token_latency=True,
        )

    def load(self) -> None:
        trace_step("模拟 Sherpa 流式模型加载")
        model_files = self.config.get("model_files", {})
        chunk_seconds = float(self.config.get("chunk_seconds", 0.1))
        device = str(self.config.get("device", "cpu"))
        print(
            "[模型加载][Sherpa] "
            f"model_id={self.model_id} device={device} chunk_seconds={chunk_seconds} "
            f"tokens={model_files.get('tokens', '')}"
        )

    def transcribe(self, sample: AudioSample, options: DecodeOptions) -> InferenceResult:
        trace_step("模拟 Sherpa 流式逐块推理", f"sample_id={sample.sample_id}")
        chunk_seconds = float(self.config.get("chunk_seconds", 0.1))
        audio_seconds = self._estimate_audio_seconds(sample.reference_text)
        chunk_count = max(1, math.ceil(audio_seconds / chunk_seconds))
        print(
            "[模型推理][Sherpa] "
            f"sample_id={sample.sample_id} chunks={chunk_count} "
            f"chunk_seconds={chunk_seconds} hotwords={options.hotwords}"
        )

        start = time.time()
        prediction = self._simulate_streaming_decode(sample.reference_text, sample.sample_id, options.hotwords)
        total_latency, decode_latency, first_token_latency = self._estimate_streaming_latency(
            sample.sample_id,
            chunk_count,
            chunk_seconds,
        )
        end = time.time()
        overhead = max(0.0, round(end - start, 4))

        runtime = RuntimeStats(
            total_latency_sec=round(total_latency + overhead, 4),
            decode_latency_sec=round(decode_latency + overhead * 0.7, 4),
            first_token_latency_sec=round(first_token_latency + overhead * 0.3, 4),
            extra={
                "backend": "sherpa_onnx_streaming_mock",
                "chunk_count": chunk_count,
                "stream_mode": "online_chunk_decode",
            },
        )
        trace_benefit(
            "统一 RuntimeStats 字段后，流式模型的首包时延可直接进入同一套评测与报告体系。",
            "asr_framework/contracts.py:22",
        )
        return InferenceResult(text=prediction, runtime=runtime, raw_output={"mock": "sherpa"})

    def close(self) -> None:
        trace_step("模拟 Sherpa 模型资源释放")
        print(f"[模型释放][Sherpa] model_id={self.model_id}")

    @staticmethod
    def _estimate_audio_seconds(reference_text: str) -> float:
        trace_step("根据文本长度估算音频时长，用于模拟流式分块")
        if not reference_text:
            return 1.0
        if " " in reference_text:
            return max(1.0, len(reference_text.split()) * 0.42)
        return max(1.0, len(reference_text) * 0.09)

    @staticmethod
    def _simulate_streaming_decode(reference_text: str, sample_id: str, hotwords) -> str:
        trace_step("生成贴近流式转写风格的可复现输出", f"sample_id={sample_id}")
        text = (reference_text or "").strip()
        if not text:
            return ""
        digest = hashlib.md5(sample_id.encode("utf-8")).hexdigest()
        bucket = int(digest[:2], 16) % 4

        if " " in text:
            tokens = text.split()
            if bucket == 1 and len(tokens) > 2:
                tokens.pop(1)
            elif bucket == 2 and hotwords:
                tokens.insert(0, str(hotwords[0]))
            elif bucket == 3:
                tokens.append("uh")
            return " ".join(tokens)

        if bucket == 1 and len(text) > 3:
            return text[1:]
        if bucket == 2 and hotwords:
            return str(hotwords[0]) + text
        if bucket == 3:
            return text + "嗯"
        return text

    @staticmethod
    def _estimate_streaming_latency(sample_id: str, chunk_count: int, chunk_seconds: float):
        trace_step("估算流式时延（总时延/解码时延/首包时延）")
        digest = hashlib.md5((sample_id + "stream").encode("utf-8")).hexdigest()
        jitter = (int(digest[:2], 16) % 7) * 0.001
        first_token = round(max(0.04, chunk_seconds * 1.8 + jitter), 4)
        decode = round(chunk_count * chunk_seconds * 0.55 + 0.03 + jitter, 4)
        total = round(decode + 0.03 + chunk_seconds * 0.5, 4)
        return total, decode, first_token
