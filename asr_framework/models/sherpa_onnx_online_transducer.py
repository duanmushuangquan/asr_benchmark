"""Sherpa ONNX streaming transducer adapter."""

from __future__ import annotations

import importlib
import logging
import time
import wave
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

import numpy as np

from asr_benchmark.core.contracts import AudioSample
from asr_benchmark.core.contracts import DecodeOptions
from asr_benchmark.core.contracts import InferenceResult
from asr_benchmark.core.contracts import ModelCapabilities
from asr_benchmark.core.contracts import RuntimeStats
from asr_benchmark.models.base import BaseASREngine

logger = logging.getLogger(__name__)


class SherpaOnnxOnlineTransducerEngine(BaseASREngine):
    """Adapter for `sherpa_onnx.OnlineRecognizer.from_transducer`."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._recognizer: Optional[Any] = None

    @property
    def model_id(self) -> str:
        return str(self.config.get("model_id", "sherpa_onnx_online_transducer"))

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            streaming=True,
            supports_hotwords=True,
            supports_first_token_latency=False,
        )

    def load(self) -> None:
        """Load Sherpa streaming recognizer."""
        required = {
            "tokens": self._required_file_path("tokens"),
            "encoder": self._required_file_path("encoder"),
            "decoder": self._required_file_path("decoder"),
            "joiner": self._required_file_path("joiner"),
        }

        kwargs = {
            "tokens": str(required["tokens"]),
            "encoder": str(required["encoder"]),
            "decoder": str(required["decoder"]),
            "joiner": str(required["joiner"]),
            "num_threads": int(self.config.get("num_threads", 1)),
            "provider": str(self.config.get("provider", "cpu")),
            "sample_rate": int(self.config.get("sample_rate", 16000)),
            "feature_dim": int(self.config.get("feature_dim", 80)),
            "decoding_method": str(self.config.get("decoding_method", "greedy_search")),
            "max_active_paths": int(self.config.get("max_active_paths", 4)),
            "modeling_unit": str(self.config.get("modeling_unit", "")),
            "blank_penalty": float(self.config.get("blank_penalty", 0.0)),
            "bpe_vocab": str(self._optional_file_path("bpe_vocab") or ""),
            "lm": str(self._optional_file_path("lm") or ""),
            "lm_scale": float(self.config.get("lm_scale", 0.1)),
            "lodr_fst": str(self._optional_file_path("lodr_fst") or ""),
            "lodr_scale": float(self.config.get("lodr_scale", -0.1)),
            "hotwords_file": str(self._optional_file_path("hotwords_file") or ""),
            "hotwords_score": float(self.config.get("hotwords_score", 1.5)),
            "hr_lexicon": str(self._optional_file_path("hr_lexicon") or ""),
            "hr_rule_fsts": str(self._optional_file_path("hr_rule_fsts") or ""),
        }

        logger.info("Loading Sherpa recognizer with config: %s", kwargs)
        sherpa_module = self._resolve_sherpa_module()
        self._recognizer = sherpa_module.OnlineRecognizer.from_transducer(**kwargs)

    def transcribe(self, sample: AudioSample, options: DecodeOptions) -> InferenceResult:
        """Run one streaming inference with a single WAV file."""
        if self._recognizer is None:
            raise RuntimeError("Sherpa recognizer is not loaded. Call `load()` first.")

        if options.hotwords:
            logger.warning(
                "DecodeOptions.hotwords is not applied at runtime for Sherpa. "
                "Use `hotwords_file` in model config."
            )

        audio_path = Path(sample.audio_path).expanduser()
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

        samples, sample_rate = read_wave(audio_path)
        stream = self._recognizer.create_stream()
        start_time = time.perf_counter()
        stream.accept_waveform(sample_rate, samples)
        tail_paddings = np.zeros(int(0.66 * sample_rate), dtype=np.float32)
        stream.accept_waveform(sample_rate, tail_paddings)
        stream.input_finished()
        self._decode_stream(stream)
        result_text = str(self._recognizer.get_result(stream))
        elapsed = time.perf_counter() - start_time

        audio_duration = 0.0
        if sample_rate > 0:
            audio_duration = float(len(samples) / sample_rate)
        rtf = float(elapsed / audio_duration) if audio_duration > 0 else 0.0

        return InferenceResult(
            text=result_text,
            runtime=RuntimeStats(
                total_latency_sec=elapsed,
                decode_latency_sec=elapsed,
                extra={
                    "audio_duration_sec": audio_duration,
                    "rtf": rtf,
                    "provider": str(self.config.get("provider", "cpu")),
                    "decoding_method": str(
                        self.config.get("decoding_method", "greedy_search")
                    ),
                },
            ),
            raw_output={"result_text": result_text},
        )

    def close(self) -> None:
        """Release recognizer resources."""
        self._recognizer = None

    def _decode_stream(self, stream: Any) -> None:
        decode_stream = getattr(self._recognizer, "decode_stream", None)
        decode_streams = getattr(self._recognizer, "decode_streams", None)

        if callable(decode_stream):
            while self._recognizer.is_ready(stream):
                decode_stream(stream)
            return

        if callable(decode_streams):
            while self._recognizer.is_ready(stream):
                decode_streams([stream])
            return

        raise RuntimeError("Unsupported Sherpa recognizer decode API.")

    def _required_file_path(self, key: str) -> Path:
        value = self.config.get(key)
        if not value:
            raise ValueError(f"Missing required config key: {key}")

        path = Path(str(value)).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Required file does not exist for {key}: {path}")
        return path

    def _optional_file_path(self, key: str) -> Optional[Path]:
        value = self.config.get(key)
        if not value:
            return None

        path = Path(str(value)).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Configured file does not exist for {key}: {path}")
        return path

    @staticmethod
    def _resolve_sherpa_module() -> Any:
        try:
            return importlib.import_module("sherpa_onnx")
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency `sherpa_onnx`. Install it before loading Sherpa engine."
            ) from exc


def read_wave(wave_path: Path) -> Tuple[np.ndarray, int]:
    """Read a WAV file into float32 samples and sample rate."""
    with wave.open(str(wave_path), "rb") as stream:
        channels = stream.getnchannels()
        sample_width = stream.getsampwidth()
        if channels != 1:
            raise ValueError(f"Expected mono WAV. Got channels={channels}")
        if sample_width != 2:
            raise ValueError(f"Expected 16-bit WAV. Got sample_width={sample_width}")

        num_samples = stream.getnframes()
        data = stream.readframes(num_samples)
        samples_int16 = np.frombuffer(data, dtype=np.int16)
        samples_float32 = samples_int16.astype(np.float32) / 32768.0
        return samples_float32, int(stream.getframerate())
