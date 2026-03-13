"""FunASR ONNX SeacoParaformer adapter."""

from __future__ import annotations

import importlib
import logging
import time
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence

from asr_benchmark.core.contracts import AudioSample
from asr_benchmark.core.contracts import DecodeOptions
from asr_benchmark.core.contracts import InferenceResult
from asr_benchmark.core.contracts import ModelCapabilities
from asr_benchmark.core.contracts import RuntimeStats
from asr_benchmark.models.base import BaseASREngine

logger = logging.getLogger(__name__)


class FunasrOnnxSeacoParaformerEngine(BaseASREngine):
    """Adapter for `funasr_onnx.SeacoParaformer` offline inference."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._model: Optional[Any] = None
        self._default_hotwords = self._load_hotwords_from_yaml(
            self.config.get("hotwords_yaml")
        )

    @property
    def model_id(self) -> str:
        return str(self.config.get("model_id", "funasr_onnx_seaco_paraformer"))

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            streaming=False,
            supports_hotwords=True,
            supports_first_token_latency=False,
        )

    def load(self) -> None:
        """Load FunASR ONNX model resources."""
        model_dir = Path(str(self.config.get("model_dir", ""))).expanduser()
        if not model_dir.exists():
            raise FileNotFoundError(f"FunASR model directory does not exist: {model_dir}")

        seaco_cls = self._resolve_seaco_paraformer_class()
        kwargs = {
            "model_dir": str(model_dir),
            "batch_size": int(self.config.get("batch_size", 1)),
            "device_id": int(self.config.get("device_id", -1)),
            "plot_timestamp_to": str(self.config.get("plot_timestamp_to", "")),
            "quantize": bool(self.config.get("quantize", True)),
            "intra_op_num_threads": int(self.config.get("intra_op_num_threads", 4)),
        }

        logger.info("Loading FunASR model with config: %s", kwargs)
        self._model = seaco_cls(**kwargs)

    def transcribe(self, sample: AudioSample, options: DecodeOptions) -> InferenceResult:
        """Run one-sample offline transcription."""
        if self._model is None:
            raise RuntimeError("FunASR model is not loaded. Call `load()` first.")

        audio_path = Path(sample.audio_path).expanduser()
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

        hotwords = self._resolve_hotwords(options.hotwords)
        hotword_text = " ".join(hotwords)

        logger.debug(
            "FunASR transcribe sample_id=%s audio_path=%s hotwords=%s",
            sample.sample_id,
            str(audio_path),
            hotwords,
        )
        start = time.perf_counter()
        raw_output = self._model([str(audio_path)], hotword_text)
        elapsed = time.perf_counter() - start
        text = self._extract_text(raw_output)

        return InferenceResult(
            text=text,
            runtime=RuntimeStats(
                total_latency_sec=elapsed,
                decode_latency_sec=elapsed,
            ),
            raw_output=raw_output,
        )

    def close(self) -> None:
        """Release model resources."""
        self._model = None

    def _resolve_hotwords(self, request_hotwords: Sequence[str]) -> List[str]:
        if request_hotwords:
            return [str(value).strip() for value in request_hotwords if str(value).strip()]
        return list(self._default_hotwords)

    @staticmethod
    def _extract_text(raw_output: Any) -> str:
        if isinstance(raw_output, list) and raw_output:
            first_item = raw_output[0]
            if isinstance(first_item, dict) and "preds" in first_item:
                return str(first_item.get("preds", ""))
            return str(first_item)
        return str(raw_output or "")

    @staticmethod
    def _resolve_seaco_paraformer_class() -> Any:
        try:
            module = importlib.import_module("funasr_onnx")
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency `funasr_onnx`. Install it before loading FunASR engine."
            ) from exc

        seaco_cls = getattr(module, "SeacoParaformer", None)
        if seaco_cls is None:
            raise RuntimeError("`funasr_onnx.SeacoParaformer` is not available.")
        return seaco_cls

    @staticmethod
    def _load_hotwords_from_yaml(path_value: Any) -> List[str]:
        if not path_value:
            return []

        yaml_path = Path(str(path_value)).expanduser()
        if not yaml_path.exists():
            logger.warning("Hotword yaml not found: %s", str(yaml_path))
            return []

        try:
            import yaml  # type: ignore
        except ImportError:
            logger.warning("PyYAML is not installed, skip loading hotwords yaml.")
            return []

        with yaml_path.open("r", encoding="utf-8") as file_obj:
            data = yaml.safe_load(file_obj) or {}
        values = data.get("hotwords", [])
        if not isinstance(values, list):
            logger.warning("Invalid hotwords yaml format at %s", str(yaml_path))
            return []

        hotwords = [str(value).strip() for value in values if str(value).strip()]
        logger.info("Loaded %d hotwords from %s", len(hotwords), str(yaml_path))
        return hotwords
