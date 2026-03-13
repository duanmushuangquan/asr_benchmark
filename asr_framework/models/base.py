"""Base interface for all ASR model adapters."""

from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict

from asr_benchmark.core.contracts import AudioSample
from asr_benchmark.core.contracts import DecodeOptions
from asr_benchmark.core.contracts import InferenceResult
from asr_benchmark.core.contracts import ModelCapabilities


class BaseASREngine(ABC):
    """Abstract interface implemented by concrete ASR backends."""

    def __init__(self, config: Dict[str, Any]):
        """Store backend-specific model config."""
        self.config = config

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Return a stable model identifier used in reports."""
        raise NotImplementedError

    @abstractmethod
    def capabilities(self) -> ModelCapabilities:
        """Return backend capabilities for evaluator/report logic."""
        raise NotImplementedError

    def load(self) -> None:
        """Initialize model resources. Override when required."""

    @abstractmethod
    def transcribe(self, sample: AudioSample, options: DecodeOptions) -> InferenceResult:
        """Run one-sample inference and return normalized result contract."""
        raise NotImplementedError

    def close(self) -> None:
        """Release model resources. Override when required."""
