from abc import ABC, abstractmethod
from typing import Any, Dict

from asr_framework.contracts import AudioSample, DecodeOptions, InferenceResult, ModelCapabilities
from asr_framework.debug_trace import trace_step


class BaseASREngine(ABC):
    def __init__(self, config: Dict[str, Any]):
        trace_step("Initialize base engine with backend-specific config", f"config={config}")
        self.config = config

    @property
    @abstractmethod
    def model_id(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def capabilities(self) -> ModelCapabilities:
        raise NotImplementedError

    def load(self) -> None:
        """Initialize model resources."""
        trace_step("Default no-op load in base class")

    @abstractmethod
    def transcribe(self, sample: AudioSample, options: DecodeOptions) -> InferenceResult:
        raise NotImplementedError

    def close(self) -> None:
        """Release model resources."""
        trace_step("Default no-op close in base class")
