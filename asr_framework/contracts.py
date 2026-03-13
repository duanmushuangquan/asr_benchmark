from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AudioSample:
    sample_id: str
    audio_path: str
    reference_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.metadata.get(key, default)

    @property
    def scenario(self) -> str:
        return str(self.metadata.get("task_scene", self.metadata.get("scenario", "default")))

    @property
    def language(self) -> str:
        return str(self.metadata.get("language", "unknown"))


@dataclass
class DecodeOptions:
    language: Optional[str] = None
    hotwords: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RuntimeStats:
    total_latency_sec: float
    decode_latency_sec: float
    first_token_latency_sec: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceResult:
    text: str
    runtime: RuntimeStats
    raw_output: Any = None
    warnings: List[str] = field(default_factory=list)


@dataclass
class ModelCapabilities:
    streaming: bool = False
    supports_hotwords: bool = False
    supports_first_token_latency: bool = False


@dataclass
class SampleEvaluation:
    sample_id: str
    scenario: str
    reference_raw: str
    prediction_raw: str
    reference_norm: str
    prediction_norm: str
    metrics: Dict[str, float]
    runtime: RuntimeStats
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunSummary:
    run_name: str
    dataset_id: str
    model_id: str
    total_samples: int
    succeeded_samples: int
    failed_samples: int
    aggregate_metrics: Dict[str, float]
    capabilities: Dict[str, Any]
