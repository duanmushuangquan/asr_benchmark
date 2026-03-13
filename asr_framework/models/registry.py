"""Registry for ASR model adapters."""

from typing import Any
from typing import Dict
from typing import List
from typing import Type

from asr_benchmark.models.base import BaseASREngine
from asr_benchmark.models.funasr_onnx_seaco_paraformer import (
    FunasrOnnxSeacoParaformerEngine,
)
from asr_benchmark.models.sherpa_onnx_online_transducer import (
    SherpaOnnxOnlineTransducerEngine,
)

MODEL_REGISTRY: Dict[str, Type[BaseASREngine]] = {
    "funasr_onnx_seaco_paraformer": FunasrOnnxSeacoParaformerEngine,
    "sherpa_onnx_online_transducer": SherpaOnnxOnlineTransducerEngine,
}


def register_model(name: str, engine_cls: Type[BaseASREngine]) -> None:
    """Register one model adapter class with a unique name."""
    model_name = str(name).strip()
    if not model_name:
        raise ValueError("Model name must be a non-empty string.")
    if model_name in MODEL_REGISTRY:
        raise ValueError(f"Model adapter already registered: {model_name}")
    MODEL_REGISTRY[model_name] = engine_cls


def create_engine(model_config: Dict[str, Any]) -> BaseASREngine:
    """Create one model adapter from config using the registry."""
    name = str(model_config.get("name", "")).strip()
    if not name:
        raise ValueError("Model config must include non-empty key 'name'.")

    engine_cls = MODEL_REGISTRY.get(name)
    if engine_cls is None:
        raise ValueError(
            f"Unknown model adapter: {name}. Available: {available_models()}"
        )
    return engine_cls(model_config)


def available_models() -> List[str]:
    """Return registered model adapter names."""
    return sorted(MODEL_REGISTRY)
