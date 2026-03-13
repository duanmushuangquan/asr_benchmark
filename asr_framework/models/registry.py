from typing import Any, Dict, Type

from asr_framework.debug_trace import trace_benefit, trace_step
from asr_framework.models.base import BaseASREngine
from asr_framework.models.funasr_mock import FunasrMockEngine
from asr_framework.models.sherpa_mock import SherpaMockEngine


MODEL_REGISTRY: Dict[str, Type[BaseASREngine]] = {
    "funasr_mock": FunasrMockEngine,
    "sherpa_mock": SherpaMockEngine,
}


def create_engine(model_config: Dict[str, Any]) -> BaseASREngine:
    trace_step("Create model engine from registry", f"model_config={model_config}")
    name = str(model_config.get("name", "funasr_mock"))
    engine_cls = MODEL_REGISTRY.get(name)
    if engine_cls is None:
        raise ValueError(f"Unknown model adapter: {name}. Available: {sorted(MODEL_REGISTRY)}")
    trace_benefit(
        "Registry pattern removes if/else in pipeline and makes adding new adapters a one-line registration change.",
        "asr_framework/models/registry.py:8",
    )
    return engine_cls(model_config)
