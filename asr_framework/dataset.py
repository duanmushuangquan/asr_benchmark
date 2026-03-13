import json
from typing import List

from asr_framework.contracts import AudioSample
from asr_framework.debug_trace import trace_benefit, trace_step


def load_samples_from_jsonl(metadata_path: str) -> List[AudioSample]:
    trace_step("Load dataset samples from JSONL metadata", f"metadata_path={metadata_path}")
    samples: List[AudioSample] = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            if not line.strip():
                continue
            trace_step("Parse one metadata line into AudioSample", f"line_index={idx}")
            item = json.loads(line)
            sample_id = str(item.get("sample_id", f"sample_{idx:06d}"))
            audio_path = str(item.get("audio_path", ""))
            reference_text = str(item.get("text", ""))
            reserved = {"sample_id", "audio_path", "text"}
            metadata = {k: v for k, v in item.items() if k not in reserved}
            samples.append(
                AudioSample(
                    sample_id=sample_id,
                    audio_path=audio_path,
                    reference_text=reference_text,
                    metadata=metadata,
                )
            )
    trace_benefit(
        "All downstream modules consume the same AudioSample contract, so data layer is decoupled from model/evaluator details.",
        "asr_framework/contracts.py:5",
    )
    trace_step("Dataset loading complete", f"samples={len(samples)}")
    return samples
