import re
from dataclasses import dataclass

from asr_framework.debug_trace import trace_step


@dataclass
class NormalizerConfig:
    to_lower: bool = True
    remove_punctuation: bool = True
    strip_spaces: bool = True
    debug: bool = False


class TextNormalizer:
    def __init__(self, config: NormalizerConfig):
        trace_step("Initialize text normalizer with explicit config", f"config={config}")
        self.config = config
        self._punct_pattern = re.compile(r"[^\w\s]")

    def normalize(self, text: str) -> str:
        trace_step("Normalize text for fair metric comparison", f"raw_text={text}")
        value = text or ""
        if self.config.to_lower:
            if self.config.debug:
                print(f"[DEBUG][Normalizer] lower前: {value}")
            value = value.lower()
            if self.config.debug:
                print(f"[DEBUG][Normalizer] lower后: {value}")
        if self.config.remove_punctuation:
            if self.config.debug:
                print(f"[DEBUG][Normalizer] 去标点前: {value}")
            value = self._punct_pattern.sub("", value)
            if self.config.debug:
                print(f"[DEBUG][Normalizer] 去标点后: {value}")
        if self.config.strip_spaces:
            if self.config.debug:
                print(f"[DEBUG][Normalizer] 压缩空格前: {value}")
            value = re.sub(r"\s+", " ", value).strip()
            if self.config.debug:
                print(f"[DEBUG][Normalizer] 压缩空格后: {value}")
        trace_step("Text normalization complete", f"normalized_text={value}")
        return value
