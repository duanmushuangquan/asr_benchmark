from typing import List

from asr_framework.contracts import DecodeOptions, SampleEvaluation
from asr_framework.debug_trace import trace_benefit, trace_step
from asr_framework.models.base import BaseASREngine
from asr_framework.normalizer import TextNormalizer
from asr_framework.operators import calculate_cer, calculate_hotword_metrics, calculate_wer


class Evaluator:
    def __init__(self, engine: BaseASREngine, normalizer: TextNormalizer, debug: bool = False):
        trace_step("Initialize evaluator with pluggable model engine and normalizer")
        self.engine = engine
        self.normalizer = normalizer
        self.debug = debug

    def evaluate(self, samples, options: DecodeOptions) -> List[SampleEvaluation]:
        trace_step("Run sample-by-sample evaluation loop", f"samples={len(samples)}")
        results: List[SampleEvaluation] = []
        for sample in samples:
            trace_step("Evaluate one sample", f"sample_id={sample.sample_id}")
            effective_hotwords = self._resolve_hotwords(sample, options)
            effective_options = DecodeOptions(
                language=options.language,
                hotwords=effective_hotwords,
                extra=options.extra,
            )
            inference = self.engine.transcribe(sample, effective_options)
            reference_norm = self.normalizer.normalize(sample.reference_text)
            prediction_norm = self.normalizer.normalize(inference.text)
            if self.debug:
                print(
                    "[DEBUG][Evaluator] "
                    f"sample_id={sample.sample_id} "
                    f"gt_raw={sample.reference_text} pred_raw={inference.text} "
                    f"gt_norm={reference_norm} pred_norm={prediction_norm} "
                    f"hotwords={effective_hotwords}"
                )

            metrics = {
                "wer": calculate_wer(reference_norm, prediction_norm, debug=self.debug),
                "cer": calculate_cer(reference_norm, prediction_norm, debug=self.debug),
            }
            metrics.update(
                calculate_hotword_metrics(
                    effective_hotwords,
                    reference_norm,
                    prediction_norm,
                    debug=self.debug,
                )
            )

            results.append(
                SampleEvaluation(
                    sample_id=sample.sample_id,
                    scenario=sample.scenario,
                    reference_raw=sample.reference_text,
                    prediction_raw=inference.text,
                    reference_norm=reference_norm,
                    prediction_norm=prediction_norm,
                    metrics=metrics,
                    runtime=inference.runtime,
                    metadata=sample.metadata,
                )
            )
        trace_benefit(
            "Evaluator never checks model type. Any new backend works as long as it implements BaseASREngine.transcribe.",
            "asr_framework/models/base.py:23",
        )
        trace_step("Evaluation loop complete", f"generated_results={len(results)}")
        return results

    @staticmethod
    def _resolve_hotwords(sample, options: DecodeOptions) -> List[str]:
        raw = sample.get("hotword")
        if raw is None:
            return list(options.hotwords)
        if isinstance(raw, list):
            return [str(x).strip() for x in raw if str(x).strip() and str(x).strip().lower() != "no"]
        value = str(raw).strip()
        if not value or value.lower() == "no":
            return []
        if "," in value:
            return [seg.strip() for seg in value.split(",") if seg.strip()]
        return [value]
