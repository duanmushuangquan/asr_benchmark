import statistics
from typing import Dict, List, Sequence

from asr_framework.contracts import SampleEvaluation
from asr_framework.debug_trace import trace_benefit, trace_step


def _levenshtein_distance(ref: Sequence[str], hyp: Sequence[str], debug: bool = False) -> int:
    trace_step("Compute edit distance core for WER/CER", f"ref_len={len(ref)} hyp_len={len(hyp)}")
    n = len(ref)
    m = len(hyp)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    if debug:
        print(f"[DEBUG][Operators] 编辑距离结果={dp[n][m]} ref={list(ref)} hyp={list(hyp)}")
    return dp[n][m]


def calculate_wer(reference: str, hypothesis: str, debug: bool = False) -> float:
    trace_step("Calculate WER", f"reference={reference} hypothesis={hypothesis}")
    ref_tokens = reference.split() if " " in reference.strip() else list(reference.replace(" ", ""))
    hyp_tokens = hypothesis.split() if " " in hypothesis.strip() else list(hypothesis.replace(" ", ""))
    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0
    if debug:
        print(f"[DEBUG][Operators] WER分词 ref_tokens={ref_tokens} hyp_tokens={hyp_tokens}")
    dist = _levenshtein_distance(ref_tokens, hyp_tokens, debug=debug)
    value = float(dist / len(ref_tokens))
    trace_step("WER calculation complete", f"wer={value}")
    return value


def calculate_cer(reference: str, hypothesis: str, debug: bool = False) -> float:
    trace_step("Calculate CER", f"reference={reference} hypothesis={hypothesis}")
    ref_chars = list(reference.replace(" ", ""))
    hyp_chars = list(hypothesis.replace(" ", ""))
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    if debug:
        print(f"[DEBUG][Operators] CER字符 ref_chars={ref_chars} hyp_chars={hyp_chars}")
    dist = _levenshtein_distance(ref_chars, hyp_chars, debug=debug)
    value = float(dist / len(ref_chars))
    trace_step("CER calculation complete", f"cer={value}")
    return value


def calculate_hotword_metrics(
    hotwords: List[str],
    reference: str,
    hypothesis: str,
    debug: bool = False,
) -> Dict[str, float]:
    trace_step("Calculate hotword metrics", f"hotwords={hotwords}")
    if not hotwords:
        return {}
    total_ref = 0
    total_hyp = 0
    correct = 0
    for hw in hotwords:
        r = reference.count(hw)
        h = hypothesis.count(hw)
        total_ref += r
        total_hyp += h
        correct += min(r, h)
    recall = 1.0 if total_ref == 0 else float(correct / total_ref)
    precision = 1.0 if total_hyp == 0 else float(correct / total_hyp)
    false_trigger = float(max(0, total_hyp - total_ref))
    output = {
        "hotword_recall": recall,
        "hotword_precision": precision,
        "hotword_false_trigger": false_trigger,
    }
    if debug:
        print(
            "[DEBUG][Operators] 热词统计 "
            f"total_ref={total_ref} total_hyp={total_hyp} correct={correct} metrics={output}"
        )
    trace_step("Hotword metrics complete", f"metrics={output}")
    return output


def aggregate_results(results: List[SampleEvaluation], debug: bool = False) -> Dict[str, float]:
    trace_step("Aggregate sample-level metrics into run summary", f"results_count={len(results)}")
    if not results:
        return {
            "wer": 0.0,
            "cer": 0.0,
            "avg_latency_sec": 0.0,
            "p50_latency_sec": 0.0,
            "p90_latency_sec": 0.0,
        }

    wers = [r.metrics.get("wer", 1.0) for r in results]
    cers = [r.metrics.get("cer", 1.0) for r in results]
    latencies = [r.runtime.total_latency_sec for r in results]

    summary = {
        "wer": float(statistics.mean(wers)),
        "cer": float(statistics.mean(cers)),
        "avg_latency_sec": float(statistics.mean(latencies)),
        "p50_latency_sec": _percentile(latencies, 50, debug=debug),
        "p90_latency_sec": _percentile(latencies, 90, debug=debug),
    }
    if debug:
        print(
            "[DEBUG][Operators] 聚合输入 "
            f"wers={wers} cers={cers} latencies={latencies}"
        )
    trace_benefit(
        "Aggregation is independent of model backend because evaluator emits a stable SampleEvaluation contract.",
        "asr_framework/contracts.py:45",
    )
    trace_step("Aggregation complete", f"summary={summary}")
    return summary


def _percentile(values: List[float], percentile: int, debug: bool = False) -> float:
    trace_step("Compute percentile helper", f"percentile={percentile} values_count={len(values)}")
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (percentile / 100.0) * (len(sorted_values) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    value = float(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight)
    if debug:
        print(
            "[DEBUG][Operators] 分位数计算 "
            f"sorted={sorted_values} lower={lower} upper={upper} weight={weight} value={value}"
        )
    trace_step("Percentile helper complete", f"value={value}")
    return value
