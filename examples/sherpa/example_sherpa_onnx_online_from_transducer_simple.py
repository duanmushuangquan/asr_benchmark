#!/usr/bin/env python3

import time
import wave
from pathlib import Path
from typing import Tuple

import numpy as np
import sherpa_onnx

# ===================== Config =====================
PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = PROJECT_ROOT / "models/sherpa/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"

TOKENS = MODEL_PATH / "tokens.txt"
ENCODER = MODEL_PATH / "encoder-epoch-99-avg-1.int8.onnx"
DECODER = MODEL_PATH / "decoder-epoch-99-avg-1.onnx"
JOINER = MODEL_PATH / "joiner-epoch-99-avg-1.int8.onnx"

NUM_THREADS = 1
PROVIDER = "cpu"
DECODING_METHOD = "greedy_search"
MAX_ACTIVE_PATHS = 4
MODELING_UNIT = "cjkchar+bpe"
BPE_VOCAB = MODEL_PATH / "bpe.vocab"
BLANK_PENALTY = 0.0
HR_LEXICON = PROJECT_ROOT / "assets/lexicon.txt"
HR_RULE_FSTS = PROJECT_ROOT / "assets/replace.fst"

SOUND_FILES = [
    MODEL_PATH / "test_wavs/0.wav",
    PROJECT_ROOT / "assets/test01_hotword.wav",
]
# ==================================================


def assert_file_exists(path: Path) -> None:
    assert path.is_file(), f"{path} does not exist"


def read_wave(wave_filename: Path) -> Tuple[np.ndarray, int]:
    with wave.open(str(wave_filename), "rb") as f:
        assert f.getnchannels() == 1, f.getnchannels()
        assert f.getsampwidth() == 2, f.getsampwidth()
        num_samples = f.getnframes()
        samples = f.readframes(num_samples)
        samples_int16 = np.frombuffer(samples, dtype=np.int16)
        samples_float32 = samples_int16.astype(np.float32) / 32768.0
        return samples_float32, f.getframerate()


def build_recognizer() -> sherpa_onnx.OnlineRecognizer:
    required_files = [
        TOKENS,
        ENCODER,
        DECODER,
        JOINER,
        BPE_VOCAB,
        HR_LEXICON,
        HR_RULE_FSTS,
    ]
    for file_path in required_files:
        assert_file_exists(file_path)

    return sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=str(TOKENS),
        encoder=str(ENCODER),
        decoder=str(DECODER),
        joiner=str(JOINER),
        num_threads=NUM_THREADS,
        provider=PROVIDER,
        sample_rate=16000,
        feature_dim=80,
        decoding_method=DECODING_METHOD,
        max_active_paths=MAX_ACTIVE_PATHS,
        modeling_unit=MODELING_UNIT,
        bpe_vocab=str(BPE_VOCAB),
        blank_penalty=BLANK_PENALTY,
        hr_rule_fsts=str(HR_RULE_FSTS),
        hr_lexicon=str(HR_LEXICON),
    )


def main() -> None:
    recognizer = build_recognizer()

    print("Started!")
    start_time = time.time()

    streams = []
    total_duration = 0.0

    for wave_filename in SOUND_FILES:
        assert_file_exists(wave_filename)
        samples, sample_rate = read_wave(wave_filename)
        duration = len(samples) / sample_rate
        total_duration += duration

        stream = recognizer.create_stream()
        stream.accept_waveform(sample_rate, samples)

        tail_paddings = np.zeros(int(0.66 * sample_rate), dtype=np.float32)
        stream.accept_waveform(sample_rate, tail_paddings)
        stream.input_finished()
        streams.append(stream)

    while True:
        ready_streams = [stream for stream in streams if recognizer.is_ready(stream)]
        if not ready_streams:
            break
        recognizer.decode_streams(ready_streams)

    results = [recognizer.get_result(stream) for stream in streams]
    elapsed_seconds = time.time() - start_time

    print("Done!")
    for wave_filename, result in zip(SOUND_FILES, results):
        print(f"{wave_filename}\n{result}")
        print("-" * 10)

    rtf = elapsed_seconds / total_duration
    print(f"num_threads: {NUM_THREADS}")
    print(f"provider: {PROVIDER}")
    print(f"decoding_method: {DECODING_METHOD}")
    print(f"Wave duration: {total_duration:.3f} s")
    print(f"Elapsed time: {elapsed_seconds:.3f} s")
    print(
        f"Real time factor (RTF): {elapsed_seconds:.3f}/{total_duration:.3f} = {rtf:.3f}"
    )


if __name__ == "__main__":
    main()
