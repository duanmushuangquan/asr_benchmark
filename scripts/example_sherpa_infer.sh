# copy from https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/online-decode-files.py
# 1.0 查找项目根目录
set -euo pipefail
# 根据当前脚本， 查找项目目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

echo "[1] 测试 funasr 模型"
cd ${PROJECT_ROOT}
python examples/models/funasr/example_funasr_onnx_SeacoParaformer.py

echo "[2] 测试 sherpa 模型"

MODEL_PATH=${PROJECT_ROOT}/models/sherpa/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20

cd ${PROJECT_ROOT}/examples/models/sherpa
# without hotword
# python example_sherpa_onnx_online_from_transducer.py \
#     --tokens=${MODEL_PATH}/tokens.txt \
#     --encoder=${MODEL_PATH}/encoder-epoch-99-avg-1.int8.onnx \
#     --decoder=${MODEL_PATH}/decoder-epoch-99-avg-1.onnx \
#     --joiner=${MODEL_PATH}/joiner-epoch-99-avg-1.int8.onnx \
#     --num-threads=1 \
#     --decoding-method=greedy_search \
#     --provider=cpu \
#     ${MODEL_PATH}/test_wavs/0.wav \
#     /home/gjt/data/workspace/asr_benchmark/assets/test01_hotword.wav

# with hotword
# python example_sherpa_onnx_online_from_transducer.py \
#     --tokens=${MODEL_PATH}/tokens.txt \
#     --encoder=${MODEL_PATH}/encoder-epoch-99-avg-1.int8.onnx \
#     --decoder=${MODEL_PATH}/decoder-epoch-99-avg-1.onnx \
#     --joiner=${MODEL_PATH}/joiner-epoch-99-avg-1.int8.onnx \
#     --num-threads=1 \
#     --decoding-method=modified_beam_search \
#     --max-active-paths=4 \
#     --lm="" \
#     --lm-scale=0.1 \
#     --lodr-fst="" \
#     --lodr-scale=0.1 \
#     --provider=cpu \
#     --hotwords-file=/home/gjt/data/workspace/asr_benchmark/assets/hotwords.txt \
#     --hotwords-score=1.5 \
#     --modeling-unit="" \
#     --bpe-vocab="" \
#     --blank-penalty=0.0 \
#     ${MODEL_PATH}/test_wavs/0.wav \
#     ${PROJECT_ROOT}/assets/test01_hotword.wav

# only support hotword, not support lm and lodr
echo "[2-1] 测试 sherpa 模型的热词"
python example_sherpa_onnx_online_from_transducer.py \
    --tokens=${MODEL_PATH}/tokens.txt \
    --encoder=${MODEL_PATH}/encoder-epoch-99-avg-1.int8.onnx \
    --decoder=${MODEL_PATH}/decoder-epoch-99-avg-1.onnx \
    --joiner=${MODEL_PATH}/joiner-epoch-99-avg-1.int8.onnx \
    --num-threads=1 \
    --decoding-method=modified_beam_search \
    --max-active-paths=4 \
    --provider=cpu \
    --hotwords-file=${PROJECT_ROOT}/assets/sherpa_hotword.txt \
    --hotwords-score=1.5 \
    --modeling-unit=cjkchar+bpe \
    --bpe-vocab=${MODEL_PATH}/bpe.vocab \
    --blank-penalty=0.0 \
    ${MODEL_PATH}/test_wavs/0.wav \
    ${PROJECT_ROOT}/assets/test01_hotword.wav

# only support hotword, not support lm and lodr
# 实际上已经可以不用hotword了， 而且decoding-method使用greedy_search 比modified_beam_search更快。 切换后RTF从0.042 降到0.035 (4090)  0.137 降到0.119 (CPU)  
echo "[2-2] 测试 sherpa 模型的fst功能"
python example_sherpa_onnx_online_from_transducer.py \
    --tokens=${MODEL_PATH}/tokens.txt \
    --encoder=${MODEL_PATH}/encoder-epoch-99-avg-1.int8.onnx \
    --decoder=${MODEL_PATH}/decoder-epoch-99-avg-1.onnx \
    --joiner=${MODEL_PATH}/joiner-epoch-99-avg-1.int8.onnx \
    --num-threads=1 \
    --decoding-method=greedy_search \
    --max-active-paths=4 \
    --provider=cpu \
    --modeling-unit=cjkchar+bpe \
    --bpe-vocab=${MODEL_PATH}/bpe.vocab \
    --blank-penalty=0.0 \
    --hr-lexicon=${PROJECT_ROOT}/assets/lexicon.txt \
    --hr-rule-fsts=${PROJECT_ROOT}/assets/replace.fst \
    ${MODEL_PATH}/test_wavs/0.wav \
    ${PROJECT_ROOT}/assets/test01_hotword.wav
# 说明
# --decoding-method  modified_beam_search 支持热词，greedy_search 不支持热词

echo "[3] 测试 sherpa 模型的最小案例"
python examples/models/sherpa/example_sherpa_onnx_online_from_transducer_simple.py