#!/usr/bin/env bash
# 1.0 查找项目根目录
set -euo pipefail
# 根据当前脚本， 查找项目目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# 2.0 FunASR
# 2.1 FunASR - nextbig/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch-onnx
pip install modelscope
mkdir -p models/funasr/nextbig
# 不支持热词
modelscope download \
  --model nextbig/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch-onnx \
  --local_dir ./models/funasr/nextbig

# 2.2 FunASR - marxyz/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-onnx-quant
mkdir -p models/funasr/marxyz
# 支持热词
modelscope download \
  --model marxyz/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-onnx-quant \
  --local_dir ./models/funasr/marxyz

### 2.2.1 download 热词相关
cd "${PROJECT_ROOT}/assets"
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/hr-files/lexicon.txt
cd "${PROJECT_ROOT}"

# 3.0 sherpa
SHERPA_BASE_DIR="${PROJECT_ROOT}/models/sherpa"
SHERPA_MODEL_NAME="sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"
SHERPA_ARCHIVE="${SHERPA_MODEL_NAME}.tar.bz2"
SHERPA_MODEL_DIR="${SHERPA_BASE_DIR}/${SHERPA_MODEL_NAME}"
SHERPA_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/${SHERPA_ARCHIVE}"

mkdir -p "${SHERPA_BASE_DIR}"

if [ -d "${SHERPA_MODEL_DIR}" ]; then
  echo "[INFO] sherpa model already exists: ${SHERPA_MODEL_DIR}"
  echo "[INFO] skip download."
else
  cd "${SHERPA_BASE_DIR}"

  if [ ! -f "${SHERPA_ARCHIVE}" ]; then
    echo "[INFO] downloading sherpa archive: ${SHERPA_ARCHIVE}"
    wget -O "${SHERPA_ARCHIVE}" "${SHERPA_URL}"
  else
    echo "[INFO] sherpa archive already exists: ${SHERPA_ARCHIVE}"
    echo "[INFO] skip wget."
  fi

  echo "[INFO] extracting sherpa archive..."
  tar xvf "${SHERPA_ARCHIVE}"
fi

cd "${PROJECT_ROOT}"