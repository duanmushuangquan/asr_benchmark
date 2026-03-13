"""
默认使用funasr-onnx进行推理，不支持hotword功能
"""
# from runtime.python.onnxruntime.funasr_onnx.paraformer_bin import Paraformer
import warnings

# Suppress noisy warning triggered by `jieba` importing `pkg_resources` under newer setuptools.
warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

from funasr_onnx import Paraformer

# 测试音频路径
wav_path = [
    "./models/funasr/nextbig/example/asr_example.wav",
    "./assets/test01_hotword.wav"
    ]

# 模型路径
model_dir="./models/funasr/nextbig"
model = Paraformer(
    model_dir, 
    batch_size=1, 
    quantize=True
    )


result = model(wav_path)
print(result)
