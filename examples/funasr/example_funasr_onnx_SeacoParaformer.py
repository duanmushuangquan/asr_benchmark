import warnings

# Suppress noisy warning triggered by `jieba` importing `pkg_resources` under newer setuptools.
warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

from funasr_onnx import SeacoParaformer
from pathlib import Path
import os
import yaml

model_dir = "./models/funasr/marxyz"
model = SeacoParaformer(
    model_dir=model_dir, 
    batch_size=1,           # 如果wav_path有都跳音频，注意修改这个值，否则只推理第一个音频。
    device_id=-1,           # Orin暂不支持设置为0，需适配。源码把它传给 OrtInferSession(...)。默认 "-1"，通常表示 CPU；非 -1 时一般对应某个 GPU 设备号。 如gpu推理需要安装onnxruntime-gpu
    plot_timestamp_to="",   # 如果非空，会把时间戳结果画图输出到这个目录。源码里在有时间戳输出时会调用 plot_wave_timestamp(...)，最后保存成 timestamp.png
    quantize=True,          # 是否量化，下载的模型只有量化版本的onnx，所以设置 True。
    intra_op_num_threads=4, # ONNX Runtime 的算子内线程数, 默认4。源码直接把它传给 OrtInferSession(..., intra_op_num_threads=...)。
    )

# 音频路径
wav_path = [
    # "./models/funasr/marxyz/example/asr_example.wav",
    "./assets/test01_hotword.wav"
    ]

# 热词配置文件路径
# hotwords = "傅利叶 小傅 小傅小傅 傅利叶智能科技"
# hotwords = ""
hotword_yaml_path = "./configs/hotwords.yaml"
with open(hotword_yaml_path, 'r', encoding='utf-8') as f:
    hw_config = yaml.safe_load(f)
    if hw_config and 'hotwords' in hw_config:
        hotwords = hw_config['hotwords']
        hotwords = " ".join(hotwords)
        print(f"Loaded {len(hotwords)} hotwords from {hotword_yaml_path}")
        
result = model(wav_path, hotwords)

# batch_size=1是的案例
print(result)
# 使用热词
# [{'preds': '欢 迎 大 家 来 到 么 哒 社 区 进 行 体 验', 'timestamp': [[990, 1290], [1290, 1610], [1610, 1830], [1830, 2010], [2010, 2170], [2170, 2430], [2430, 2570], [2570, 2850], [2850, 3050], [3050, 3390], [3390, 3570], [3570, 3910], [3910, 4110], [4110, 4345]], 'raw_tokens': ['欢', '迎', '大', '家', '来', '到', '么', '哒', '社', '区', '进', '行', '体', '验']}]
print(result[0]['preds'])
#  使用热词
# 当 前 音 频 是 用 于 测 试 热 词 的 然 后 傅 利 叶 智 能 科 技 有 限 公 司 富 利 叶 小 傅 小 傅

#  不使用热词
# 当 前 音 频 是 用 于 测 试 热 词 的 然 后 富 利 叶 智 能 科 技 有 限 公 司 富 利 叶 小 富 小 富
