# 创建脚本文件

#!/bin/bash
# ffplay_wav.sh - 通过蓝牙耳机播放 WAV 文件

# 检查是否提供了文件名参数
if [ $# -eq 0 ]; then
    echo "错误：请指定要播放的 WAV 文件"
    echo "用法：$0 <wav_file_path>"
    exit 1
fi

# 获取音频文件路径
WAV_FILE="$1"

# 检查文件是否存在
if [ ! -f "$WAV_FILE" ]; then
    echo "错误：文件不存在 - $WAV_FILE"
    exit 1
fi

# 检查文件是否为 WAV 格式（简单检查扩展名）
if [[ ! "$WAV_FILE" =~ \.wav$ ]]; then
    echo "警告：文件可能不是 WAV 格式，建议使用 .wav 文件"
fi

# 设置蓝牙设备作为默认输出并播放
echo "正在通过蓝牙耳机播放：$WAV_FILE"
PULSE_SINK=bluez_sink.64_8F_DB_B6_8C_89.a2dp_sink ffplay "$WAV_FILE" -nodisp -autoexit

# 检查播放结果
if [ $? -eq 0 ]; then
    echo "播放完成"
else
    echo "播放失败"
    exit 1
fi