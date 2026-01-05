import os
import numpy as np
import torch
from pedalboard import Pedalboard, Compressor, HighpassFilter, HighShelfFilter
from pedalboard.io import AudioFile
import pyloudnorm as pyln
import folder_paths
import tempfile
from datetime import datetime


class AudioNormalizeNode:
    """
    ComfyUI音频处理节点
    功能：去浑浊、提亮、轻压缩、统一音量（-14 LUFS）
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "output_format": (["wav", "mp3"], {
                    "default": "wav"
                }),
                "target_loudness": ("FLOAT", {
                    "default": -14.0,
                    "min": -30.0,
                    "max": -5.0,
                    "step": 0.5,
                    "display": "number"
                }),
            },
        }
    
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "file_path")
    FUNCTION = "process_audio"
    CATEGORY = "audio/processing"
    OUTPUT_NODE = True
    
    def process_audio(self, audio, output_format="wav", target_loudness=-14.0):
        """
        处理音频：去浑浊、提亮、压缩、音量归一化
        """
        # 获取音频数据
        waveform = audio["waveform"]  # shape: [batch, channels, samples]
        sample_rate = audio["sample_rate"]
        
        # 转换为numpy数组处理（取第一个batch）
        if isinstance(waveform, torch.Tensor):
            audio_data = waveform[0].cpu().numpy()  # [channels, samples]
        else:
            audio_data = waveform[0]
        
        # 转置为 [samples, channels] 格式（pedalboard需要）
        audio_data = audio_data.T
        
        # 创建音频处理链
        board = Pedalboard([
            # 去浑浊：高通滤波器，去除低频浑浊声音
            HighpassFilter(cutoff_frequency_hz=80.0),
            
            # 提亮：高频增强
            HighShelfFilter(cutoff_frequency_hz=4000.0, gain_db=2.0, q=0.7),
            
            # 轻压缩：防止爆音，让音量更均衡
            Compressor(
                threshold_db=-20.0,  # 压缩阈值
                ratio=3.0,           # 压缩比 3:1 (轻压缩)
                attack_ms=10.0,      # 快速启动
                release_ms=100.0     # 平滑释放
            ),
        ])
        
        # 应用效果处理
        processed_audio = board(audio_data, sample_rate)
        
        # 音量归一化到目标LUFS
        meter = pyln.Meter(sample_rate)
        
        # 测量当前响度
        try:
            loudness = meter.integrated_loudness(processed_audio)
            # 归一化到目标响度
            processed_audio = pyln.normalize.loudness(
                processed_audio, 
                loudness, 
                target_loudness
            )
        except Exception as e:
            print(f"警告：音量归一化失败 - {e}，使用峰值归一化")
            # 如果LUFS归一化失败，使用峰值归一化
            peak = np.abs(processed_audio).max()
            if peak > 0:
                processed_audio = processed_audio * (0.95 / peak)
        
        # 确保不超过[-1, 1]范围
        processed_audio = np.clip(processed_audio, -1.0, 1.0)
        
        # 保存文件
        output_dir = folder_paths.get_output_directory()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"audio_normalized_{timestamp}.{output_format}"
        filepath = os.path.join(output_dir, filename)
        
        # 保存音频文件
        if output_format == "mp3":
            # MP3格式（使用较高的比特率以保持质量）
            with AudioFile(filepath, 'w', sample_rate, processed_audio.shape[1]) as f:
                f.write(processed_audio.T)  # 转回 [channels, samples]
        else:
            # WAV格式（无损）
            with AudioFile(filepath, 'w', sample_rate, processed_audio.shape[1]) as f:
                f.write(processed_audio.T)  # 转回 [channels, samples]
        
        print(f"音频已保存至: {filepath}")
        
        # 转换回PyTorch格式返回
        processed_tensor = torch.from_numpy(processed_audio.T).unsqueeze(0)  # [1, channels, samples]
        
        output_audio = {
            "waveform": processed_tensor,
            "sample_rate": sample_rate
        }
        
        # 返回结果，包含音频播放器UI
        return {
            "ui": {
                "audio": [{
                    "filename": filename,
                    "subfolder": "",
                    "type": "output",
                    "format": f"audio/{output_format}"
                }]
            },
            "result": (output_audio, filepath)
        }


# ComfyUI节点注册
NODE_CLASS_MAPPINGS = {
    "AudioNormalizeNode": AudioNormalizeNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioNormalizeNode": "Audio Normalize (去浑浊+压缩+音量归一化)"
}
