import os
import numpy as np
import torch
from pedalboard import Pedalboard, Compressor, HighpassFilter, HighShelfFilter, LowShelfFilter, PeakFilter
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


class AudioProcessNode:
    """
    ComfyUI音频处理节点（仅处理不保存）
    功能：去浑浊、提亮、轻压缩、统一音量（-14 LUFS）
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "target_loudness": ("FLOAT", {
                    "default": -14.0,
                    "min": -30.0,
                    "max": -5.0,
                    "step": 0.5,
                    "display": "number"
                }),
            },
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "process_audio"
    CATEGORY = "audio/processing"
    
    def process_audio(self, audio, target_loudness=-14.0):
        """
        处理音频：去浑浊、提亮、压缩、音量归一化（不保存文件）
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
        
        # 转换回PyTorch格式返回
        processed_tensor = torch.from_numpy(processed_audio.T).unsqueeze(0)  # [1, channels, samples]
        
        output_audio = {
            "waveform": processed_tensor,
            "sample_rate": sample_rate
        }
        
        return (output_audio,)


class VocalEnhanceNode:
    """
    ComfyUI人声增强节点
    功能：人声频段增强、去齿音、去低频噪音、压缩、音量归一化
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "target_loudness": ("FLOAT", {
                    "default": -16.0,
                    "min": -30.0,
                    "max": -5.0,
                    "step": 0.5,
                    "display": "number"
                }),
                "de_esser_strength": ("FLOAT", {
                    "default": 3.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.5,
                    "display": "slider"
                }),
            },
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "enhance_vocal"
    CATEGORY = "audio/processing"
    
    def enhance_vocal(self, audio, target_loudness=-16.0, de_esser_strength=3.0):
        """
        人声增强处理
        """
        # 获取音频数据
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        
        if isinstance(waveform, torch.Tensor):
            audio_data = waveform[0].cpu().numpy()
        else:
            audio_data = waveform[0]
        
        audio_data = audio_data.T
        
        # 人声处理链
        board = Pedalboard([
            # 去低频噪音和隆隆声（人声通常从100Hz开始）
            HighpassFilter(cutoff_frequency_hz=100.0),
            
            # 温暖人声基频（200-500Hz）
            PeakFilter(cutoff_frequency_hz=250.0, gain_db=2.0, q=1.0),
            
            # 增强人声清晰度（2-4kHz）
            PeakFilter(cutoff_frequency_hz=3000.0, gain_db=3.0, q=1.5),
            
            # 去齿音（降低6-8kHz的尖锐声）
            PeakFilter(cutoff_frequency_hz=7000.0, gain_db=-de_esser_strength, q=2.0),
            
            # 轻微提亮高频（增加空气感）
            HighShelfFilter(cutoff_frequency_hz=10000.0, gain_db=1.5, q=0.7),
            
            # 人声压缩（让音量更稳定）
            Compressor(
                threshold_db=-18.0,
                ratio=4.0,           # 较强压缩，让人声更稳定
                attack_ms=5.0,       # 快速启动
                release_ms=50.0      # 快速释放
            ),
        ])
        
        processed_audio = board(audio_data, sample_rate)
        
        # 音量归一化
        meter = pyln.Meter(sample_rate)
        try:
            loudness = meter.integrated_loudness(processed_audio)
            processed_audio = pyln.normalize.loudness(
                processed_audio, loudness, target_loudness
            )
        except Exception as e:
            print(f"警告：音量归一化失败 - {e}")
            peak = np.abs(processed_audio).max()
            if peak > 0:
                processed_audio = processed_audio * (0.95 / peak)
        
        processed_audio = np.clip(processed_audio, -1.0, 1.0)
        
        processed_tensor = torch.from_numpy(processed_audio.T).unsqueeze(0)
        output_audio = {
            "waveform": processed_tensor,
            "sample_rate": sample_rate
        }
        
        return (output_audio,)


class InstrumentalEnhanceNode:
    """
    ComfyUI伴奏增强节点
    功能：低音增强、高频亮度、中频清晰、压缩、音量归一化
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "target_loudness": ("FLOAT", {
                    "default": -14.0,
                    "min": -30.0,
                    "max": -5.0,
                    "step": 0.5,
                    "display": "number"
                }),
                "bass_boost": ("FLOAT", {
                    "default": 3.0,
                    "min": 0.0,
                    "max": 8.0,
                    "step": 0.5,
                    "display": "slider"
                }),
                "treble_boost": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.0,
                    "max": 6.0,
                    "step": 0.5,
                    "display": "slider"
                }),
            },
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "enhance_instrumental"
    CATEGORY = "audio/processing"
    
    def enhance_instrumental(self, audio, target_loudness=-14.0, bass_boost=3.0, treble_boost=2.0):
        """
        伴奏增强处理
        """
        # 获取音频数据
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        
        if isinstance(waveform, torch.Tensor):
            audio_data = waveform[0].cpu().numpy()
        else:
            audio_data = waveform[0]
        
        audio_data = audio_data.T
        
        # 伴奏处理链
        board = Pedalboard([
            # 去除极低频噪音
            HighpassFilter(cutoff_frequency_hz=30.0),
            
            # 增强低音（60-150Hz）- 让鼓和贝斯更有力量
            LowShelfFilter(cutoff_frequency_hz=100.0, gain_db=bass_boost, q=0.7),
            
            # 增强低中频（200-400Hz）- 让乐器更饱满
            PeakFilter(cutoff_frequency_hz=300.0, gain_db=1.5, q=1.0),
            
            # 清晰中频（1-3kHz）- 让乐器分离度更好
            PeakFilter(cutoff_frequency_hz=2000.0, gain_db=2.0, q=1.2),
            
            # 增强高频（6kHz以上）- 增加亮度和空气感
            HighShelfFilter(cutoff_frequency_hz=6000.0, gain_db=treble_boost, q=0.7),
            
            # 轻压缩（保持动态范围）
            Compressor(
                threshold_db=-22.0,
                ratio=2.5,           # 较轻压缩，保留伴奏动态
                attack_ms=15.0,
                release_ms=150.0
            ),
        ])
        
        processed_audio = board(audio_data, sample_rate)
        
        # 音量归一化
        meter = pyln.Meter(sample_rate)
        try:
            loudness = meter.integrated_loudness(processed_audio)
            processed_audio = pyln.normalize.loudness(
                processed_audio, loudness, target_loudness
            )
        except Exception as e:
            print(f"警告：音量归一化失败 - {e}")
            peak = np.abs(processed_audio).max()
            if peak > 0:
                processed_audio = processed_audio * (0.95 / peak)
        
        processed_audio = np.clip(processed_audio, -1.0, 1.0)
        
        processed_tensor = torch.from_numpy(processed_audio.T).unsqueeze(0)
        output_audio = {
            "waveform": processed_tensor,
            "sample_rate": sample_rate
        }
        
        return (output_audio,)


class LoudnessNormalizeNode:
    """
    ComfyUI响度归一化节点
    功能：仅调整音量到目标LUFS，不做任何其他处理
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "target_loudness": ("FLOAT", {
                    "default": -14.0,
                    "min": -30.0,
                    "max": -5.0,
                    "step": 0.1,
                    "display": "number"
                }),
                "use_peak_limiting": ("BOOLEAN", {
                    "default": True,
                    "label_on": "启用",
                    "label_off": "禁用"
                }),
            },
        }
    
    RETURN_TYPES = ("AUDIO", "FLOAT")
    RETURN_NAMES = ("audio", "original_loudness")
    FUNCTION = "normalize_loudness"
    CATEGORY = "audio/processing"
    
    def normalize_loudness(self, audio, target_loudness=-14.0, use_peak_limiting=True):
        """
        仅调整响度到目标LUFS值
        """
        # 获取音频数据
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        
        if isinstance(waveform, torch.Tensor):
            audio_data = waveform[0].cpu().numpy()
        else:
            audio_data = waveform[0]
        
        # 转置为 [samples, channels] 格式
        audio_data = audio_data.T
        
        # 音量归一化到目标LUFS
        meter = pyln.Meter(sample_rate)
        
        try:
            # 测量当前响度
            original_loudness = meter.integrated_loudness(audio_data)
            print(f"原始响度: {original_loudness:.2f} LUFS → 目标响度: {target_loudness:.2f} LUFS")
            
            # 归一化到目标响度
            processed_audio = pyln.normalize.loudness(
                audio_data, 
                original_loudness, 
                target_loudness
            )
            
        except Exception as e:
            print(f"警告：LUFS归一化失败 - {e}，使用峰值归一化")
            original_loudness = 0.0
            # 如果LUFS归一化失败，使用峰值归一化
            peak = np.abs(audio_data).max()
            if peak > 0:
                # 根据目标响度估算增益（粗略）
                target_peak = 10 ** (target_loudness / 20)
                processed_audio = audio_data * (target_peak / peak)
            else:
                processed_audio = audio_data
        
        # 峰值限制（防止削波）
        if use_peak_limiting:
            processed_audio = np.clip(processed_audio, -1.0, 1.0)
        
        # 转换回PyTorch格式
        processed_tensor = torch.from_numpy(processed_audio.T).unsqueeze(0)
        output_audio = {
            "waveform": processed_tensor,
            "sample_rate": sample_rate
        }
        
        return (output_audio, float(original_loudness))


class AudioMixerNode:
    """
    ComfyUI音频混音节点
    功能：混合最多6个音频轨道
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_1": ("AUDIO",),
            },
            "optional": {
                "audio_2": ("AUDIO",),
                "audio_3": ("AUDIO",),
                "audio_4": ("AUDIO",),
                "audio_5": ("AUDIO",),
                "audio_6": ("AUDIO",),
                "volume_1": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "display": "slider"
                }),
                "volume_2": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "display": "slider"
                }),
                "volume_3": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "display": "slider"
                }),
                "volume_4": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "display": "slider"
                }),
                "volume_5": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "display": "slider"
                }),
                "volume_6": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "display": "slider"
                }),
                "normalize_output": ("BOOLEAN", {
                    "default": True,
                    "label_on": "启用",
                    "label_off": "禁用"
                }),
            },
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("mixed_audio",)
    FUNCTION = "mix_audio"
    CATEGORY = "audio/processing"
    
    def mix_audio(self, audio_1, audio_2=None, audio_3=None, audio_4=None, 
                  audio_5=None, audio_6=None, volume_1=1.0, volume_2=1.0, 
                  volume_3=1.0, volume_4=1.0, volume_5=1.0, volume_6=1.0,
                  normalize_output=True):
        """
        混合多个音频轨道
        """
        # 收集所有输入的音频和音量
        audio_tracks = []
        volumes = []
        
        for i, (audio, volume) in enumerate([
            (audio_1, volume_1), (audio_2, volume_2), (audio_3, volume_3),
            (audio_4, volume_4), (audio_5, volume_5), (audio_6, volume_6)
        ], 1):
            if audio is not None:
                audio_tracks.append(audio)
                volumes.append(volume)
                print(f"轨道 {i}: 音量 {volume:.2f}")
        
        if not audio_tracks:
            raise ValueError("至少需要一个音频输入")
        
        # 获取采样率（使用第一个音频的采样率）
        sample_rate = audio_tracks[0]["sample_rate"]
        
        # 检查所有音频的采样率是否一致
        for i, audio in enumerate(audio_tracks, 1):
            if audio["sample_rate"] != sample_rate:
                print(f"警告：轨道 {i} 的采样率 ({audio['sample_rate']}Hz) 与轨道1不同 ({sample_rate}Hz)")
        
        # 转换所有音频为numpy数组
        audio_arrays = []
        for audio in audio_tracks:
            waveform = audio["waveform"]
            if isinstance(waveform, torch.Tensor):
                audio_data = waveform[0].cpu().numpy()  # [channels, samples]
            else:
                audio_data = waveform[0]
            audio_arrays.append(audio_data)
        
        # 找到最长的音频长度和最大声道数
        max_channels = max(arr.shape[0] for arr in audio_arrays)
        max_length = max(arr.shape[1] for arr in audio_arrays)
        
        print(f"混音信息：{len(audio_tracks)}个轨道，{max_channels}声道，{max_length}采样点")
        
        # 创建混合后的音频数组
        mixed_audio = np.zeros((max_channels, max_length), dtype=np.float32)
        
        # 混合所有轨道
        for audio_data, volume in zip(audio_arrays, volumes):
            channels, length = audio_data.shape
            
            # 应用音量
            audio_with_volume = audio_data * volume
            
            # 如果声道数不足，复制到所有声道
            if channels == 1 and max_channels > 1:
                for ch in range(max_channels):
                    mixed_audio[ch, :length] += audio_with_volume[0, :length]
            else:
                # 混合到对应的声道
                for ch in range(min(channels, max_channels)):
                    mixed_audio[ch, :length] += audio_with_volume[ch, :length]
        
        # 归一化输出（防止削波）
        if normalize_output:
            peak = np.abs(mixed_audio).max()
            if peak > 1.0:
                mixed_audio = mixed_audio / peak
                print(f"混音后峰值 {peak:.2f} > 1.0，已归一化")
            elif peak > 0.95:
                print(f"混音后峰值 {peak:.2f}，接近满载")
        
        # 转换回PyTorch格式
        mixed_tensor = torch.from_numpy(mixed_audio).unsqueeze(0)
        output_audio = {
            "waveform": mixed_tensor,
            "sample_rate": sample_rate
        }
        
        return (output_audio,)


# ComfyUI节点注册
NODE_CLASS_MAPPINGS = {
    "AudioNormalizeNode": AudioNormalizeNode,
    "AudioProcessNode": AudioProcessNode,
    "VocalEnhanceNode": VocalEnhanceNode,
    "InstrumentalEnhanceNode": InstrumentalEnhanceNode,
    "LoudnessNormalizeNode": LoudnessNormalizeNode,
    "AudioMixerNode": AudioMixerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioNormalizeNode": "Audio Normalize & Save (处理+保存)",
    "AudioProcessNode": "Audio Process (仅处理)",
    "VocalEnhanceNode": "Vocal Enhance (人声增强)",
    "InstrumentalEnhanceNode": "Instrumental Enhance (伴奏增强)",
    "LoudnessNormalizeNode": "Loudness Normalize (响度归一化)",
    "AudioMixerNode": "Audio Mixer (混音器)"
}
