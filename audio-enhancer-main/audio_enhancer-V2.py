#!/usr/bin/env python3
"""
ACE-Step 音频后处理脚本 - 基础版
解决问题：沉闷、不清亮、乐器糊在一起、人声不清晰
"""

from pedalboard import (
    Pedalboard,
    HighpassFilter,
    Compressor,
    Limiter,
    HighShelfFilter,
    LowShelfFilter,
    PeakFilter,
)
from pedalboard.io import AudioFile
import numpy as np
import pyloudnorm as pyln
from pathlib import Path


class ACEStepAudioEnhancer:
    """ACE-Step 音频增强器 - 基础版"""

    def __init__(self, target_lufs=-14.0):
        """
        初始化增强器

        Args:
            target_lufs: 目标响度值（默认 -14 LUFS，符合流媒体标准）
        """
        self.target_lufs = target_lufs
        self.sample_rate = 44100  # 默认采样率

        # 初始化处理链
        self.board = self._create_enhancement_chain()

    def _create_enhancement_chain(self):
        """
        创建音频增强处理链
        解决：沉闷、不清亮、乐器糊、人声不清晰

        ═══════════════════════════════════════════════════════════════
        参数调整指南 - 解决"声音粘连感"
        ═══════════════════════════════════════════════════════════════
        """
        board = Pedalboard()

        # ═══════════════════════════════════════════════════════════
        # 【第1步：清理低频浑浊】
        # ═══════════════════════════════════════════════════════════

        # 高通滤波器（切除低频）
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 作用：切除指定频率以下的低频，去除浑浊感
        # 影响：过低会失去厚度，过高会变薄
        # 调整范围：80-200 Hz
        #   - 80-100 Hz：轻微清理，保留更多低频
        #   - 120-150 Hz：平衡（当前设置）
        #   - 180-200 Hz：激进清理，声音变薄
        # 方向：如果声音"太薄"→降低到 80-100；如果"浑浊"→提高到 150-200
        board.append(HighpassFilter(cutoff_frequency_hz=120))

        # 低频搁架（整体调整低频）
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 作用：整体提升或削减 200Hz 以下的低频
        # 影响：正值为温暖/厚重，负值为清爽/轻薄
        # 调整范围：-6 到 +3 dB
        #   - -6 到 -4 dB：大幅削减，非常清爽（可能太薄）
        #   - -3 到 -2 dB：适度削减，去除 boxiness（当前设置）
        #   - -1 到 0 dB：轻微调整
        #   +1 到 +3 dB：增加厚度（可能重新浑浊）
        # 方向：如果声音"太薄"→增加到 -1 到 0；如果"浑浊"→减少到 -4 到 -6
        board.append(LowShelfFilter(
            cutoff_frequency_hz=200,
            gain_db=-2  # 【可调整】范围：-6 到 +3
        ))

        # ═══════════════════════════════════════════════════════════
        # 【第2步：中频清晰度处理】⚠️ 对抗"粘连感"的关键
        # ═══════════════════════════════════════════════════════════

        # 中低频削减（去除 mud）
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 作用：削减 250-400Hz 的"泥泞"频段，这是粘连感的主要来源
        # 影响：减少会让声音更干净，但过多会失去人声厚度
        # 调整范围：-6 到 0 dB
        #   - -6 到 -4 dB：大幅削减，非常干净（人声可能变薄）
        #   - -3 到 -2 dB：适度削减（当前设置）
        #   - -1 到 0 dB：轻微削减，保留厚度
        # 方向：如果"粘连感严重"→增加到 -4 到 -6；如果"人声太薄"→减少到 -1 到 0
        board.append(PeakFilter(
            cutoff_frequency_hz=300,  # 【可调整】范围：250-400
            gain_db=-3,  # ⚠️ 【关键参数】范围：-6 到 0
            q=1.0  # 【可调整】带宽：0.7(宽)-3(窄)
        ))

        # 中高频提升（增加清晰度）
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 作用：提升 2-4kHz，增强人声和主乐器的清晰度
        # 影响：增加让声音更靠前、更清晰，但过多会刺耳
        # 调整范围：0 到 +6 dB
        #   - 0 到 +1 dB：轻微提升
        #   - +2 到 +3 dB：适度提升（当前设置）
        #   - +4 到 +6 dB：大幅提升，非常清晰（可能刺耳）
        # 方向：如果"人声/乐器不清晰"→增加到 +4 到 +6；如果"刺耳"→减少到 0 到 +1
        board.append(PeakFilter(
            cutoff_frequency_hz=2500,  # 【可调整】范围：2000-4000
            gain_db=2.5,  # ⚠️ 【关键参数】范围：0 到 +6
            q=1.5  # 【可调整】带宽：0.7(宽)-3(窄)
        ))

        # ═══════════════════════════════════════════════════════════
        # 【第3步：高频清亮处理】⚠️ 解决"沉闷"的关键
        # ═══════════════════════════════════════════════════════════

        # Presence 提升（临场感）
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 作用：提升 4-6kHz，增加临场感和清晰度
        # 影响：让声音更"就在眼前"，但过多会嘶嘶声
        # 调整范围：0 到 +6 dB
        #   - 0 到 +1 dB：轻微提升
        #   - +2 到 +4 dB：适度提升（当前设置）
        #   - +5 到 +6 dB：大幅提升（可能太尖锐）
        # 方向：如果"声音沉闷"→增加到 +4 到 +6；如果"太尖锐"→减少到 0 到 +2
        board.append(PeakFilter(
            cutoff_frequency_hz=5000,  # 【可调整】范围：4000-6000
            gain_db=3,  # ⚠️ 【关键参数】范围：0 到 +6
            q=2.0  # 【可调整】带宽：0.7(宽)-3(窄)
        ))

        # 高频搁架（空气感）
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 作用：整体提升 8kHz 以上的高频，增加空气感和亮度
        # 影响：让声音更敞亮、细腻，但过多会有嘶嘶声
        # 调整范围：0 到 +6 dB
        #   - 0 到 +1 dB：轻微提升
        #   - +2 到 +3 dB：适度提升（当前设置）
        #   - +4 到 +6 dB：大幅提升，非常亮（可能太刺）
        # 方向：如果"声音暗/闷"→增加到 +4 到 +6；如果"太刺/有嘶嘶声"→减少到 0 到 +1
        board.append(HighShelfFilter(
            cutoff_frequency_hz=8000,  # 【可调整】范围：6000-10000
            gain_db=2,  # ⚠️ 【关键参数】范围：0 到 +6
        ))

        # ═══════════════════════════════════════════════════════════
        # 【第4步：动态控制】⚠️ 解决"乐器糊在一起"的关键
        # ═══════════════════════════════════════════════════════════

        # 压缩器（动态控制）
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 作用：压缩动态范围，让乐器更分离、音量更均衡
        # 影响：减少动态范围，让小音量变大音量变小，更"粘合"
        #
        # threshold_db（阈值）：超过这个音量才被压缩
        #   - -10 到 -15 dB：轻度压缩，保留更多动态
        #   - -18 到 -25 dB：中度压缩，平衡（当前：-20）
        #   - -30 到 -40 dB：重度压缩，非常平（可能失去动态）
        #   方向：如果"动态太大、忽大忽小"→降低到 -25 到 -30
        #        如果"失去动态、太平"→提高到 -15 到 -18
        #
        # ratio（压缩比）：超过阈值时，多少 dB 输入变成 1 dB 输出
        #   - 1.5:1 到 2:1：轻度压缩
        #   - 2.5:1 到 4:1：中度压缩（当前：3:1）
        #   - 6:1 到 10:1：重度压缩
        #   方向：如果"还是不平"→增加到 4 到 6；如果"失去动态"→减少到 1.5 到 2
        #
        # attack_ms（启动时间）：超过阈值后多久开始压缩（毫秒）
        #   - 5-10 ms：快速，保留更多 transient
        #   - 15-30 ms：中等，更平滑（当前：10）
        #   - 50+ ms：慢速，更自然但可能漏掉峰值
        #
        # release_ms（释放时间）：回到正常多久（毫秒）
        #   - 50-100 ms：快速，更有 punch（当前：100）
        #   - 150-300 ms：中等，更平滑
        #   - 400+ ms：慢速，可能" pumping"
        board.append(Compressor(
            threshold_db=-20,  # 【可调整】范围：-10 到 -40
            ratio=3.0,  # 【可调整】范围：1.5 到 10
            attack_ms=10,  # 【可调整】范围：5 到 50
            release_ms=100,  # 【可调整】范围：50 到 500
        ))

        # ═══════════════════════════════════════════════════════════
        # 【第5步：限幅保护】防止爆音
        # ═══════════════════════════════════════════════════════════
        # threshold_db：最高音量限制
        #   - -0.5 到 -1.0 dB：安全，留有 headroom（推荐）
        #   - -0.1 到 -0.3 dB：更响，但可能爆音
        board.append(Limiter(
            threshold_db=-1.0  # 【可调整】范围：-0.5 到 -3
        ))

        return board

    def process(self, input_file, output_file):
        """
        处理音频文件

        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径
        """
        print(f"🎵 开始处理: {input_file}")

        # 读取音频文件
        with AudioFile(input_file) as f:
            self.sample_rate = f.samplerate
            audio = f.read(f.frames)
            print(f"   采样率: {self.sample_rate} Hz")
            print(f"   声道数: {audio.shape[0]}")
            print(f"   时长: {f.duration:.2f} 秒")

        # 转换为 pedalboard 格式 (samples, channels)
        if audio.ndim == 1:
            audio = audio.reshape(-1, 1)
        else:
            audio = audio.T

        # ========== 步骤1: 应用效果链 ==========
        print("⚙️  应用 EQ 和动态处理...")
        effected = self.board(audio, self.sample_rate)

        # ========== 步骤2: 响度归一化 ==========
        print(f"🔊 归一化到 {self.target_lufs} LUFS...")

        # pedalboard 返回 (samples, channels)，pyloudnorm 也期望 (samples, channels)
        # 创建响度归一化器
        meter = pyln.Meter(self.sample_rate)  # 创建 BS.1770 响度计
        loudness = meter.integrated_loudness(effected)  # 测量当前响度
        print(f"   原始响度: {loudness:.2f} LUFS")

        # 计算增益
        gain = self.target_lufs - loudness
        print(f"   应用增益: {gain:.2f} dB")
        effected = effected * (10 ** (gain / 20))

        # 最终限幅保护（防止归一化后爆音）
        effected = np.clip(effected, -1.0, 1.0)

        # 转换回 (channels, samples) 格式
        if effected.ndim == 1:
            effected = effected.reshape(1, -1)
        else:
            effected = effected.T

        # ========== 步骤3: 保存文件 ==========
        print(f"💾 保存到: {output_file}")
        with AudioFile(
            output_file,
            'w',
            samplerate=self.sample_rate,
            num_channels=effected.shape[0]
        ) as f:
            f.write(effected)

        print("✅ 处理完成！")


def convert_windows_path(path):
    """将 Windows 路径转换为 WSL 路径"""
    import re

    # 如果路径已经是 WSL 格式，直接返回
    if path.startswith('/mnt/'):
        return path

    # 匹配 C: 模式
    match = re.match(r'^([A-Za-z]):(.*)$', path)
    if match:
        drive = match.group(1).lower()
        rest = match.group(2)

        # 如果没有斜杠，说明反斜杠被 shell 移除了
        # 需要智能还原：C:UsersadminDownloadsacestep_03452_.mp3
        if '/' not in rest and '\\' not in rest:
            # 在小写字母后跟大写字母的地方插入斜杠
            # UsersadminDownloads -> Users/admin/Downloads
            rest = re.sub(r'([a-z])([A-Z])', r'\1/\2', rest)

            # 处理数字后跟大写字母的情况
            rest = re.sub(r'([0-9_])([A-Z])', r'\1/\2', rest)

        # 替换剩余的反斜杠
        rest = rest.replace('\\', '/')

        return f'/mnt/{drive}/{rest}'

    return path


def main():
    """示例使用"""
    import sys

    if len(sys.argv) < 2:
        print("使用方法: python audio_enhancer.py <input_file> [output_file]")
        print("\n示例:")
        print("  python audio_enhancer.py input.wav output.wav")
        print("  python audio_enhancer.py input.wav  # 自动命名为 input_enhanced.wav")
        print("\n支持 Windows 路径（自动转换为 WSL 路径）:")
        print("  python audio_enhancer.py C:\\Users\\admin\\Downloads\\song.mp3")
        return

    input_file = sys.argv[1]
    input_file = convert_windows_path(input_file)

    # 自动生成输出文件名
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
        output_file = convert_windows_path(output_file)
    else:
        input_path = Path(input_file)
        output_file = str(input_path.parent / f"{input_path.stem}_enhanced{input_path.suffix}")

    # 创建增强器并处理
    enhancer = ACEStepAudioEnhancer(target_lufs=-14.0)
    enhancer.process(input_file, output_file)


if __name__ == "__main__":
    main()
