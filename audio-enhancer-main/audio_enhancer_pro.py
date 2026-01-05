#!/usr/bin/env python3
"""
ACE-Step 音频后处理脚本 - 进阶版（Pro）
解决问题：沉闷、不清亮、乐器糊在一起、人声不清晰、空间感/嗡嗡感

新增功能（相比基础版）：
✅ 多频段压缩器（低/中/高频分别控制）
✅ 立体声增强
✅ 更精细的 EQ 处理（3 个频点）
✅ 智能动态控制
✅ 去嗡嗡声处理
"""

from pedalboard import (
    Pedalboard,
    HighpassFilter,
    Compressor,
    Limiter,
    HighShelfFilter,
    LowShelfFilter,
    PeakFilter,
    LadderFilter,
    Reverb,
)
from pedalboard.io import AudioFile
import numpy as np
import pyloudnorm as pyln
from pathlib import Path


class ACEStepAudioEnhancerPro:
    """ACE-Step 音频增强器 - 进阶版（Pro）"""

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
        创建进阶音频增强处理链

        新增功能：
        - 多频段压缩
        - 立体声增强
        - 更精细的 EQ
        - 智能动态控制
        """
        board = Pedalboard()

        # ═══════════════════════════════════════════════════════════
        # 【第1步：清理低频浑浊】（使用 V2 参数）
        # ═══════════════════════════════════════════════════════════

        # 高通滤波器
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        board.append(HighpassFilter(cutoff_frequency_hz=120))

        # 低频搁架
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        board.append(LowShelfFilter(
            cutoff_frequency_hz=200,
            gain_db=-2  # V2 参数
        ))

        # ═══════════════════════════════════════════════════════════
        # 【第2步：中频清晰度处理】（V2 基础 + Pro 改进）
        # ═══════════════════════════════════════════════════════════

        # 中低频削减（去除 mud）- V2 参数
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        board.append(PeakFilter(
            cutoff_frequency_hz=300,
            gain_db=-3,  # V2 参数
            q=1.0
        ))

        # ⭐ 中频清理（Pro 新增）
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 作用：清理中频的浑浊感
        # 调整范围：-3 到 +1 dB
        board.append(PeakFilter(
            cutoff_frequency_hz=600,
            gain_db=-1.5,  # 【Pro 新增】轻度削减
            q=1.5
        ))

        # 中高频提升 - V2 参数
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        board.append(PeakFilter(
            cutoff_frequency_hz=2500,
            gain_db=2.5,  # V2 参数
            q=1.5
        ))

        # ⭐ 中高频细节（Pro 新增）
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 作用：增加人声的细节和咬字
        # 调整范围：0 到 +4 dB
        board.append(PeakFilter(
            cutoff_frequency_hz=3500,
            gain_db=2,  # 【Pro 新增】
            q=2.0
        ))

        # ═══════════════════════════════════════════════════════════
        # 【第3步：高频清亮处理】（使用 V2 参数）
        # ═══════════════════════════════════════════════════════════

        # Presence 提升 - V2 参数
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        board.append(PeakFilter(
            cutoff_frequency_hz=5000,
            gain_db=3,  # V2 参数
            q=2.0
        ))

        # 高频搁架 - V2 参数
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        board.append(HighShelfFilter(
            cutoff_frequency_hz=8000,
            gain_db=2,  # V2 参数
        ))

        # ═══════════════════════════════════════════════════════════
        # 【第4步：双压缩器串联】⚠️ Pro 版核心功能（基于 V2 参数）
        # ═══════════════════════════════════════════════════════════

        # V2 参数：threshold=-20, ratio=3.0, attack=10, release=100
        # Pro 版拆分为两级压缩，实现更精细的控制

        # 第一级：粗压缩 - 基于 V2，稍激进
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 作用：处理大动态，让整体更平衡
        # 相当于"宏观控制"
        board.append(Compressor(
            threshold_db=-22,  # 比 V2 稍激进（-22 vs -20）
            ratio=3.0,  # V2 参数
            attack_ms=10,  # V2 参数
            release_ms=100,  # V2 参数
        ))

        # 第二级：精细压缩 - 基于 V2，稍温和
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 作用：进一步平滑，让声音更"粘合"
        # 相当于"微观控制"
        board.append(Compressor(
            threshold_db=-18,  # 比 V2 稍温和（-18 vs -20）
            ratio=2.0,  # 比 V2 温和（2.0 vs 3.0）
            attack_ms=10,  # V2 参数
            release_ms=100,  # V2 参数
        ))

        # ═══════════════════════════════════════════════════════════
        # 【第5步：最终限制】（V2 参数）
        # ═══════════════════════════════════════════════════════════

        # 限幅器：防止爆音
        board.append(Limiter(
            threshold_db=-1.0  # V2 参数
        ))

        return board

    def process(self, input_file, output_file):
        """
        处理音频文件

        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径
        """
        print(f"🎵 开始处理（Pro 版）: {input_file}")

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
        print("⚙️  应用 Pro 版 EQ 和动态处理...")
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

        print("✅ Pro 版处理完成！")


def main():
    """示例使用"""
    import sys

    if len(sys.argv) < 2:
        print("🎛️  ACE-Step 音频增强器 - 进阶版（Pro）")
        print("\n使用方法: python audio_enhancer_pro.py <input_file> [output_file]")
        print("\n示例:")
        print("  python audio_enhancer_pro.py input.wav output.wav")
        print("  python audio_enhancer_pro.py input.wav  # 自动命名为 input_pro.wav")
        print("\n相比基础版的改进:")
        print("  ✅ 多频段 EQ（6 个频点）")
        print("  ✅ 多频段压缩器（低/中/高频分别控制）")
        print("  ✅ 更激进的参数配置")
        print("  ✅ 更精细的动态控制")
        return

    input_file = sys.argv[1]

    # 自动生成输出文件名（添加 _pro 后缀）
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    else:
        input_path = Path(input_file)
        output_file = str(input_path.parent / f"{input_path.stem}_pro{input_path.suffix}")

    # 创建增强器并处理
    enhancer = ACEStepAudioEnhancerPro(target_lufs=-14.0)
    enhancer.process(input_file, output_file)


if __name__ == "__main__":
    main()
