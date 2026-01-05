#!/usr/bin/env python3
"""ACE-Step 输出音频后处理 - Pro（稳定增强版 v2 - 兼容版）

解决点（针对你当前环境的真实报错）：
- 你的 pedalboard 版本里没有 Saturation 类（ImportError）
- 本版本对 Saturation 做兼容：
  - 若 pedalboard 支持 Saturation：用内置 Saturation
  - 若不支持：用 numpy 的 soft saturation（tanh soft clip），效果更温和且可控

设计目标（产品向）：
- 听感：更清晰、更通透、更“成品”，但尽量不刺、不薄、不失真
- 稳定：尽可能降低“某些歌处理后翻车”的概率
- 性能：CPU 0.x~1s 级（不包含可选 stem 分离）

关键最佳实践顺序：
1) EQ/动态/（可选饱和）
2) LUFS 归一
3) 最终 Limiter（归一后兜底）

依赖：pedalboard, pyloudnorm, numpy
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pyloudnorm as pyln

# --- pedalboard imports (兼容 Saturation) ---
from pedalboard import (
    Pedalboard,
    HighpassFilter,
    LowShelfFilter,
    PeakFilter,
    HighShelfFilter,
    Compressor,
    Limiter,
)

try:
    # 某些 pedalboard 版本没有 Saturation
    from pedalboard import Saturation  # type: ignore
except Exception:
    Saturation = None  # type: ignore

from pedalboard.io import AudioFile


@dataclass
class ProPreset:
    # Loudness / limiter
    target_lufs: float = -14.0
    true_peak_ceiling_db: float = -1.0  # 更保守可用 -1.5

    # Cleanup
    highpass_hz: float = 40.0  # Pro 默认别高于 60
    low_shelf_hz: float = 200.0
    low_shelf_db: float = -1.0

    # De-mud
    mud_hz: float = 300.0
    mud_db: float = -2.0
    mud_q: float = 0.9

    # Presence / clarity (keep gentle)
    clarity_hz: float = 3200.0
    clarity_db: float = 1.0
    clarity_q: float = 0.9

    # Air
    air_hz: float = 10000.0
    air_db: float = 1.0

    # Optional hum notch (set hum_hz=None to disable)
    hum_hz: Optional[float] = None  # 50 or 60
    hum_db: float = -12.0
    hum_q: float = 30.0
    hum_harmonics: int = 3  # notch 1x..Nx (e.g., 60/120/180)

    # Dynamics (2-stage compression: gentle)
    comp1_threshold_db: float = -22.0
    comp1_ratio: float = 2.5
    comp1_attack_ms: float = 20.0
    comp1_release_ms: float = 160.0

    comp2_threshold_db: float = -18.0
    comp2_ratio: float = 1.8
    comp2_attack_ms: float = 12.0
    comp2_release_ms: float = 120.0

    # Saturation
    sat_drive_db: float = 3.0  # Pro 轻微谐波增强；0 表示关闭


class ACEStepAudioEnhancerProV2Compat:
    def __init__(self, preset: ProPreset):
        self.preset = preset
        self.board_pre = self._build_pre_chain(preset)
        self.board_post = self._build_post_chain(preset)
        self._has_builtin_saturation = Saturation is not None

    @staticmethod
    def _build_pre_chain(p: ProPreset) -> Pedalboard:
        fx = [
            HighpassFilter(cutoff_frequency_hz=float(p.highpass_hz)),
            LowShelfFilter(cutoff_frequency_hz=float(p.low_shelf_hz), gain_db=float(p.low_shelf_db)),
            PeakFilter(cutoff_frequency_hz=float(p.mud_hz), gain_db=float(p.mud_db), q=float(p.mud_q)),
            PeakFilter(cutoff_frequency_hz=float(p.clarity_hz), gain_db=float(p.clarity_db), q=float(p.clarity_q)),
            HighShelfFilter(cutoff_frequency_hz=float(p.air_hz), gain_db=float(p.air_db)),
            Compressor(
                threshold_db=float(p.comp1_threshold_db),
                ratio=float(p.comp1_ratio),
                attack_ms=float(p.comp1_attack_ms),
                release_ms=float(p.comp1_release_ms),
            ),
            Compressor(
                threshold_db=float(p.comp2_threshold_db),
                ratio=float(p.comp2_ratio),
                attack_ms=float(p.comp2_attack_ms),
                release_ms=float(p.comp2_release_ms),
            ),
        ]

        # Optional hum notch (50/60Hz + harmonics)
        if p.hum_hz is not None and p.hum_hz > 0:
            for k in range(1, int(p.hum_harmonics) + 1):
                fx.insert(
                    1,  # after highpass
                    PeakFilter(
                        cutoff_frequency_hz=float(p.hum_hz * k),
                        gain_db=float(p.hum_db),
                        q=float(p.hum_q),
                    ),
                )

        # Builtin saturation if available
        if p.sat_drive_db and p.sat_drive_db > 0 and Saturation is not None:
            fx.append(Saturation(drive_db=float(p.sat_drive_db)))  # type: ignore

        return Pedalboard(fx)

    @staticmethod
    def _build_post_chain(p: ProPreset) -> Pedalboard:
        # Final limiter after loudness normalization
        return Pedalboard([
            Limiter(threshold_db=float(p.true_peak_ceiling_db)),
        ])

    @staticmethod
    def _to_float_audio(audio: np.ndarray) -> np.ndarray:
        if audio.dtype.kind != "f":
            audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
        else:
            audio = audio.astype(np.float32, copy=False)
        if audio.ndim == 1:
            audio = audio[:, None]
        return audio

    @staticmethod
    def _measure_lufs(audio: np.ndarray, sr: int) -> float:
        meter = pyln.Meter(sr)  # ITU-R BS.1770
        return float(meter.integrated_loudness(audio))

    @staticmethod
    def _soft_saturate(audio: np.ndarray, drive_db: float) -> np.ndarray:
        """Soft saturation fallback (tanh). audio: float32, (samples, channels)."""
        if drive_db <= 0:
            return audio
        drive = 10 ** (drive_db / 20.0)
        # Normalize by tanh(drive) to keep level comparable
        denom = np.tanh(drive)
        if denom == 0:
            return audio
        y = np.tanh(audio * drive) / denom
        return y.astype(np.float32, copy=False)

    def process(self, input_file: str, output_file: str) -> Tuple[float, float]:
        in_path = Path(input_file)
        if not in_path.exists():
            raise FileNotFoundError(f"Input not found: {input_file}")

        # Read audio (AudioFile may return (channels, samples))
        with AudioFile(str(in_path)) as f:
            sr = f.samplerate
            audio = f.read(f.frames)

        if isinstance(audio, (list, tuple)):
            audio = np.asarray(audio)
        if getattr(audio, "ndim", 0) == 2 and audio.shape[0] <= 8 and audio.shape[0] < audio.shape[1]:
            audio = audio.T
        audio = self._to_float_audio(audio)

        # 1) Pre chain
        effected = self.board_pre(audio, sr)

        # 1b) Fallback saturation if builtin not available
        if (not self._has_builtin_saturation) and self.preset.sat_drive_db and self.preset.sat_drive_db > 0:
            effected = self._soft_saturate(effected, float(self.preset.sat_drive_db))

        # 2) Loudness normalization
        in_lufs = self._measure_lufs(effected, sr)
        target = float(self.preset.target_lufs)
        gain_db = target - in_lufs
        gain = 10 ** (gain_db / 20.0)
        normalized = effected * gain

        # 3) Final limiter (after normalization)
        limited = self.board_post(normalized, sr)

        # 4) Write output (force WAV to avoid codec surprises in WSL)
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix.lower() not in [".wav", ".wave"]:
            out_path = out_path.with_suffix(".wav")

        # Safety clamp (should rarely do anything because limiter already ran)
        limited = np.clip(limited, -1.0, 1.0).astype(np.float32, copy=False)
        out_cs = limited.T if limited.ndim == 2 else limited.reshape(1, -1)

        with AudioFile(str(out_path), "w", samplerate=sr, num_channels=out_cs.shape[0]) as f:
            f.write(out_cs)

        return in_lufs, float(gain_db)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="ACE-Step Pro audio post-processing (stable v2 compat).")
    ap.add_argument("input", help="Input audio file (.mp3/.wav/.flac). Output defaults to WAV.")
    ap.add_argument("-o", "--output", help="Output file path. Default: input_stem + _pro.wav")
    ap.add_argument("--target-lufs", type=float, default=-14.0, help="Target integrated loudness (LUFS).")
    ap.add_argument("--ceiling-db", type=float, default=-1.0, help="Limiter ceiling in dBFS (e.g., -1.0 or -1.5).")
    ap.add_argument("--hum-hz", type=float, default=None, help="Optional hum notch base freq (50 or 60).")
    ap.add_argument("--sat-drive-db", type=float, default=3.0, help="Saturation drive in dB (0 disables).")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else in_path.with_name(f"{in_path.stem}_pro.wav")

    preset = ProPreset(
        target_lufs=float(args.target_lufs),
        true_peak_ceiling_db=float(args.ceiling_db),
        hum_hz=(float(args.hum_hz) if args.hum_hz is not None else None),
        sat_drive_db=float(args.sat_drive_db),
    )

    enhancer = ACEStepAudioEnhancerProV2Compat(preset)
    in_lufs, gain_db = enhancer.process(str(in_path), str(out_path))

    sat_mode = "builtin" if (Saturation is not None) else "soft(tanh)"
    print(f"✅ Done: {out_path}")
    print(f"   Measured LUFS(before norm): {in_lufs:.2f}")
    print(f"   Applied gain: {gain_db:+.2f} dB")
    print(f"   Limiter ceiling: {preset.true_peak_ceiling_db:.2f} dBFS")
    print(f"   Saturation: {sat_mode}, drive={preset.sat_drive_db:.1f} dB")


if __name__ == "__main__":
    main()
