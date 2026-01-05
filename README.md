# ComfyUI 音频归一化节点

一个强大的ComfyUI自定义节点，用于音频处理和归一化。

## 功能特性

- ✨ **去浑浊**: 使用高通滤波器去除80Hz以下的低频浑浊声音
- 🌟 **提亮**: 高频增强（4kHz以上+2dB），让声音更清晰明亮
- 🛡️ **轻压缩**: 3:1压缩比，防止爆音，让音量更均衡
- 📊 **音量归一化**: 使用LUFS标准将音量统一到-14 LUFS（适合大多数平台）
- 💾 **灵活导出**: 支持WAV（无损）和MP3格式保存

## 安装步骤

### 方法一：通过ComfyUI Manager安装（推荐）⭐

1. 确保已安装 [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager)
2. 在ComfyUI界面中打开Manager
3. 点击 "Install Custom Nodes"
4. 搜索 "Audio Normalize" 或直接使用Git URL安装
5. 点击 "Install" 并重启ComfyUI

**或者使用Git URL直接安装：**

```bash
cd /path/to/ComfyUI/custom_nodes/
git clone https://github.com/YOUR_USERNAME/audio_normalized.git
cd audio_normalized
pip install -r requirements.txt
```

然后重启ComfyUI即可。

### 方法二：手动安装

1. 下载本仓库的所有文件
2. 将整个文件夹复制到ComfyUI的 `custom_nodes` 目录
3. 安装依赖：

```bash
cd /path/to/ComfyUI/custom_nodes/audio_normalized/
pip install -r requirements.txt
```

4. 重启ComfyUI

节点会自动加载到 `audio/processing` 分类下。

## 使用方法

### 节点参数

- **audio** (输入): 音频数据流，来自其他音频节点
- **output_format**: 输出格式选择
  - `wav`: 无损WAV格式（默认）
  - `mp3`: MP3格式（压缩但保持高质量）
- **target_loudness**: 目标响度（LUFS）
  - 默认: -14.0 LUFS（适合YouTube、Spotify等平台）
  - 范围: -30.0 到 -5.0
  - 常用值:
    - -14 LUFS: 流媒体平台标准
    - -16 LUFS: 更温和的音量
    - -11 LUFS: 更响亮（适合某些应用）

### 工作流示例

```
[Load Audio] -> [Audio Normalize Node] -> [Preview Audio]
                         ↓
                   [保存的文件路径]
```

### 输出

- **audio**: 处理后的音频数据，可以继续传递给其他节点
- **file_path**: 保存的音频文件路径（含时间戳）

## 技术细节

### 音频处理流程

1. **高通滤波** (80Hz): 去除低频浑浊和隆隆声
2. **高频增强** (4kHz+): 提升清晰度和亮度
3. **动态压缩**: 
   - 阈值: -20dB
   - 压缩比: 3:1
   - 启动时间: 10ms
   - 释放时间: 100ms
4. **LUFS归一化**: 统一响度到目标值
5. **峰值限制**: 确保不超过±1.0，防止削波

### 使用的库

- **pedalboard**: Spotify开发的专业音频处理库
- **pyloudnorm**: ITU-R BS.1770-4标准的响度测量和归一化
- **numpy**: 数值计算
- **torch**: 与ComfyUI的音频格式兼容

## 常见问题

### Q: 文件保存在哪里？
A: 文件保存在ComfyUI的output目录中，文件名格式为 `audio_normalized_YYYYMMDD_HHMMSS.{格式}`

### Q: 可以调整压缩强度吗？
A: 可以！在代码中修改 `Compressor` 的参数：
- `threshold_db`: 降低值会压缩更多
- `ratio`: 提高比例会压缩更强

### Q: -14 LUFS是什么？
A: LUFS (Loudness Units Full Scale) 是响度的标准单位。-14 LUFS是流媒体平台（YouTube、Spotify等）的推荐响度。

### Q: MP3质量如何？
A: pedalboard使用高质量的编码器，适合大多数应用。如需无损质量，请选择WAV格式。

## 许可证

MIT License

## 贡献

欢迎提交问题和改进建议！
