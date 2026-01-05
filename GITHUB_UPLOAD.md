# GitHub上传指南

## 快速上传到GitHub

### 1. 在GitHub上创建新仓库

访问 https://github.com/new 创建新仓库，命名为 `audio_normalized` 或 `comfyui-audio-normalized`

**注意：不要勾选添加README、.gitignore或license（我们已经有了）**

### 2. 在本地初始化Git仓库

在项目目录中执行：

```bash
cd /Users/pfc/Documents/audio_normalized

# 初始化Git仓库
git init

# 添加所有文件
git add .

# 提交
git commit -m "Initial commit: ComfyUI Audio Normalize Node"

# 添加远程仓库（替换YOUR_USERNAME为你的GitHub用户名）
git remote add origin https://github.com/YOUR_USERNAME/audio_normalized.git

# 推送到GitHub
git branch -M main
git push -u origin main
```

### 3. 用户如何通过ComfyUI Manager安装

上传后，用户可以通过以下方式安装：

#### 方式A：ComfyUI Manager（如果已被收录）
1. 打开ComfyUI Manager
2. 搜索 "Audio Normalize"
3. 点击安装

#### 方式B：直接使用Git URL
1. 打开ComfyUI Manager
2. 点击 "Install via Git URL"
3. 输入：`https://github.com/YOUR_USERNAME/audio_normalized.git`
4. 点击安装

#### 方式C：手动安装
```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/YOUR_USERNAME/audio_normalized.git
cd audio_normalized
pip install -r requirements.txt
```

### 4. 提交到ComfyUI Manager节点列表（可选）

如果想让你的节点出现在Manager的官方列表中：

1. Fork [ComfyUI-Manager仓库](https://github.com/ltdrdata/ComfyUI-Manager)
2. 编辑 `custom-node-list.json` 文件
3. 添加你的节点信息：

```json
{
    "author": "Your Name",
    "title": "ComfyUI Audio Normalize Node",
    "reference": "https://github.com/YOUR_USERNAME/audio_normalized",
    "files": [
        "https://github.com/YOUR_USERNAME/audio_normalized"
    ],
    "install_type": "git-clone",
    "description": "音频处理节点：去浑浊、提亮、压缩、LUFS音量归一化"
}
```

4. 提交Pull Request

### 5. 更新README中的链接

别忘了更新 `README.md` 和 `pyproject.toml` 中的 `YOUR_USERNAME` 为你的实际GitHub用户名！

## 项目文件结构

```
audio_normalized/
├── __init__.py                 # 节点包初始化
├── audio_normalize_node.py     # 主节点代码
├── requirements.txt            # 依赖列表
├── README.md                   # 使用说明
├── pyproject.toml              # Python项目配置
├── .gitignore                  # Git忽略文件
└── GITHUB_UPLOAD.md           # 本文件
```

所有文件都已经准备好，可以直接上传！
