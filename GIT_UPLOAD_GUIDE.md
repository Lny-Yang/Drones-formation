# Git上传与服务器同步指南

## 第一步: 上传到GitHub/GitLab

### 方法1: 使用GitHub (推荐)

#### 1.1 在GitHub网站创建新仓库
1. 访问 https://github.com/new
2. Repository name: `Drones-formation` (或自定义)
3. 设置为 Private (如果不想公开) 或 Public
4. **不要勾选** "Initialize with README" (我们已经有了)
5. 点击 "Create repository"

#### 1.2 关联远程仓库并推送

在本地项目目录执行:

```powershell
# 添加远程仓库 (替换 YOUR_USERNAME 为你的GitHub用户名)
git remote add origin https://github.com/YOUR_USERNAME/Drones-formation.git

# 推送到GitHub
git branch -M main
git push -u origin main
```

**如果遇到认证问题:**

```powershell
# GitHub现在需要使用Personal Access Token (PAT)
# 1. 访问 https://github.com/settings/tokens
# 2. 点击 "Generate new token (classic)"
# 3. 勾选 "repo" 权限
# 4. 复制生成的token
# 5. 推送时使用 token 作为密码
git push -u origin main
# Username: YOUR_USERNAME
# Password: 粘贴你的token
```

### 方法2: 使用GitLab

#### 2.1 在GitLab创建新仓库
1. 访问 https://gitlab.com/projects/new
2. Project name: `Drones-formation`
3. Visibility: Private/Public
4. 点击 "Create project"

#### 2.2 推送代码

```powershell
git remote add origin https://gitlab.com/YOUR_USERNAME/Drones-formation.git
git branch -M main
git push -u origin main
```

---

## 第二步: 同步到服务器

### 方案1: 通过Git Clone (推荐)

#### 在服务器上执行:

```bash
# SSH登录服务器
ssh your_username@your_server_ip

# 克隆项目
cd ~  # 或切换到你想要的目录
git clone https://github.com/YOUR_USERNAME/Drones-formation.git

# 进入项目目录
cd Drones-formation

# 创建conda环境
conda create -n drone_sim python=3.8 -y
conda activate drone_sim

# 安装依赖
pip install -r requirements.txt
```

#### 后续更新 (当你在本地修改代码后):

**本地推送更新:**
```powershell
# 在本地提交更改
git add .
git commit -m "描述你的更改"
git push
```

**服务器拉取更新:**
```bash
# SSH到服务器
ssh your_username@your_server_ip
cd ~/Drones-formation

# 拉取最新代码
git pull
```

---

### 方案2: 直接SCP传输 (不推荐,仅用于紧急情况)

```powershell
# 从本地传输到服务器
scp -r E:\pybullet-drones\UAV-formation\real_form\Drones-formation your_username@your_server_ip:~/

# 或使用rsync (更高效,支持增量传输)
rsync -avz --exclude '.git' --exclude '__pycache__' `
  E:\pybullet-drones\UAV-formation\real_form\Drones-formation\ `
  your_username@your_server_ip:~/Drones-formation/
```

---

## 第三步: 服务器上运行训练

### 3.1 测试环境

```bash
# 激活环境
conda activate drone_sim

# 测试PyBullet
python -c "import pybullet as p; print('PyBullet version:', p.getVersionInfo())"

# 测试环境加载
python -c "from muti_formation.drone_envs.envs.drone_env_multi import DroneNavigationMulti; print('Environment OK')"
```

### 3.2 启动训练 (使用tmux防止断开)

```bash
# 安装tmux (如果没有)
sudo apt-get install tmux  # Ubuntu/Debian
# 或
sudo yum install tmux      # CentOS/RHEL

# 创建训练会话
tmux new -s drone_training

# 在tmux中启动训练
cd ~/Drones-formation
conda activate drone_sim
python muti_formation/train_phase1_SB3.py

# 按 Ctrl+B, 然后按 D 退出tmux会话 (训练继续运行)
# 重新连接: tmux attach -t drone_training
```

### 3.3 监控训练进度

```bash
# 方法1: 查看日志文件
tail -f muti_formation/agent/log_SB3/training_data.json

# 方法2: 使用TensorBoard (如果配置了)
tensorboard --logdir=muti_formation/tensorboard --port=6006
# 然后在本地浏览器访问: http://your_server_ip:6006
```

---

## 第四步: 从服务器下载训练结果

### 下载模型文件

```powershell
# 下载最新模型
scp your_username@your_server_ip:~/Drones-formation/muti_formation/agent/model_SB3/leader_phase1_final.zip `
  E:\pybullet-drones\UAV-formation\real_form\Drones-formation\muti_formation\agent\model_SB3\

# 下载训练日志
scp your_username@your_server_ip:~/Drones-formation/muti_formation/agent/log_SB3/training_data.json `
  E:\pybullet-drones\UAV-formation\real_form\Drones-formation\muti_formation\agent\log_SB3\

# 或使用rsync同步整个结果目录
rsync -avz your_username@your_server_ip:~/Drones-formation/muti_formation/agent/log_SB3/ `
  E:\pybullet-drones\UAV-formation\real_form\Drones-formation\muti_formation\agent\log_SB3\
```

---

## 常见问题

### Q1: Git push被拒绝 (large file error)

**原因:** GitHub限制单个文件最大100MB

**解决方案:**
```powershell
# 检查大文件
git ls-files -s | awk '$4 > 100000000 {print $4, $NF}'

# 已在 .gitignore 中排除了 *.pth 和 *.zip
# 确认这些文件未被跟踪:
git status --ignored
```

### Q2: 服务器无GPU,训练很慢

**解决方案:**
```bash
# 修改配置使用CPU
# 在 train_phase1_SB3.py 中:
device = "cpu"  # 而不是 "cuda"

# 或减少批次大小和网络复杂度
```

### Q3: SSH连接超时

**本地配置SSH保活:**
```powershell
# 编辑 ~/.ssh/config (Windows: C:\Users\YOUR_NAME\.ssh\config)
echo "Host *" >> ~/.ssh/config
echo "    ServerAliveInterval 60" >> ~/.ssh/config
echo "    ServerAliveCountMax 3" >> ~/.ssh/config
```

### Q4: 想要在服务器上可视化GUI

**方案A: X11转发 (慢)**
```bash
ssh -X your_username@your_server_ip
# 然后运行Python脚本
```

**方案B: 录制视频后下载**
```python
# 在代码中添加视频录制
env = gym.make("DroneNavigation-v0")
env = gym.wrappers.RecordVideo(env, "videos/", episode_trigger=lambda x: x % 10 == 0)
```

---

## 快速命令参考

```powershell
# === 本地操作 ===
# 提交更改
git add .
git commit -m "更新描述"
git push

# 查看状态
git status
git log --oneline -5

# === 服务器操作 ===
# SSH登录
ssh your_username@your_server_ip

# 拉取更新
cd ~/Drones-formation && git pull

# tmux会话管理
tmux new -s train          # 创建会话
tmux ls                     # 列出会话
tmux attach -t train       # 连接会话
# Ctrl+B, D                 # 断开会话

# 下载结果
scp -r your_username@your_server_ip:~/Drones-formation/muti_formation/agent/log_SB3/ ./local_dir/
```

---

## 项目文件大小说明

**.gitignore 已排除的大文件:**
- `agent/model_SB3/*.zip` (模型检查点, 每个 ~50MB)
- `agent/log_SB3/training_data.json` (训练日志, 可能很大)
- `tensorboard/` (TensorBoard日志)
- `*.pth` (PyTorch模型文件)
- `__pycache__/`, `*.pyc` (Python缓存)

**上传到Git的内容:**
- ✅ 源代码 (.py文件)
- ✅ 配置文件 (config.py)
- ✅ URDF模型资源
- ✅ README.md, requirements.txt
- ✅ 小型日志文件 (.json < 10MB)
- ❌ 训练好的模型 (需单独传输)
- ❌ 大型日志和检查点

**首次上传后仓库大小:** 预计 ~50MB (不含模型文件)

---

**祝训练顺利!如有问题,可查阅:**
- Git文档: https://git-scm.com/doc
- GitHub帮助: https://docs.github.com/
- Stable-Baselines3: https://stable-baselines3.readthedocs.io/
