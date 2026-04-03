#!/bin/bash
# AtriNexus VPS 环境初始化脚本
# 使用方法: sudo bash setup_vps.sh

set -e

echo "=== AtriNexus VPS 部署脚本 ==="
echo "正在检查系统环境..."

# 1. 更新系统并安装依赖
echo ">>> 更新软件源并安装 Python3, Pip, Git..."
if command -v apt-get >/dev/null; then
    sudo apt-get update
    sudo apt-get install -y python3 python3-pip python3-venv git
elif command -v yum >/dev/null; then
    sudo yum update -y
    sudo yum install -y python3 python3-pip git
else
    echo "未知的包管理器，请手动安装 python3, pip, git"
    exit 1
fi

# 2. 设置项目目录权限
PROJECT_DIR=$(pwd)
echo ">>> 当前项目目录: $PROJECT_DIR"

# 3. 创建虚拟环境
echo ">>> 创建 Python 虚拟环境 (.venv)..."
if [ ! -f ".venv/bin/pip" ]; then
    echo "虚拟环境不存在或已损坏，重新创建..."
    rm -rf .venv
    python3 -m venv .venv
    echo "虚拟环境创建成功"
else
    echo "虚拟环境已存在且正常"
fi

# 4. 安装依赖
echo ">>> 安装项目依赖..."
./.venv/bin/pip install --upgrade pip
if [ -f "requirements.txt" ]; then
    ./.venv/bin/pip install -r requirements.txt
else
    echo "警告: 未找到 requirements.txt"
fi
# 确保 wechatpy 和 uvicorn 安装
./.venv/bin/pip install wechatpy[cryptography] uvicorn fastapi httpx python-dotenv

# 5. 配置 Systemd 服务
echo ">>> 配置 Systemd 服务..."
SERVICE_FILE="deployment/kourichat.service"
TARGET_SERVICE="/etc/systemd/system/kourichat.service"

if [ -f "$SERVICE_FILE" ]; then
    # 替换服务文件中的路径为当前路径
    sed "s|/opt/kourichat|$PROJECT_DIR|g" "$SERVICE_FILE" > temp_kourichat.service
    
    # 替换 User 为当前用户 (如果不是 root)
    CURRENT_USER=$(whoami)
    if [ "$CURRENT_USER" != "root" ]; then
        sed -i "s|User=root|User=$CURRENT_USER|g" temp_kourichat.service
    fi

    echo "正在安装服务文件到 $TARGET_SERVICE"
    sudo mv temp_kourichat.service "$TARGET_SERVICE"
    
    echo "重载 Systemd..."
    sudo systemctl daemon-reload
    sudo systemctl enable kourichat
    
    echo ">>> 服务配置完成！"
    echo "请修改 config.json 后，运行以下命令启动服务："
    echo "sudo systemctl start kourichat"
    echo "查看状态: sudo systemctl status kourichat"
    echo "查看日志: sudo journalctl -u kourichat -f"
else
    echo "错误: 未找到 $SERVICE_FILE，请确保脚本在项目根目录下运行"
fi
