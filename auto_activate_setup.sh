#!/bin/bash

# TimesNet项目自动激活设置脚本
# 运行此脚本将在你的.bashrc中添加自动激活功能

TIMESNET_PATH="/home/wanting/TimesNet"
BASHRC_FILE="$HOME/.bashrc"

echo "🔧 正在设置TimesNet项目虚拟环境自动激活..."

# 检查是否已经添加过配置
if grep -q "# TimesNet Auto Activation" "$BASHRC_FILE" 2>/dev/null; then
    echo "⚠️  .bashrc中已存在TimesNet自动激活配置"
    echo "如需重新配置，请先手动删除现有配置后再运行此脚本"
    exit 1
fi

# 添加自动激活功能到.bashrc
cat >> "$BASHRC_FILE" << 'EOF'

# TimesNet Auto Activation
# 当进入TimesNet项目目录时自动激活虚拟环境
timesnet_auto_activate() {
    local current_dir="$(pwd)"
    local timesnet_path="/home/wanting/TimesNet"
    
    # 检查当前目录是否在TimesNet项目路径下
    if [[ "$current_dir" == "$timesnet_path"* ]]; then
        local venv_path="$timesnet_path/venv"
        
        # 如果虚拟环境存在且尚未激活
        if [ -d "$venv_path" ] && [ "$VIRTUAL_ENV" != "$venv_path" ]; then
            echo "🚀 检测到TimesNet项目，正在激活虚拟环境..."
            source "$venv_path/bin/activate"
            echo "✅ TimesNet虚拟环境已激活"
        fi
    else
        # 如果离开TimesNet项目目录且激活的是TimesNet的虚拟环境，则deactivate
        if [ "$VIRTUAL_ENV" == "$timesnet_path/venv" ]; then
            echo "👋 离开TimesNet项目，正在关闭虚拟环境..."
            deactivate
        fi
    fi
}

# 在每次cd命令后执行检查
cd() {
    builtin cd "$@"
    timesnet_auto_activate
}

# 在启动新shell时检查当前目录
timesnet_auto_activate
EOF

echo "✅ 自动激活配置已添加到 $BASHRC_FILE"
echo ""
echo "📋 使用说明："
echo "1. 重新加载.bashrc: source ~/.bashrc"
echo "2. 或者重新打开终端"
echo "3. 当你cd到TimesNet项目目录时，虚拟环境会自动激活"
echo "4. 当你离开项目目录时，虚拟环境会自动关闭"
echo ""
echo "🔧 如需移除自动激活功能，请手动编辑 $BASHRC_FILE 并删除 '# TimesNet Auto Activation' 部分" 