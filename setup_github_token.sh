#!/bin/bash
# GitHub Token 配置脚本

echo "=========================================="
echo "GitHub MCP 服务器 Token 配置"
echo "=========================================="
echo ""

# 检查是否已有 GITHUB_PERSONAL_ACCESS_TOKEN
if [ -n "$GITHUB_PERSONAL_ACCESS_TOKEN" ]; then
    echo "✓ 检测到已设置 GITHUB_PERSONAL_ACCESS_TOKEN 环境变量"
    echo "  当前值: ${GITHUB_PERSONAL_ACCESS_TOKEN:0:10}..."
    read -p "是否要更新？(y/n): " update
    if [ "$update" != "y" ]; then
        echo "保持现有配置"
        exit 0
    fi
fi

echo ""
echo "请按照以下步骤获取 GitHub Personal Access Token:"
echo ""
echo "1. 访问: https://github.com/settings/tokens"
echo "2. 点击 'Generate new token (classic)'"
echo "3. 填写备注（如: Claude Code MCP）"
echo "4. 选择过期时间"
echo "5. 勾选权限:"
echo "   - repo (完整仓库访问)"
echo "   - read:org (读取组织信息)"
echo "   - read:user (读取用户信息)"
echo "   - user:email (访问用户邮箱)"
echo "6. 点击 'Generate token'"
echo "7. 复制生成的 token（以 ghp_ 开头）"
echo ""
read -p "请输入您的 GitHub Token: " token

if [ -z "$token" ]; then
    echo "错误: Token 不能为空"
    exit 1
fi

# 检测 shell 类型
if [ -n "$ZSH_VERSION" ]; then
    SHELL_FILE="$HOME/.zshrc"
elif [ -n "$BASH_VERSION" ]; then
    SHELL_FILE="$HOME/.bashrc"
else
    SHELL_FILE="$HOME/.profile"
fi

echo ""
echo "检测到 shell 配置文件: $SHELL_FILE"

# 检查是否已存在 GITHUB_PERSONAL_ACCESS_TOKEN
if grep -q "GITHUB_PERSONAL_ACCESS_TOKEN" "$SHELL_FILE" 2>/dev/null; then
    echo "发现已存在的 GITHUB_PERSONAL_ACCESS_TOKEN 配置，将更新它"
    # 使用 sed 更新（兼容 macOS 和 Linux）
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "s|export GITHUB_PERSONAL_ACCESS_TOKEN=.*|export GITHUB_PERSONAL_ACCESS_TOKEN=\"$token\"|" "$SHELL_FILE"
    else
        sed -i "s|export GITHUB_PERSONAL_ACCESS_TOKEN=.*|export GITHUB_PERSONAL_ACCESS_TOKEN=\"$token\"|" "$SHELL_FILE"
    fi
else
    echo "添加 GITHUB_PERSONAL_ACCESS_TOKEN 到 $SHELL_FILE"
    echo "" >> "$SHELL_FILE"
    echo "# GitHub Personal Access Token for MCP Server" >> "$SHELL_FILE"
    echo "export GITHUB_PERSONAL_ACCESS_TOKEN=\"$token\"" >> "$SHELL_FILE"
fi

# 立即导出到当前会话
export GITHUB_PERSONAL_ACCESS_TOKEN="$token"

echo ""
echo "✓ Token 已配置！"
echo ""
echo "当前会话已生效，但需要重新加载 shell 配置才能永久生效。"
echo ""
echo "请运行以下命令之一："
echo "  source $SHELL_FILE"
echo "  或者"
echo "  重新打开终端"
echo ""
echo "然后重启 Claude Code 以使用 GitHub MCP 服务器。"

