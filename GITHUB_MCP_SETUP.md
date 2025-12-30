# GitHub MCP 服务器配置指南

本指南将帮助您配置 GitHub MCP 服务器，使 Claude Code 能够与 GitHub 进行交互。

## 步骤 1: 创建 GitHub Personal Access Token

### 1.1 访问 GitHub Token 设置页面

1. 登录您的 GitHub 账户
2. 点击右上角头像，选择 **Settings（设置）**
3. 在左侧菜单中，滚动到底部，点击 **Developer settings（开发者设置）**
4. 点击 **Personal access tokens（个人访问令牌）** > **Tokens (classic)（令牌（经典））**
5. 或者直接访问：https://github.com/settings/tokens

### 1.2 生成新 Token

1. 点击 **Generate new token（生成新令牌）** > **Generate new token (classic)（生成新令牌（经典））**
2. 填写 Token 信息：
   - **Note（备注）**: 例如 "Claude Code MCP"
   - **Expiration（过期时间）**: 选择过期时间（建议选择较长时间，如 90 天或 1 年）
   - **Select scopes（选择权限）**: 勾选以下权限：
     - ✅ `repo` - 完整仓库访问权限（包括私有仓库）
     - ✅ `read:org` - 读取组织信息（如果需要访问组织仓库）
     - ✅ `read:user` - 读取用户信息
     - ✅ `user:email` - 访问用户邮箱

3. 点击 **Generate token（生成令牌）**

### 1.3 复制 Token

⚠️ **重要**: Token 只会显示一次，请立即复制并保存！

复制生成的 token（以 `ghp_` 开头的字符串）

## 步骤 2: 配置 .mcp.json 文件

### 方法 1: 直接编辑配置文件

编辑项目根目录下的 `.mcp.json` 文件，将 token 填入：

```json
"github": {
  "type": "stdio",
  "command": "npx",
  "args": [
    "-y",
    "@modelcontextprotocol/server-github"
  ],
  "env": {
    "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_your_token_here"
  }
}
```

### 方法 2: 使用系统环境变量（推荐，更安全）

MCP 服务器会自动从系统环境变量中读取 `GITHUB_PERSONAL_ACCESS_TOKEN`，所以您只需要：

1. **保持 `.mcp.json` 中的 `env` 为空**（已配置好）：

```json
"github": {
  "type": "stdio",
  "command": "npx",
  "args": [
    "-y",
    "@modelcontextprotocol/server-github"
  ],
  "env": {}
}
```

2. 在您的 shell 配置文件中（如 `~/.bashrc` 或 `~/.zshrc`）添加：

```bash
export GITHUB_PERSONAL_ACCESS_TOKEN="ghp_your_token_here"
```

3. 重新加载 shell 配置：

```bash
source ~/.bashrc  # 或 source ~/.zshrc
```

**注意**: MCP 配置不支持 `${VAR}` 环境变量展开语法，必须使用系统环境变量。

## 步骤 3: 验证配置

1. **重启 Claude Code** 以加载新配置

2. 运行 `/mcp` 命令，检查 GitHub 服务器状态

3. 测试 GitHub 功能，例如：
   - 查看仓库信息
   - 列出 issues
   - 创建 issue

## 安全建议

1. **不要将 token 提交到 Git 仓库**
   - 确保 `.mcp.json` 在 `.gitignore` 中（如果包含敏感信息）
   - 或者使用环境变量方式

2. **定期更新 token**
   - 如果 token 泄露，立即在 GitHub 设置中撤销它
   - 创建新 token 替换旧 token

3. **使用最小权限原则**
   - 只授予必要的权限
   - 如果只需要读取公开仓库，可以不勾选 `repo` 权限

## 故障排除

### 问题 1: Token 无效
- 检查 token 是否正确复制（没有多余空格）
- 确认 token 未过期
- 验证 token 权限是否足够

### 问题 2: 无法访问私有仓库
- 确保 token 有 `repo` 权限
- 检查仓库访问权限

### 问题 3: 服务器无法启动
- 检查网络连接
- 运行 `npx -y @modelcontextprotocol/server-github` 查看错误信息
- 确认 Node.js 和 npm 版本正常

## GitHub MCP 服务器功能

配置成功后，您可以使用以下功能：

- 📖 读取仓库内容
- 🔍 搜索代码和 issues
- 📝 创建和管理 issues
- 🔄 查看和管理 Pull Requests
- 👥 查看协作者和贡献者
- 📊 获取仓库统计信息
- 🏷️ 管理标签和里程碑

## 相关链接

- [GitHub Personal Access Tokens 文档](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token)
- [MCP GitHub 服务器文档](https://modelcontextprotocol.io)

