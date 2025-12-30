# MCP 服务器配置说明

本项目已配置以下 MCP（Model Context Protocol）服务器，用于增强 Claude Code 的功能：

## 已配置的 MCP 服务器

### 1. **memory** - 知识图谱记忆服务器
- **包名**: `@modelcontextprotocol/server-memory`
- **功能**: 提供持久化记忆功能，通过知识图谱存储和检索信息
- **用途**: 帮助 Claude 记住项目上下文、对话历史等重要信息
- **配置**: 无需额外配置

### 2. **filesystem** - 文件系统服务器
- **包名**: `@modelcontextprotocol/server-filesystem`
- **功能**: 提供安全的文件系统访问，支持读取和写入文件
- **用途**: 允许 Claude 访问和操作项目文件
- **配置**: 已配置访问 `/home/cloud_lin/projects` 目录

### 3. **github** - GitHub 集成服务器
- **包名**: `@modelcontextprotocol/server-github`
- **功能**: 提供 GitHub API 访问，可以读取仓库、创建 issue、管理 PR 等
- **用途**: 与 GitHub 仓库进行交互，查看代码、创建 issue 等
- **配置**: 
  - 需要在 `.mcp.json` 中设置 `GITHUB_PERSONAL_ACCESS_TOKEN` 环境变量
  - 获取 Token: https://github.com/settings/tokens
  - 建议权限: `repo`, `read:org` 等

### 4. **sequential-thinking** - 顺序思考服务器
- **包名**: `@modelcontextprotocol/server-sequential-thinking`
- **功能**: 提供结构化的顺序思考能力，帮助解决复杂问题
- **用途**: 增强 Claude 的逻辑推理和问题解决能力
- **配置**: 无需额外配置

### 5. **playwright** - 浏览器自动化服务器
- **包名**: `@playwright/mcp`
- **功能**: 使用 Playwright 进行浏览器自动化操作
- **用途**: 网页抓取、自动化测试、网页交互等
- **配置**: 无需额外配置（首次使用会自动安装浏览器）

### 6. **chrome-devtools** - Chrome DevTools 服务器
- **包名**: `chrome-devtools-mcp`
- **功能**: 集成 Chrome DevTools，用于调试和分析网页
- **用途**: 网页调试、性能分析、网络监控等
- **配置**: 无需额外配置

## 使用方法

1. **重启 Claude Code** 以加载新的 MCP 配置
2. 运行 `/mcp` 命令查看所有 MCP 服务器状态
3. 使用 `@github` 等命令来启用/禁用特定服务器

## 配置 GitHub Token（可选）

如果需要使用 GitHub 功能，请编辑 `.mcp.json` 文件，在 `github` 服务器的 `env` 部分添加你的 token：

```json
"env": {
  "GITHUB_PERSONAL_ACCESS_TOKEN": "your_token_here"
}
```

## 注意事项

- 所有服务器都使用 `npx -y` 自动下载和运行，无需手动安装
- 首次使用时，npx 会自动下载相应的包
- 如果某个服务器无法启动，检查网络连接和 npm 配置

