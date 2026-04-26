# Claude Code

这个安装需要使用外网的VPN

```bash
curl -fsSL https://claude.ai/install.sh | bash
export PATH="$HOME/.local/bin:$PATH"
```

## 使用中转站

```bash
# 临时设置（当前会话有效）
$env:ANTHROPIC_BASE_URL="https://api.weelinking.com"
$env:ANTHROPIC_AUTH_TOKEN="sk-你的APIKey"

# 启动 Claude Code
claude
```

