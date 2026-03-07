# OpenClaw

一个可以集成到各种聊天软件里面的AI工具

## 基础概念

### 智能体agent

一个**智能体**是一个完全独立作用域的大脑，拥有自己的：

-   **工作区**（文件、AGENTS.md/SOUL.md/USER.md、本地笔记、人设规则）。
-   **状态目录**（`agentDir`）用于认证配置文件、模型注册表和每智能体配置。
-   **会话存储**（聊天历史 + 路由状态）位于 `~/.openclaw/agents/<agentId>/sessions` 下。

认证配置文件是**每智能体独立的**。每个智能体从自己的位置读取：

```
~/.openclaw/agents/<agentId>/agent/auth-profiles.json
```

主智能体凭证**不会**自动共享。切勿在智能体之间重用 `agentDir`（这会导致认证/会话冲突）。如果你想共享凭证，请将 `auth-profiles.json` 复制到另一个智能体的 `agentDir`。Skills 通过每个工作区的 `skills/` 文件夹实现每智能体独立，共享的 Skills 可从 `~/.openclaw/skills` 获取

-   `agentId` 默认为 **`main`**。
-   会话键为 `agent:main:<mainKey>`。

使用**多个智能体**，每个 `agentId` 成为一个**完全隔离的人格**

-   **不同的电话号码/账户**（每渠道 `accountId`）。
-   **不同的人格**（每智能体工作区文件如 `AGENTS.md` 和 `SOUL.md`）。
-   **独立的认证 + 会话**（除非明确启用，否则无交叉通信）。

### 工作区

工作区是智能体的家。它是文件工具和工作区上下文使用的唯一工作目录。请保持其私密性并将其视为记忆。

-   `~/.openclaw/workspace`（或 `~/.openclaw/workspace-<agentId>`）

>   工作区默认为 `~/.openclaw/workspace`（或当设置了 `OPENCLAW_PROFILE` 时为 `~/.openclaw/workspace-<profile>`）

### 会话

直接聊天折叠为 `agent:<agentId>:<mainKey>`（默认 `main`），而群组/频道聊天获得各自的键。`session.mainKey` 会被遵循

## 基础命令

https://tbbbk.com/openclaw-cli-commands-guide/

### 控制台命令

```bash
openclaw gateway # 启动主程序
openclaw tui # 控制台
openclaw dashboard # 网页控制台
openclaw channels logout # 退出登录的聊天软件
openclaw channels login # 登录聊天软件
openclaw config # 配置, 切换模型等
openclaw onboard # 交互向导
openclaw doctor # 健康检测修复
```

OpenClaw的核心是一个Websocket服务

```bash
# 查看 Gateway 状态
openclaw gateway status

# 启动服务（通过 systemd/launchd）
openclaw gateway start

# 停止服务
openclaw gateway stop

# 重启服务（改完配置后常用）
openclaw gateway restart

# 前台运行（调试时使用，可以看到实时日志）
openclaw gateway run --verbose

# 强制启动（杀掉占用端口的进程）
openclaw gateway start --force

# 指定端口
openclaw gateway --port 19000

# 绑定模式
openclaw gateway --bind loopback   # 仅本机访问（默认，最安全）
openclaw gateway --bind lan        # 局域网可访问
openclaw gateway --bind tailnet    # Tailscale 网络

# 通过 Tailscale 暴露服务
openclaw gateway --tailscale serve   # 内网暴露
openclaw gateway --tailscale funnel  # 公网暴露（需要 Tailscale 账户）
```

### agent操作

使用智能体向导添加新的隔离智能体：

```bash
openclaw agents add work
```

然后添加 `bindings`（或让向导完成）来路由入站消息。验证：

```
openclaw agents list --bindings
```

### 通道

他接入的聊天称为通道, 支持多种聊天平台接入, 比如飞书, OpenClaw 通过这些通道与你交互

```bash
# 列出已配置的频道
openclaw channels list

# 查看频道连接状态
openclaw channels status

# 添加新频道（交互式）
openclaw channels add

# 登录频道（如 WhatsApp 需要扫码）
openclaw channels login

# 登出频道
openclaw channels logout

# 查看频道支持的功能
openclaw channels capabilities
```

### 模型

使用的模型配置

```bash
# 列出已配置的模型
openclaw models list

# 查看当前使用的模型
openclaw models status

# 设置默认模型
openclaw models set claude-sonnet-4-20250514
openclaw models set gemini-2.5-pro
openclaw models set gpt-4o

# 设置图像生成模型
openclaw models set-image dall-e-3

# 管理模型别名（给长模型名起短名）
openclaw models aliases

# 设置 fallback 链（主模型挂了自动切备用）
openclaw models fallbacks

# 管理 API Key
openclaw models auth
```

### 会话

会话, 所有开启的对话, 包含历史记录和上下文。

```bash
# 列出所有会话
openclaw sessions

# 只看最近 2 小时活跃的会话
openclaw sessions --active 120

# JSON 格式输出（方便脚本处理）
openclaw sessions --json
```

终端对话

```bash
# 基础对话
openclaw agent --message "今天天气怎么样？"

# 指定会话 ID（继续之前的对话）
openclaw agent --session-id 1234 --message "继续上次的话题"

# 指定思考级别（让 AI 想久一点）
openclaw agent --message "分析这段代码的性能问题" --thinking medium

# 对话完把结果发到 Telegram
openclaw agent --message "生成周报" --deliver --channel telegram --reply-to "@mychat"
```

### 记忆系统

```bash
# 查看记忆索引状态
openclaw memory status

# 重建索引（添加新文件后）
openclaw memory index

# 语义搜索记忆
openclaw memory search "上次讨论的服务器配置"
```

### 定时任务

```bash
# 查看调度器状态
openclaw cron status

# 列出所有定时任务
openclaw cron list

# 添加新任务（交互式）
openclaw cron add

# 手动触发某个任务
openclaw cron run <job-id>

# 启用/禁用任务
openclaw cron enable <job-id>
openclaw cron disable <job-id>

# 编辑任务
openclaw cron edit <job-id>

# 查看任务运行历史
openclaw cron runs <job-id>
```

### 插件skills

```bash
# 列出所有可用的 Skills
openclaw skills list

# 查看某个 Skill 的详细信息
openclaw skills info weather

# 检查哪些 Skills 满足依赖、可以使用
openclaw skills check
```

### 网页操作

网页自动化操作

```bash
# 查看浏览器状态
openclaw browser status

# 启动浏览器
openclaw browser start

# 打开网页
openclaw browser open https://example.com

# 列出所有标签页
openclaw browser tabs

# 截图
openclaw browser screenshot
openclaw browser screenshot --full-page

# 获取页面快照（AI 可读的结构化格式）
openclaw browser snapshot

# 点击页面元素（通过 ref 编号）
openclaw browser click 12

# 输入文字
openclaw browser type 23 "hello" --submit

# 按键
openclaw browser press Enter

# 导航到 URL
openclaw browser navigate https://google.com
```

### 多设备

多设备, 提供额外的能力和传感器。

```bash
# 查看已配对的设备节点
openclaw nodes status

# 查看某个节点的详细能力
openclaw nodes describe <node-id>

# 给设备发通知
openclaw nodes notify --node mac --body "服务器任务完成！"

# 远程拍照（需要摄像头权限）
openclaw nodes camera --node iphone --facing back

# 获取设备位置
openclaw nodes location --node iphone

# 远程执行命令（仅 Mac）
openclaw nodes run --node mac -- "ls -la ~/Desktop"
```

### 日志

```bash
# 查看最近日志
openclaw logs

# 实时跟踪日志（调试必备）
openclaw logs --follow
```

### 控制台

```bash
openclaw dashboard
```

### 聊天命令

基础命令

-   `/status` — 查看当前状态（模型、Token 用量、会话信息）
-   `/new` 或 `/reset` — 开始新对话（清空当前会话上下文）
-   `/model` — 查看或切换当前使用的模型
-   `/reasoning` — 切换思考模式（off/on/stream）

思考模式说明

-   **off** — 关闭深度思考，直接回答，速度最快
-   **on** — 开启深度思考，但思考过程不显示
-   **stream** — 开启深度思考，并实时显示思考过程

```bash
/reasoning off
/reasoning on
/reasoning stream
```

会话管理

-   `/sessions` — 列出所有活跃会话
-   `/session` — 查看当前会话详情

其他实用命令

-   `/help` — 查看帮助信息
-   `/ping` — 检查 Bot 是否在线
-   `/version` — 查看 OpenClaw 版本

上下文, 查看当前的提示词, skills等信息

```bash
/context list
/context detail
```

## 提示词配置

这些文件会自动注入到每个会话的上下文中（位于 `~/.openclaw/workspace/`）：

| 文件             | 用途             | 示例内容                       |
| ---------------- | ---------------- | ------------------------------ |
| **SOUL.md**      | Agent 性格和行为 | 核心价值观、边界、风格         |
| **IDENTITY.md**  | Agent 身份信息   | 名字、类型、表情符号、头像     |
| **USER.md**      | 用户信息         | 名字、称呼、时区、偏好         |
| **TOOLS.md**     | 环境特定笔记     | 摄像头名称、SSH 主机、TTS 设置 |
| **AGENTS.md**    | 工作区指南       | 项目上下文、行为规则           |
| **HEARTBEAT.md** | 心跳任务         | 定期检查清单                   |
| **MEMORY.md**    | 长期记忆         | 重要事件、决策、经验教训       |
| **BOOTSTRAP.md** | 首次运行向导     | 仅用于新工作区，完成后删除     |

## SKILL

放在文件`~/.openclaw/skills`目录下面即可加载

### 使用MCP

https://clawhub.ai/steipete/mcporter

通过添加一个使用mcporter的使用指南, `npm i -g mcporter`使用这个命令可以安装

>   默认是有这个插件的

可以在配置文件里面添加MCP工具, 配置文件在`/Users/jiao/.openclaw/workspace/mcporter.json`文件里面

## websocket连接

通过websocket连接网关可以实现完整的控制

连接的时候需要一个connect的数据

```json
{
  "type": "req",
  "id": "req-1772427238-3623",
  "method": "connect",
  "params": {
    "minProtocol": 3,
    "maxProtocol": 3,
    "client": {
      "id": "cli", // 命令行客户端
      "version": "1.0.0",
      "platform": "darwin", // 苹果系统
      "mode": "backend"
    },
    "role": "operator", // 操作员
    "scopes": [ // 使用的权限
      "operator.read",
      "operator.write"
    ],
    "caps": [],
    "commands": [],
    "permissions": {},
    "auth": {
      "token": "da0d4609f10a7ae0b2002a9c00e48d36e6485b83865ed97e"
    },
    "locale": "zh-CN",
    "userAgent": "openclaw-python-client/1.0.0",
    "device": {
      "id": "05cad715c32590f1a05fccfb48e8c9c9e33c49f5ddaae6c18be1c9a65ee4cef0",
      "publicKey": "+0itjmRUBPMLcs0BGFb5D7XbiXcfaquYhKLG5vgjUPA=",
      "signature": "1DpR1hNyFdCUBurVisRjzHWEJwrnthstLdOKDYNk5WlOg+OlGL9RrikC6oZrB2sJ3HY25HxFEaq/Of+eFE8GAg==",
      "signedAt": 1772427238847,
      "nonce": "0d902fcf-50e1-49b0-a4a7-da4aaff2babe"
    }
  }
}
```

### 秘钥

使用的是一个**ED25519 私钥格式**, 这个秘钥之后对用于设备的身份认证, 使用私钥签署消息

```bash
# 步骤1: 生成私钥
self.private_key = ed25519.Ed25519PrivateKey.generate()  # 随机32字节

# 步骤2: 从私钥推导公钥
self.public_key = self.private_key.public_key()

# 步骤3: 从公钥计算设备ID
fingerprint = hashlib.sha256(public_bytes).hexdigest()
self.device_id = fingerprint  # 例如: "a3f2e1..."
```

在发送消息的时候, 对如下的消息格式使用私钥进行签署

```bash
"v3|{device_id}|cli|backend|operator|operator.read,operator.write|{timestamp}|{token}|{nonce}|darwin|"
```

### 随机数nonce

在websocket连接的时候, 网关发送过来一个随机数, 用于之后一段的认真

### request_id

唯一的请求 ID, `"req-{int(time.time())}-{random.randint(0, 9999)}"`

### 系统

```bash
if sys_platform.system() == "Darwin":
    platform = "darwin"
elif sys_platform.system() == "Linux":
    platform = "linux"
elif sys_platform.system() == "Windows":
    platform = "windows"
else:
    platform = "unknown"
```

### client_id使用的模式

| 可选值      | 含义         |
| ----------- | ------------ |
| `"cli"`     | 命令行客户端 |
| `"web"`     | 网页客户端   |
| `"mobile"`  | 移动端客户端 |
| `"desktop"` | 桌面应用     |

### client_mode

| 模式             | 含义       | 典型场景                | 功能级别       |
| ---------------- | ---------- | ----------------------- | -------------- |
| **`"backend"`**  | 后端客户端 | 服务器端、CLI工具、脚本 | 🔓 **完整权限** |
| **`"frontend"`** | 前端客户端 | 浏览器、移动应用        | 🔒 **受限权限** |

## 局域网连联机

```bash
openclaw config set gateway.bind lan
```

