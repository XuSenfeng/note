# picoclaw

是一个使用go实现的支持多种架构的claw实现, 轻量级可以运行在rv1106

## 代码实现

### 目录

+ cmd: 命令行工具, 最终还是使用pkg里面的实现
+ web: picoclaw 的 web 服务（web/backend 和 web/frontend）

1. 后端（web/backend）通过 API 层与 picoclaw 交互：
    - web/backend/api/ 目录下有多个以 go 语言实现的 API 文件（如 gateway.go、pico.go、channels.go 等），这些文件定义了 HTTP 路由和处理函数，负责接收前端请求并调用 picoclaw 的核心逻辑。
    - 这些 API 处理函数会调用 pkg/ 目录下的业务逻辑（如 agent、gateway、session、skills 等），实现数据的读取、写入和业务操作。
2. 前端（web/frontend）通过 HTTP 请求与后端通信：
    - web/frontend/src/api/ 目录下有多个 ts 文件（如 gateway.ts、pico.ts、channels.ts 等），这些文件封装了对后端 API 的 HTTP 请求。
    - 前端页面通过这些 API 文件发起请求，获取数据或提交操作，后端再将请求转发到 picoclaw 的核心逻辑。
3. 通信流程总结：
    - 用户在前端操作，前端通过 HTTP API（RESTful 风格）请求 web/backend。
    - web/backend 的 API 层接收请求，调用 pkg/ 目录下的 Go 代码（picoclaw 的核心模块）处理业务。
    - 处理结果通过 API 返回给前端，前端再渲染展示。

+ pkg: 包含了 picoclaw 的主要功能模块，比如 agent（智能体）、channels（多平台通道）、commands（命令系统）、session（会话）、skills（技能）、gateway（网关）、mcp（模型上下文协议）、providers（第三方服务集成）等。每个子目录实现了对应的业务逻辑和接口

### 功能实现

#### 工作区创建

把工作区的文件夹记录为一个二进制的文件系统, 创建的时候把文件复制出来

### 状态记录state

state.json文件里面记录最后一次对话的时间以及对话消息, 只会把最后一次活跃的频道（LastChannel）、聊天ID（LastChatID）和更新时间（Timestamp）覆盖写入 state.json 文件。每次调用 SetLastChannel 或 SetLastChatID，都会用最新的数据覆盖原有内容
