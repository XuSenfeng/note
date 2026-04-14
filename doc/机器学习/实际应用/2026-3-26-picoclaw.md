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

### 基础概念

channel: 各种不同的消息输入终端, 比如QQ, 微信等

agent: 实际的AI对话处理的位置

### 功能实现

#### 工作区创建

把工作区的文件夹记录为一个二进制的文件系统, 创建的时候把文件复制出来

### 状态记录state

state.json文件里面记录最后一次对话的时间以及对话消息, 只会把最后一次活跃的频道（LastChannel）、聊天ID（LastChatID）和更新时间（Timestamp）覆盖写入 state.json 文件。每次调用 SetLastChannel 或 SetLastChatID，都会用最新的数据覆盖原有内容

### bus消息总线

不同的消息终端到agent的消息总线, 实际是使用go里面的chan实现的, 总线和agent可以相互发布消息给对方

#### agent处理


pkg/agent 目录是 picoclaw 的“智能体”核心模块，负责对话上下文、推理、技能调用、hook、事件等智能处理。下面简要介绍各文件/功能：

1. context.go / context_budget.go / context_cache.go
     对话上下文管理，包括上下文对象、缓存、token预算等，保证多轮对话的连续性和资源控制。

     提示词里面有固定提示词, 工作区文件提示词, 近期记忆以及永久记忆提示词, 各种skill的基础信息

     使用cacheBaseline记录所有文件的最后更新时间

     context_budget文件里面处理对话的可以的分界, 以及大致计算使用的token数量, 防止超出数量

2. definition.go
     定义 agent 的结构、接口、能力描述等，是 agent 的“蓝图”。(加载workspace里面的提示词文件)

3. eventbus.go / events.go
     agent 内部事件总线，实现事件的发布、订阅、分发，支持解耦的事件驱动逻辑。

     eventbus: 轻量级的事件总线（EventBus），用于在 agent 内部进行事件广播和订阅，支持多订阅者、并发安全、非阻塞广播, 只可以广播

     events: 定义了所有 agent 主循环及其子流程中可能发生的事件类型（如回合开始/结束、LLM 请求/响应、工具调用、上下文压缩、子回合、错误等），用 EventKind 枚举表示

     创建基础的事件元素Event

4. hook_mount.go / hook_process.go / hooks.go
     hook 系统，支持在 agent 生命周期、消息处理等关键节点挂载自定义逻辑（如前置/后置处理、拦截等）。

     在处理的时候把一个进程的标准输入输出作为管道使用, 不同的进程启动使用的是json字符串以及执行命令的形式

     执行的命令作为另一个进程处理收取到的信息, 可以处理对话前后, 工具调用前后, 以及工具的使能等事件(实际好像都没有用到)

5. instance.go
     agent 实例的创建、生命周期管理，支持多 agent 并发运行。

     使用AgentInstance保存代理配置与能力（模型、工具、上下文构建器、会话存储等）。

6. loop.go / loop_mcp.go / loop_media.go
     agent 的主循环，负责消息的接收、处理、回复。loop_mcp/loop_media 针对不同场景（如 MCP 协议、媒体消息）做适配。

     loop_mcp: 管理 MCP（Model Context Protocol）服务器的初始化和运行时状态, 加载使能的MCP工具, 添加工具搜索功能

     loop_media: 处理媒体相关的信息, 比如图片, 通过读取文件的前面几个字节实现, 把图片变为base64, 其他的文件变为路径等, 以及转换提示里面的[file]为[file:/test.txt]等功能

     

7. memory.go
     agent 的记忆系统，存储和检索对话历史、知识等。获取近期的记忆以及存储记忆的实现

8. model_resolution.go
     模型选择与分辨，决定调用哪个大模型或推理后端。把模型按照顺序进行排序, 添加供应商以及工作区, 同时提供模型查找的接口

9. registry.go
     agent 注册表，管理所有 agent 的注册、查找、调度。

10. steering.go
     对话引导、意图识别、上下文 steering（如多 agent 协作时的分流/转发）。

11. subturn.go / turn.go
      对话轮次管理，支持多轮对话、子轮次（如多步推理）。

12. thinking.go
      推理与思考流程，支持复杂的 agent 思考链（Chain-of-Thought）。

### media

### store

+ FileMediaStore: 

1. 管理 media:// 引用与你本地文件路径之间的映射
2. 保存每个媒体文件的元数据（如 MIME 类型、来源、清理策略）
3. 按作用域（scope）组织引用，便于批量释放
4. 支持可选的后台过期清理

>  `Store(...)`：把本地文件注册为 `media://...` 引用
>
> `Resolve(...)` / `ResolveWithMeta(...)`：根据引用查出真实本地路径和元数据
>
> `ReleaseAll(scope)`：删除某个 scope 下的所有引用，并在需要时删除文件
>
> `CleanExpired()`：按 TTL 清理过期媒体
>
> `Start()` / `Stop()`：启动/停止后台清理线程
