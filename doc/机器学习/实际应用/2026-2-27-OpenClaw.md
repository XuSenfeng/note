# OpenClaw

一个可以集成到各种聊天软件里面的AI工具

## 基础命令

```bash
openclaw gateway # 启动主程序
openclaw tui # 控制台
openclaw dashboard # 网页控制台
openclaw channels logout # 退出登录的聊天软件
openclaw channels login # 登录聊天软件
openclaw config # 配置, 切换模型等
```

## SKILL

放在文件`~/.openclaw/skills`目录下面即可加载

### 使用MCP

https://clawhub.ai/steipete/mcporter

通过添加一个使用mcporter的使用指南, `npm i -g mcporter`使用这个命令可以安装

>   默认是有这个插件的

可以在配置文件里面添加MCP工具, 配置文件在`/Users/jiao/.openclaw/workspace/mcporter.json`文件里面



