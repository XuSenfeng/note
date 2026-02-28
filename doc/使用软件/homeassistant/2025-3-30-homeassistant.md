# HomeAssistant

智能家居的控制, 不同品牌的智能家居

[AlexxIT/HassWP: Portable version of Home Assistant for Windows (no need to install)](https://github.com/AlexxIT/HassWP)

使用这个项目在Windows下面可以使用脚本进行部署

尝试使用docker部署成功

## 部署命令

```bash
docker pull homeassistant/home-assistant:latest
mkdir -p ~/JHY/homeassistant/config && cd ~/JHY/homeassistant/
docker run -d --restart always --name homeassistant  -v ~/JHY/homeassistant/config:/config -e TZ=Asia/Shanghai -p 8123:8123   homeassistant/home-assistant:latest
docker compose up -d
docker ps
```

## MCP访问

安装插件

![image-20260228170512095](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/mac-picture/image-20260228170512095.png)

获取令牌

![image-20260228170442102](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/mac-picture/image-20260228170442102.png)

工具配置

```json
"home-assistant": {
    "command": "uvx",
    "args": [
        "ha-mcp@latest"
    ],
    "env": {
        "HOMEASSISTANT_URL": "http://127.0.0.1:8123",
        "HOMEASSISTANT_TOKEN": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJlMTFjZjc2MzIzZDc0ZmIxODYxMmJlYWJhYWFjODY5YiIsImlhdCI6MTc3MjI2OTExMCwiZXhwIjoyMDg3NjI5MTEwfQ.b9tW08xJxfEnNWlEahzQW2APk-OhztHUmZBjOijAanw"
    }
}
```

## 插件安装

在容器里面执行

```bash
wget -O - https://get.hacs.xyz | bash -
```

执行以后重启容器

![image-20260228171746184](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/mac-picture/image-20260228171746184.png)

![image-20260228171729164](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/mac-picture/image-20260228171729164.png)

网站登录输入验证码即可

### Xiaomi

集成小米可以使用Xiaomi Miot这个插件, 下载以后重启

![image-20260228172153517](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/mac-picture/image-20260228172153517.png)

可以使用小米的账号集成

### 主题

可以使用your name插件, 下载以后出现一个themes/yourname文件夹, 这个文件里面是实际使用的配置文件
