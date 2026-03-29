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

部署以后可以在`127.0.0.1`进行访问

## 插件商店安装

在容器里面执行

```bash
wget -O - https://get.hacs.xyz | bash -
```

执行以后重启容器

![image-20260228171746184](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/mac-picture/image-20260228171746184.png)

![image-20260228171729164](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/mac-picture/image-20260228171729164.png)

同意所有协议, 网站登录输入验证码即可

之后用于小米插件安装

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

添加到openclaw使用的mcporter的配置文件里面

### Xiaomi

集成小米可以使用Xiaomi Miot这个插件, 下载以后重启

![image-20260228172153517](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/mac-picture/image-20260228172153517.png)

可以使用小米的账号集成

### 主题

可以使用your name插件, 下载以后出现一个themes/yourname文件夹, 这个文件里面是实际使用的配置文件

### Mqtt设备

在设置的设备和集成页面里面搜索MQTT, 添加一个MQTT组件, 连接到MQTT服务器

之后发布消息在home assistant注册**属性(实体)**, 不同的属性通过dev里面的参数区分是在哪一个设备下面

#### 消息

+   设备订阅

在`homeassistant/sensor/esp-temp-01/config`主题发送消息

```json
{
    "unique_id": "esp-temp-01",
    "name": "温度传感器-01",
    "icon": "mdi:thermometer",
    "state_topic": "home/sensor/esp-temp-01/state",
    "json_attributes_topic": "home/sensor/esp-temp-01/attributes",
    "unit_of_measurement": "℃",
  	"value_template": "{{ value_json.humidity }}",
    "device": {
        "identifiers": "ESP32-01",
        "manufacturer": "若甫科技有限公司",
        "model": "HA",
        "name": "ESP32-01",
        "sw_version": "1.0"
    }
}
```

##### 发布的主题

HomeAssistant 设备自动发现的 MQTT 主题一般格式如下：

```
homeassistant/{component}/{unique_id}/config
```

-   `{component}`：设备类型，如 sensor（传感器）、switch（开关）
-   `{unique_id}`：设备唯一标识，建议使用设备类型+编号，例如 esp-temp-01

如果我们要注册多个同种设备，只需要保持注册信息里的`device `不变

##### 通用字段

| 字段                    | 说明                                                         |
| :---------------------- | :----------------------------------------------------------- |
| `unique_id`             | 设备的唯一 ID，HomeAssistant 用它来区分设备，必须唯一        |
| `name`                  | 设备在 HomeAssistant 中显示的名称                            |
| `icon`                  | 设备图标（这里是温度计）                                     |
| `state_topic`           | 设备状态数据的 MQTT 主题，HomeAssistant 通过订阅这个主题获取设备的最新状态 |
| `json_attributes_topic` | 设备的额外属性（如电压、电池状态）发布的主题，HomeAssistant 会自动读取这些属性 |

+   基础的数据参数

| 参数                  | 含义                                                         |
| --------------------- | ------------------------------------------------------------ |
| `value_template`      | 从设备上报的 JSON 里**提取状态值**, 属性的解析, 没有这个的时候可以直接发送数字, 配置以后使用json字符串 |
| `command_template`    | `把 HA 的指令**包装成设备要的 JSON**                         |
| `unit_of_measurement` | 传感器单位，这里是摄氏度                                     |

+   设备基础信息

| `device`       | 设备的基本信息                                               |
| :------------- | :----------------------------------------------------------- |
| `identifiers`  | 设备唯一标识（如 ESP8266 的 MAC 地址或者自定义 ID），用于设备管理 |
| `manufacturer` | 设备制造商信息                                               |
| `model`        | 设备型号                                                     |
| `name`         | 设备名称，HomeAssistant 设备管理界面会显示                   |
| `sw_version`   | 设备固件版本                                                 |

+   控制主题配置

| 参数               | 含义                                                         |
| :----------------- | ------------------------------------------------------------ |
| `command_topic`    | 控制消息发布到的主题, `"homeassistant/switch/light/set"`     |
| `command_template` | 命令的格式, 可以配置发送的命令是json格式`"command_template": "{ \"relay_2\": {{ value }} }"` |

+   是否在线

| 参数                    | 含义                                                  |
| ----------------------- | ----------------------------------------------------- |
| `availability_topic`    | 设备在线状态主题, "home/device/esp32_01/availability" |
| `payload_available`     | 在线标识, "online"                                    |
| `payload_not_available` | 离线的标识"offline"                                   |

##### 专属参数

+   开关"platform": "switch"

| 参数名        | 作用                   | 常用值        |
| ------------- | ---------------------- | ------------- |
| `payload_on`  | HA 发出的 “开” 指令    | `"ON"` / `1`  |
| `payload_off` | HA 发出的 “关” 指令    | `"OFF"` / `0` |
| `state_on`    | 设备上报 “开” 状态的值 | `1` / `"ON"`  |
| `state_off`   | 设备上报 “关” 状态的值 | `0` / `"OFF"` |

+   传感器"platform": "sensor"

| 参数名                | 作用                    | 示例                               |
| --------------------- | ----------------------- | ---------------------------------- |
| `unit_of_measurement` | 单位                    | `"°C"` `"%"` `"V"`                 |
| `device_class`        | 设备类型（HA 自动图标） | `temperature` `humidity` `voltage` |
| `state_class`         | 状态类型                | `measurement`（测量值）            |

##### 数据发布

设备发送如下 JSON 到 `home/sensor/esp-temp-01/state `主题：

```bash
{
  "humidity": 30
}
```

##### ##### 示例

+   二进制传感器

```json
{
  "platform": "binary_sensor",  // 设备类型
  "name": "人体感应",
  "unique_id": "esp32_01_pir",
  "state_topic": "home/device/esp32_01/state",
  "value_template": "{{ value_json.pir_status }}",  // 上报1=有人，0=无人
  "device_class": "motion",  // 设备类型（HA自动识别为“运动传感器”）
  "payload_on": 1,
  "payload_off": 0,
  "device": {"ids": ["esp32_01"]}  // 归到同一个ESP32设备下
}
```

和 `sensor` 比，它只有布尔值（0/1、on/off），HA 会自动匹配对应图标（比如门磁显示 “开门 / 关门”）。

+   light

可控的灯（支持开关、亮度、色温、RGB），比如 LED 灯、智能灯泡。比 `switch` 多了亮度 / 色温 / RGB 的配置

```json
{
  "platform": "light",
  "name": "客厅灯",
  "unique_id": "esp32_01_light",
  "state_topic": "home/device/esp32_01/state",
  "command_topic": "home/device/esp32_01/set",
  "brightness": true,  // 支持亮度调节
  "value_template": "{{ value_json.light_status }}",
  "command_template": "{\"light_brightness\":{{brightness}}, \"light_status\":{{value}}}",
  "payload_on": 1,
  "payload_off": 0,
  "device": {"ids": ["esp32_01"]}
}
```

## 脚本

可以使用noderad配置 一个脚本
