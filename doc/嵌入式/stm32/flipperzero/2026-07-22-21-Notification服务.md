
# Notification服务

notification 服务是一个**后台系统服务**（`FlipperAppType.SERVICE`），系统启动时自动运行。它统一管理设备上所有的**输出外设**：

- 3 颗 RGB LED（红/绿/蓝）
- 屏幕背光
- 蜂鸣器（声音）
- 振动马达
- LCD 对比度

任何应用都不直接操作这些硬件，而是通过 `RECORD_NOTIFICATION` 记录向本服务发送"通知序列"，由服务串行执行。这样做的好处是硬件访问被集中管理，不会互相冲突

## API函数

```c
void notification_message(NotificationApp* app, const NotificationSequence* sequence);

void notification_message_block(NotificationApp* app, const NotificationSequence* sequence);
```


```c
/** Display: backlight always on lock */
const NotificationSequence sequence_display_backlight_enforce_on = {
    &message_display_backlight_enforce_on,
    NULL,
};

const NotificationSequence sequence_eat = {
    &message_note_c7,
    &message_delay_50,
    &message_sound_off,
    NULL,
};

NotificationApp* notification = furi_record_open(RECORD_NOTIFICATION);

notification_message_block(notification, &sequence_display_backlight_enforce_on);
notification_message(notification, &sequence_eat);
```

## 灯光/背光管理

双层模型（Layer 模型）是 notification 服务管理 LED 和背光的核心设计。它让"临时通知"和"常驻状态"可以叠加，通知结束后自动恢复到常驻状态。下面详细拆解。

每个灯（背光 + 红/绿/蓝）都是一个

```c
typedef enum {
    LayerInternal = 0,      // 内部层/常驻层
    LayerNotification = 1,  // 通知层/临时层
    LayerMAX = 2,
} NotificationLedLayerIndex;

typedef struct {
    uint8_t value_last[LayerMAX];  // 每层"上次设过的原始值"(未乘亮度系数)
    uint8_t value[LayerMAX];       // 每层"当前实际输出值"(已乘亮度系数)
    NotificationLedLayerIndex index; // 当前哪一层生效
    Light light;                   // 绑定的硬件(LightRed/Green/Blue/Backlight)
} NotificationLedLayer;

```

- `value[2]`：两层各自的输出值。`value[0]` 是内部层的，`value[1]` 是通知层的。
- `index`：**当前生效层**。硬件此刻显示的是 `value[index]`。
- `value_last[2]`：保存每层最近一次设置的**原始值**（点亮时是 `value = 原始值 × 亮度系数`，这里存的是没乘系数的原始值），用于亮度设置变化时按新系数重算。

### 实现思路

**内部层（LayerInternal）—— 常驻状态**

表示设备的持久状态，写入后一直保持，不会自动消失。比如"充电时红灯常亮"。通过 `notification_internal_message` 写入, 实际的处理函数是 `notification_apply_internal_led_layer`

**通知层（LayerNotification）—— 临时覆盖**

表示一次性的通知，比如"操作成功绿灯闪一下"。写入时会**抢占**显示, 使用的函数是 `notification_apply_notification_led_layer`, 这个函数会把当前的index切换为 `LayerNotification`, 通知层从不销毁内部层的值，只是临时盖住


**恢复**

使用函数 `notification_reset_notification_led_layer`, 在处理结束以后恢复 `LayerInternal` 层

默认会设置 `reset_display_mask` 标志位
```c
if(reset_mask & reset_display_mask) {
	if(!float_is_equal(display_brightness_set, app->settings.display_brightness)) {
		furi_hal_light_set(LightBacklight, app->settings.display_brightness * 0xFF);
	}
	furi_timer_start(app->display_timer, notification_settings_display_off_delay_ticks(app));
}
```

这个时钟最后会关闭背光
```c
// display timer
static void notification_display_timer(void* ctx) {
    furi_assert(ctx);
    NotificationApp* app = ctx;
    notification_message(app, &sequence_display_backlight_off);
}
```

```c
case NotificationMessageTypeLedDisplayBacklight:
	// 通知层 
	// if on - switch on and start timer
	// if off - switch off and stop timer
	// on timer - switch off
	if(notification_message->data.led.value > 0x00) {
		// 当前的背光亮度设置值 = 通知消息中的值 * 当前的背光亮度设置值
		notification_apply_notification_led_layer(
			&app->display,
			notification_message->data.led.value * display_brightness_setting);
		reset_mask |= reset_display_mask; // 设置标志位
	} else {
		// 如果通知消息中的值为 0，则表示关闭背光
		reset_mask &= ~reset_display_mask;
		// 复位
		notification_reset_notification_led_layer(&app->display);
		if(furi_timer_is_running(app->display_timer)) {
			furi_timer_stop(app->display_timer);
		}
	}
	break;
case NotificationMessageTypeLedDisplayBacklightEnforceOn:
	// 强制开启, 内部层
	furi_check(app->display_led_lock < UINT8_MAX);
	app->display_led_lock++;
	if(app->display_led_lock == 1) {
		notification_apply_internal_led_layer(
			&app->display,
			notification_message->data.led.value * display_brightness_setting);
	}
	break;
```

### led控制

底层的led使用的是**LP5562**进行驱动, 德州仪器（TI）的一款 **4 通道 LED 驱动芯片**，通过 **I²C** 总线控制：
- 4 个独立通道：红、绿、蓝、白（Flipper 用了 RGB 三个）。
- 每通道可设**电流**（限制最大亮度/校准颜色）和**PWM 值**（实时亮度 0-255）。
- 内置**程序引擎**，可以让 LED 硬件自动执行闪烁/渐变，不占用 CPU。

LP5562 内置 3 个程序引擎，每个都能独立执行一段指令序列（PWM 设值、延时、渐变、循环）。它们能力一样，可以同时运行、各驱动不同通道。一个通道同一时刻只能被一个源（I²C 或某个引擎）驱动

| 引擎          | 用途                | 驱动的通道 | 作用                                     |
| ----------- | ----------------- | ----- | -------------------------------------- |
| **Engine1** | 背光渐变（ramp，平滑淡入淡出） | 背光    | 用 `lp5562_execute_ramp` 让白通道从旧值平滑过渡到新值 |
| **Engine2** | RGB 闪烁（blink）     | 红/绿/蓝 | 跟随 Engine2 的闪烁程序                       |
| **Engine3** | 未使用               | —     | —                                      |