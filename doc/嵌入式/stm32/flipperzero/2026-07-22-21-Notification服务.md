
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