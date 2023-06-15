---
layout: post
title: "Linux自带的led驱动" 
date:   2022-9-2 15:39:08 +0800
tags: 嵌入式 原子Linux
---

# Linux自带的led驱动

一般自带的驱动都是需要使用图形界面进行配置

图形界面-->Device Driver-->LED Support --> LED Support for GPIO connected LEDs 

使能驱动之后在在文件.config中存在`CONFIG_LEDS_GPIO=y`, 会把驱动文件编译到内核之中 



## 驱动分析

使用标准的platform-driver

```c
module_platform_driver(gpio_led_driver);
```

>   使用一个宏完成函数的注册, 代替module_init和module_exit函数, 根据参数生成两个函数, 分别调用platform_driver_register, 和platform_driver_unregister, 使用宏定义完成之前写的函数

匹配的设备`{ .compatible = "gpio-leds", }`, 

驱动设备匹配之后会执行probe函数, 根据有没有设备树进行处理



## 使用

首先把驱动编译进内核里面, 根据文档设备树里面添加节点的信息

如果没有设备树, 需要使用platform_device_register向总线注册一个设备, 有设备树的话, 直接修改设备树, 添加对应的节点, 文档的目录在`./document/device-tree/bindings/leds/leds-gpio.txt`,  









































