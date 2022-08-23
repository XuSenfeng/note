---
layout: post
title: "Linux内核" 
date:   2022-8-23 15:39:08 +0800
tags: 嵌入式 原子Linux 
---

# Linux内核

默认配置文件保存在arch/arm/configs文件下

```
/bin/sh: 1: lzop: not found
arch/arm/boot/compressed/Makefile:180: recipe for target 'arch/arm/boot/compressed/piggy.lzo' failed
make[2]: *** [arch/arm/boot/compressed/piggy.lzo] Error 1
make[2]: *** 正在等待未完成的任务....
  CC      arch/arm/boot/compressed/misc.o
arch/arm/boot/Makefile:52: recipe for target 'arch/arm/boot/compressed/vmlinux' failed
make[1]: *** [arch/arm/boot/compressed/vmlinux] Error 2
arch/arm/Makefile:316: recipe for target 'zImage' failed
make: *** [zImage] Error 2
make: *** 正在等待未完成的任务....
```

出现这个安装lzop

zImage存放在 ,./arch/arm/boot文件夹, 设备树存放在./arch/arm/boot/dts文件夹

编译单个的dts文件, 直接在顶层目录make对应的.dtb文件就行了

## 来源

内核官方, www.kernel.org, 有各个额版本的

NXP会挑选一个重点维护其中一个, 相当于官方的BSP或者说SDK供用户使用

正点原子使用NXP的Linux添加修改过的内核给用户时候用, 讲解的是NXP的内核

## 目录分析

+   arch: 架构相关zImage存放在/arch/arm/boot文件夹, 设备树存放在./arch/arm/boot/dts文件夹

+   bllock块设备相关, emmc等
+   crypto加密相关
+   Documentation:文档相关目录, 最常用的是devicetree/bindings这个目录下的描述设备树绑定的信息
+   firmeare: 固件相关的目录
+   fs: 文件系统相关
+   include: 头文件
+   init: 初始化相关
+   ipc: 进程之间通信
+   kernel: 内核相关
+   lib: 库相关
+   mm: 内存管理
+   net: 网络相关
+   samples: 例程
+   scripts: 脚本
+   security: 安全相关
+   sound: 音频相关
+   tools: 工具
+   urs: 
+   virt: 虚拟化



生成的vmlinux, Image, zImage, uImage区别

>    vmlinux生成的最初的文件, 没有压缩, 很大

>   Image内核镜像文件, 取消掉一些符号表之类的信息

>   zImage: 使用gzip压缩的文件

>   uImage: 老板的uboot专用的镜像, 多加了一个头部



















