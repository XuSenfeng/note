---
layout: post
title: "uboot图形化配置文件" 
date:   2022-8-23 15:39:08 +0800
tags: 嵌入式 原子Linux 
---

# uboot图形化配置界面

通过终端配置

进入到源码根目录, 首先默认配置`make mx6ull_jiao_emmc_defconfig`

输入`make menuconfig`, 在打开之前要安装build-essential, libncurses5, libncurses5-dev

图形化配置界面对于一个功能有三种模式, y对应的功能编译, n不编译进uboot,m对应的功能编译为模块,

使用斜杠进行搜索, 

## Kconfig文件

图形化界面配置文件, 







