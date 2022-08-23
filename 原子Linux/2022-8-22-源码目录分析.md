---
layout: post
title: "源码目录分析" 
date:   2022-8-22 15:39:08 +0800
tags: 嵌入式 原子Linux 
---

# 源码目录分析

由于`uboot`会使用编译才会生成的文件, 所以在分析的时候需要编译一下

## 文件夹

|   名字   |              描述              |
| :------: | :----------------------------: |
|   api    |       硬件无关的API函数        |
|   arch   | 架构相关的代码, 针对不同的架构 |
|  board   |        开发板相关的文件        |
|   cmd    |         命令相关的文件         |
|  common  |            通用代码            |
| configs  |            配置文件            |
|   disk   |        磁盘分区相关代码        |
|   doc    |              文档              |
| drivers  |            驱动代码            |
|   dts    |             设备树             |
| examples |            示例代码            |
|    fs    |            文件系统            |
| include  |             头文件             |
|   lib    |             库文件             |
| Licenses |        许可证相关的文件        |
|   net    |            网络相关            |

>   在文件`arch->arm->CPU->u-boot.lds`就是整个`uboot`的链接脚本, 确定每一块内容的位置, 在编译过后会出现一个在主目录下的相同文件

>   在board文件夹下的`freescale`文件下的mx6slevk就是我们使用的芯片

>   configs是默认的配置文件下, 都使用`defconfig`结尾的, 根据不同的板子或平台进行配置, 一定要标志好自己的平台, 在编译的时候要指定默认的配置文件

+   在移植的时候, 重点关注broad, 根据参考进行修改
+   configs设置不同的配置
+   之后的都是通用的文件

## 文件

>   .config在我们执行make xxx_defconfig之后会生成的文件, 保存了详细的配置信息
>
>   ```
>   CONFIG_CREATE_ARCH_SYMLINK=y
>   CONFIG_HAVE_GENERIC_BOARD=y
>   CONFIG_SYS_GENERIC_BOARD=y
>   # CONFIG_ARC is not set
>   CONFIG_ARM=y
>   ```
>
>   Makefile文件中会使用在这里配置为信息
>
>   ```
>   # This selects which instruction set is used.
>   arch-$(CONFIG_CPU_ARM720T)	=-march=armv4
>   arch-$(CONFIG_CPU_ARM920T)	=-march=armv4t
>   arch-$(CONFIG_CPU_ARM926EJS)	=-march=armv5te
>   arch-$(CONFIG_CPU_ARM946ES)	=-march=armv4
>   arch-$(CONFIG_CPU_SA1100)	=-march=armv4
>   ```
>
>   使用这样的形式, 最后定义的变量名字最后是一个y

>   .u-boot.cmd文件, 相关的命令

>   Kconfig图形界面相关的

>   顶层Makefile最重要的Makefile

>   顶层READEME最重要的README文件

>   systemmap映射, 不同地址的内容

>   u-boot这个就是有elf信息的uboot可执行文件
>
>   >   u-boot：编译出来的ELF格式的uboot镜像文件。
>   >
>   >   u-boot.bin：编译出来的二进制格式的uboot可执行镜像文件。
>   >
>   >   u-boot.cfg：uboot的另外一种配置文件。
>   >
>   >   u-boot.imx：u-boot.bin添加头部信息以后的文件，NXP的CPU专用文件。
>   >
>   >   u-boot.lds：链接脚本。
>   >
>   >   u-boot.map：uboot映射文件，通过查看此文件可以知道某个函数被链接到了哪个地址上。
>   >
>   >   u-boot.srec：S-Record格式的镜像文件。
>   >
>   >   u-boot.sym：uboot符号文件。
>   >
>   >   u-boot-nodtb.bin：和u-boot.bin一样，u-boot.bin就是u-boot-nodtb.bin的复制文件。

## 启动流程

配置vscode

设定搜索以及显示的文件

新建文件夹.vscode, 在文件夹下建立settings.json文件

```
{
    "search.exclude": {
        "**/node_modules": true,
        "**/bower_components": true,
    },
    "files.exclude": {
        "**/.git": true,
        "**/.svn": true,
        "**/.hg": true,
        "**/CVS": true,
        "**/.DS_Store": true,
        "**/Thumbs.db": true
    }
}
```

### 









































