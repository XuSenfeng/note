---
layout: post
title: "顶层Makefile" 
date:   2022-8-22 15:39:08 +0800
tags: 嵌入式 原子Linux 
---

# 顶层Makefile

## 预处理

### 版本信息

最上面的是版本号

### MAKEFLAGS变量

在执行子目录下的Makefile文件的时候, 在编译的时候主目录的Makefile可以调用子目录的Makefile

主目录的Makefile可以使用如下代码来编译这个子目录

```
$(MAKE) -C subdir
```

调用make命令, 使用-C指定子目录, 可以使用用“export”来导出要传递给子make的变量, 不希望哪个变量传递给子make的话就使用“unexport”来声明不导出

有两个特殊的变量：“SHELL”和“MAKEFLAGS”，这两个变量除非使用“unexport”声明，否则的话在整个make的执行过程中，它们的值始终自动的传递给子make

```makefile
MAKEFLAGS +=-rR --include-dir=$(CURDIR)
```

“-rR”表示禁止使用内置的隐含规则和变量定义，“--include-dir”指明搜索路径，”$(CURDIR)”表示当前目录

### 命令的输出

uboot默认编译是不会在终端中显示完整的命令，都是短命令,“V=1“来实现完整的命令输出

```
73	ifeq ("$(origin V)","command line")
74		KBUILD_VERBOSE =$(V)
75	endif
76	ifndef KBUILD_VERBOSE
77		KBUILD_VERBOSE =0
78	endif
79
80	ifeq ($(KBUILD_VERBOSE),1)
81		quiet =
82		Q =
83	else
84		quiet=quiet_
85		Q = @
86	endif
```

函数origin用于告诉你变量是哪来的, `$(origin <variable>)`, 返回值就是变量来源，因此$(origin V)就是变量V的来源。如果变量V是在命令行定义的那么它的来源就是"command line"

可以通过在命令之前加@控制是否输出命令

```
$(Q)$(MAKE) $(build)=tools
```

```
quiet_cmd_sym ?= SYM          $@
cmd_sym ?= $(OBJDUMP) -t $< > $@
```

如果变量quiet为空的话，整个命令都会输出。

如果变量quiet为“quiet_”的话，仅输出短版本。_

如果变量quiet为“silent_”的话，整个命令都不会输出。

### 静默输出

编译的时候使用“make -s”即可实现静默输出

```
$(filter <pattern...>,<text>)
```

filter函数表示以pattern模式过滤text字符串中的单词，仅保留符合模式pattern的单词

>   函数firstword是获取首单词

```makefile
$(firstword <text>)
```

>   于取出text字符串中的第一个单词

```makefile
91	ifneq ($(filter 4.%,$(MAKE_VERSION)),)# make-4
92	ifneq ($(filter %s ,$(firstword x$(MAKEFLAGS))),)
93		quiet=silent_
94	endif
95	else# make-3.8x
96	ifneq ($(filter s%-s%,$(MAKEFLAGS)),)
97		quiet=silent_
98	endif
```

这几个参数会传递给子Makefile

### 设置编译结果输出目录

```
O=目录名
```

### 设置代码检查

```
C=数字
```

### 设置模块编译

```
make M=dir
```

这里的结果是为当前目录

### 获取主机架构

使用命令uname -m获取, 使用替换命令进行替换, 

使用uname -s获取当前的系统, “tr '[:upper:]' '[:lower:]'”表示将所有的大写字母替换为小写字母

### 设置编译器和交叉编译期

这里需要参数指定, 也可以自行添加到文件中

### 进行变量的处理

导入文件config.mk的变量, 然后对文件中的变量进行导出, 他的变量来源是.config文件, 对变量进行提取

config.mk对对应的CPU架构的文件夹下的config.mk进行读取, 以及对对应的板子下面的config.mk进行读取, 然后进行处理

## 编译过程分析

首先调用distclean, 清除一些文件

make xxx_defconfig, 有两个依赖的命令, 第二个命令为空, 第一个命令是为了编译出来一个软件, 他的命令就是用来生成另一个软件并利用软件生成一个.config文件

make命令, 根据之前编译出来的内容生成对应的文件

### 链接脚本

默认为u-boot.lds

uboot链接的首地址为0x87800000, 定义在芯片的初始化定义中



