---
layout: post
title: "和我一起写Makefile" 
date:   2022-8-12 15:39:08 +0800
tags: 嵌入式 原子Linux    
---

# 和我一起写`Makefile`

## `Makefile`介绍

### 自动推导

make 看到一个[.o]文件，它就会自动的把[.c]文件加在依赖关系中

并且 `cc -c  whatever.c` 也会被推导出来

```makefile
main.o : main.c defs.h 
cc -c main.c
# 使用自动化推导后
main.o : defs.h
```



### 收拢

```makefile
objects = main.o kbd.o command.o display.o \ 
insert.o search.o files.o utils.o

$(objects) : defs.h 
kbd.o command.o files.o : command.h 
display.o insert.o search.o files.o : buffer.h 
```

定义文件关联文件进行收拢

## `Makefile`综述

### `Makefile`中有什么

+   显式规则: 如何生成一个或者多个文件
+   隐晦规则: 使用自动推导, 简略书写
+   变量的定义: 变量一般都是字符串, 有点类似于宏
+   文件指示: 包括三部分, 引用其他的`Makefile`, 另一种是指定有效的部分, 还有多行命令
+   注释: 只有单行注释, #

### 文件名

默认的情况下，make 命令会在当前目录下按顺序找寻文件名为`“GNUmakefile”、 “makefile”、“Makefile”`的文件

有另外一些 make 只对全小 写的`“makefile”`文件名敏感

你 可 以 使 用 别 的 文 件 名 来 书 写 `Makefile` ， 比 如 ： `“Make.Linux” ， “Make.Solaris”，“Make.AIX”`等，如果要指定特定的` Makefile`，你可以使用 make 的 “-f”和“--file”参数，如`：make -f Make.Linux `或 `make --file Make.AIX`

### 引用其他的`Makefile`

使用 include 关键字可以把别的 `Makefile` 包含进来

```makefile
include <filename> 
```

filename 可以是当前操作系统 Shell 的文件模式（可以保含路径和通配符）

前面可以有一些空字符，但是绝不能是[Tab]键开始

你有这样几个 Makefile：a.mk、b.mk、c.mk，还有一个文件叫` foo.make`，以及一个变变量$(bar)，其包含了 e.mk 和 f.mk

```makefile
include foo.make *.mk $(bar) 
等价于： 
include foo.make a.mk b.mk c.mk e.mk f.mk
```

+   如果文件都没有指定绝对路径或是相对路径的话， make 会在当前目录下首先寻找

+   如果 make 执行时，有“-I”或“--include-dir”参数，那么 make 就会在这个参数  所指定的目录下去寻找

+   如果目录/include（一般是：`/usr/local/bin` 或`/usr/include`）存在的话， make 也会去找。

如果你想让 make 不理那些无法读取的文件，而继续执行，你可以在 include 前加一个减号“-”

### 环境变量MAKEFILES

make 会把这个变量中的值做一个类 似于 include 的动作。这个变量中的值是其它的 Makefile

这个变量中的值是其它的` Makefile`，用空格分隔

从这个环境变中引入的` Makefile `的“目标”不会起作用，如果环境变量中定义 的文件发现错误，make 也会不理

**建议不要使用这个环境变量**

### make工作方式

1.   读入所有的 `Makefile`。  
2.   读入被 include 的其它` Makefile`。  
3.   初始化文件中的变量。  
4.   推导隐晦规则，并分析所有规则。  
5.   为所有的目标文件创建依赖关系链。  
6.   根据依赖关系，决定哪些目标要重新生成。  
7.   执行生成命令。

make 使用的是拖延战术，如 果变量出现在依赖关系的规则中，那么仅当这条依赖被决定要使用了，变量才会在其内部展 开。

## 书写规则

第一条规则中的目标将被确立为 最终的目标。如果第一条规则中的目标有很多个，那么，第一个目标会成为最终的目标。make 所完成的也就是这个目标。

### 规则的语法

```makefile
targets : prerequisites 
command 
...
```

或者

```makefile
targets : prerequisites ; command 
command 
...
```

>   如果其不与`“target:prerequisites”`在一行，那么，必须以[Tab 键]开头，如果和 prerequisites 在一行，那么可以用分号做为分隔。

如果命令太长，你可以使用反斜框（‘\’）作为换行符

一般来说，make 会以 UNIX 的标准 Shell，也就是/bin/sh 来执行命令

### 在规则中使用通配符

make 支持 三各通配符：

+   *
+   ?
+   ...

波浪号（“~”）表示当前用户的$HOME 目录

通配符代替了你一系列的文件，如“*.c”表示所以后缀为 c 的文件。

```makefile
objects = *.o
```

>   通符同样可以用在变量中。并不是说[*.o]会展开，不！objects 的值就是\*.o

如果你要让通配符在变量中 展开，也就是让 objects 的值是所有[.o]的文件名的集合

```makefile
objects := $(wildcard *.o)
```

### 文件搜索

当 make 需要去找寻文件的依赖关系时，你可以在文件前加上路 径，但最好的方法是把一个路径告诉 make，让 make 在自动去找

`Makefile `文件中的特殊变量“VPATH”就是完成这个功能的，如果没有指明这个变量， make 只会在当前的目录中去找寻依赖文件和目标文件, 目录由“冒号”分隔

```makefile
VPATH = src:../headers
```

另一个设置文件搜索路径的方法是使用 make 的``“vpath”``关键字（注意，它是全小写 的），这不是变量，这是一个 make 的关键字

+   `vpath <pattern> <directories>` 为符合模式`<pattern>`的文件指定搜索目录`<direction>`
+   `vpath <pattern> ` 清除符合模式`<pattern>`的文件的搜索目录
+   `vapth` 清除所有已被设置好了的文件搜索目录

`vapth `使用方法中的`<pattern>`需要包含“%”字符。“%”的意思是匹配零或若干字符， 例如，“%.h”表示所有以“.h”结尾的文件。`<directories>`指定了要搜索的文件集，而 则指定了的文件集的搜索的目录。

我们可以连续地使用 `vpath` 语句，以指定不同搜索策略。如果连续的 `vpath` 语句中出现 了相同的`<pattern>`，或是被重复了的`<pattern>`，那么，make 会按照` vpath` 语句的先后顺 序来执行搜索

### 伪目标

向 make 说明，不管是否有这个文件，这个目标就是“伪 目标”。

伪目标一般没有依赖的文件。但是，我们也可以为伪目标指定所依赖的文件。伪目标同 样可以作为“默认目标”，只要将其放在第一个

如果你的 Makefile 需要 一口气生成若干个可执行文件，但你只想简单地敲一个 make 完事，并且，所有的目标文件 都写在一个 Makefile 中，那么你可以使用“伪目标”这个特性

```makefile
all : prog1 prog2 prog3 
.PHONY : all

prog1 : prog1.o utils.o 
cc -o prog1 prog1.o utils.o 
...
```

>   由于伪目标的特性是，总是被执行的，所以其依赖的那三 个目标就总是不如“all”这个目标新。所以，其它三个目标的规则总是会被决议。也就达 到了我们一口气生成多个目标的目的。

目标也可以成为依赖。所以，伪目标同样也 可成为依赖。

### 多目标

目标可以不止一个，其支持多目标，有可能我们的多个目标同时依赖 于一个文件，并且其生成的命令大体类似。于是我们就能把其合并起来。

### 静态模式

```
<targets ...>: <target-pattern>: <prereq-patterns ...> 
```

targets 定义了一系列的目标文件，可以有通配符。是目标的一个集合。 

 `target-parrtern` 是指明了 targets 的模式，也就是的目标集模式。  

`prereq-parrterns` 是目标的依赖模式，它对 `target-parrtern` 形成的模式再进行一次依赖 目标的定义。 

```makefile
$(SOBJS) : obj/%.o : %.s
	$(CC) -Wall -nostdlib -c -O2 $(INCLUDE) -o $@ $< 
```

>   这里是从$(SOBJS)中找出所有的符合条件的文件, 然后生成依赖文件

## 通用模板

```makefile
CROSS_COMPILE 	?= arm-linux-gnueabihf-
TARGET 			?= ledc

CC 				:= $(CROSS_COMPILE)gcc
LD 				:= $(CROSS_COMPILE)ld
OBJCOPY 		:= $(CROSS_COMPILE)objcopy
OBJDUMP 		:= $(CROSS_COMPILE)objdump


INCUDIRS		:= 	imx6u \
					bsp/clk \
					bsp/led \
					bsp/delay \
					project


SRCDIRS			:= 	project	\
					bsp/clk \
					bsp/led \
					bsp/delay
# 给所有的头文件加上-I, 用于在编译的时候添加头文件路径
INCLUDE 		:= $(patsubst %, -I %, $(INCUDIRS))
# 寻找筛选c和s文件,遍历列表, 找出复合条件的文件
SFILES			:= $(foreach dir, $(SRCDIRS), $(wildcard $(dir)/*.s) )
CFILES			:= $(foreach dir, $(SRCDIRS), $(wildcard $(dir)/*.c) )
# 保存生成的文件
# 先生成没有路径的文件名
SFILENDIR 		:= $(notdir $(SFILES))
CFILENDIR 		:= $(notdir $(CFILES))
# 对生成的中间文件进行添加目录
SOBJS			:= $(patsubst %, obj/%, $(SFILENDIR:.s=.o))
COBJS			:= $(patsubst %, obj/%, $(CFILENDIR:.c=.o))

OBJS			:= $(SOBJS) $(COBJS)
# 指定寻找的路径, 这是系统变量, 找文件的时候会从这些路径寻找
VPATH 			:=$(SRCDIRS)
.PHONY: clear

$(TARGET).bin : $(OBJS)
	# 生成带有编译信息的文件
	$(LD) -Timx6u.lds -o $(TARGET).elf $^
	# 生成目标文件
	$(OBJCOPY) -O binary -S $(TARGET).elf $@
	# 生成反汇编文件
	$(OBJDUMP) -D -m arm $(TARGET).elf > $(TARGET).dis
# 这里是用静态模式，第一项是一系列的文件，第二个是目标文件遵守的格式，第三个是依赖文件的格式
# 加入头文件的路径 
$(SOBJS) : obj/%.o : %.s
	$(CC) -Wall -nostdlib -c -O2 $(INCLUDE) -o $@ $< 
# 加入头文件的路径 
$(COBJS) : obj/%.o : %.c

	$(CC) -Wall -nostdlib -c -O2 $(INCLUDE) -o $@ $< 

clear : 
	rm -rf $(TARGET).elf $(TARGET).bin $(TARGET).dis $(OBJS)

```

















