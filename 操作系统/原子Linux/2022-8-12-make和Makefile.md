---
layout: post
title: "make和Makefile" 
date:   2022-8-12 15:39:08 +0800
tags: 嵌入式 原子Linux    
---

# make和Makefile

## make工具

自动完成编译工作

+   如果修改了源文件, 就只会再次编译修改了的文件
+   修改了头文件就会再次编译所有包含头文件的文件

需要Makefile文件

直接使用gcc编译会导致全部重复编译

## Makefile语法

```makefile
目标...: 依赖的文件集合
	命令1
	命令2
	...
```

```makefile
  1 main: main.o input.o calcu.o                                                          
  2     gcc main.o input.o calcu.o -o main
  3 main.o: main.c
  4     gcc -c main.c
  5 input.o: input.c
  6     gcc -c input.c
  7 calcu: calcu.c
  8     gcc -c calcu.c
  9 clear:
 10     rm *.0
 11     rm main

```

>   如果要更新目标文件, 所有的依赖文件都要更新, 依赖文件任何一个更新目标文件也必须更新

>   每一条命令以Tab开头

make命令会为每一个Tab开头的命令创建一个shell去执行

make的默认目标: 文件开始出现的第一个目标

make会使用当前文件夹下的Makefile文件进行, 按照定义的规则创建目标问文件, 发现目标文件不存在或者依赖的文件更新时间比目标文件昕就进行编译

### Makefile变量

只有字符串

```makefile
变量名=值
```

+   使用

```makefile
$(变量名)
```

还有两种赋值方法

+   :=
+   ?=

在使用shell命令的时候在前面加上@符号可以让不输出执行的命令

```bash
 14 name=jiao
 15 name2=$(name)
 16 name=haoyang
 17 print:
 18     echo name2= $(name2) 

jiao@jiao-virtual-machine:~/桌面/test/c_language$ make print
echo name2= haoyang 
name2= haoyang
jiao@jiao-virtual-machine:~/桌面/test/c_language$ vim Makefile   
# 在此处更改为    @echo name2= $(name2) 
jiao@jiao-virtual-machine:~/桌面/test/c_language$ make print
name2= haoyang
```



赋值符号=借助另一个变量可以把变量的真实值推迟定义, 真实值为引用的变量的最后一次有效值, 只有在被调用的时候才会被赋值

### 系统变量

```makefile
jiao@jiao-virtual-machine:~/桌面/test/c_language$ make print
echo "cc"		# 保存编译器
cc
echo "as"		# 保存汇编器
as
echo "make"		# 保存make工具
make
```

VPATH: 搜索的路径

### :=赋值符号

立即赋值, 这时候不会取最后一次修改的值

```makefile
 14 name=jiao
 15 name2:=$(name)
 16 name=haoyang
 17 print:
 18     @echo name2= $(name2) 

jiao@jiao-virtual-machine:~/桌面/test/c_language$ make print
name2= jiao
```



### ?=赋值符号

前面的变量没有赋值就进行赋值, 相当于等号, 反之不进行赋值

```makefile
 14 name=jiao
 15 name2?=$(name)
 16 name=haoyang
 17 print:
 18     @echo name2= $(name2)
 
 jiao@jiao-virtual-machine:~/桌面/test/c_language$ make print
name2= haoyang


 14 name2=kangkang
 15 name=jiao
 16 name2?=$(name)
 17 name=haoyang
 18 print:
 19     @echo name2= $(name2) 


jiao@jiao-virtual-machine:~/桌面/test/c_language$ make print
name2= kangkang

 
```



### +=赋值

追加

## 模式规则

模式规则中最少要包含%, 否则视为一般规则

>   % 相当于长度任意的非空字符创

当%出现在目标文件中的时候, 目标中的%决定了以来的%的值

```makefile
%.o : %.c
	命令
```

```makefile
%:
	echo "$@"
```

>   会匹配所有输入的命令并打印出来

## 自动化变量

实现从不同的依赖文件中生成对应的文件

|    自动化变量     |                       描述                       |
| :---------------: | :----------------------------------------------: |
|        $@         |    规则中的目标的集合, 有多个目标的话匹配所有    |
|        $%         | 目标是函数库时候表示规则中的目标成员名, 否则为空 |
|        $<         |      依赖文件集合第一个文件, %定义就是集合       |
|        $?         |              所有比目标文件新的文件              |
|        $^         |        所有依赖文件集合, 回去除重复的文件        |
|        $+         |             和$^相似, 但是不去除重复             |
|        $*         |             目标模式%以及之前的部分              |
| 常用\$@, \$<, \$^ |                                                  |

## 伪目标

不代表真正的目标名, 在执行make命令的时候通过指定伪目标执行其所在规则定义的命令

```makefile
  6 clear:
  7     rm *.o
  8     rm main
  
jiao@jiao-virtual-machine:~/桌面/test/c_language$ make clear
make: 'clear' is up to date.

```

>   如果文件夹下有一个和clear同名的文件, 由于没有依赖文件, 会被认定为更新到最新的文件, 不会执行这个命令

+   声明为伪目标

```makefile
.PHONY: clean
```

```makefile
  6 .PHONY: clear                   
  7 clear:
  8     rm *.o
  9     rm main


jiao@jiao-virtual-machine:~/桌面/test/c_language$ make clear
rm *.o
rm main
```

## 条件判断

```makefile
<条件关键字>
	<条件为真的时候>
else
	<条件为加的时候>
endif
```

条件关键字有四个

+   ifeq

```makefile
ifeq (<参数1>, <参数2>)
ifeq '<参数1>', '<参数2>'
ifeq "<参数1>", "<参数2>"
```

+   ifneq
+   ifdef

```makefile
ifdef<变量名>
```

+   ifndef



## 函数使用

```makefile
$(函数名 参数集合)
```

```makefile
${函数名 参数集合}
```

只能使用支持的函数, 不能自定义

### 常用的函数

+   patsubst: 模式替换函数, 有三个函数

$(patsubst PATTERN, REPATTERN, TEXT)

在第一个参数匹配的内容中, 如果匹配的内容出现在TEXT中, 就会换成第二个参数

```makefile
$(patsubst %.c, %.o, x.c bar.c)
```

>   因为x.c或者bar.c匹配%.c所以会被替换为x.o或bar.o

+   notdir: 取文件名(删除路径)

```makefile
$(notdir 路径或文件名)
```

+   wildcard: 获取匹配的文件名

```makefile
$(wildcard *.c)
```

>   提取当前路径所有c文件

+   foreach

$(foreach VAR, LIST, TEXT)

第二个参数是一系列的文本, 遍历LIST, 赋值给VAR, 然后执行TEXT表达式

```makefile
dirs:=a b c d
files := $(foreach dir,$(dirs),$(wildcard $(dir)/*) )
```

>   获取四个文件夹下所有的文件名



## Makefile头文件依赖

先写一个头文件, 添加到头文件目录, 实施检查头文件依赖

使用替换函数生成头文件路径, 然后用-I参数添加到gcc编译器的头文件路径

要把头文件放在编译的文件的依赖文件上

## 示例

```bash
  1 ARCH ?= x86
  2 
  3 ifeq ($(ARCH),x86)
  4     CC=gcc
  5 else
  6     CC=arm-linux-gnueabihf-gcc
  7 endif
  8 
  9 
 10 TARGET=mp3
 11 BUILD_DIR=build 
 12 SRC_DIR=module1 module2
 13                                                                                       
 14 SOURCES=$(foreach dir,$(SRC_DIR),$(wildcard $(dir)/*))    # 获取到所有的源码文件
 15 OBJS=$(patsubst %.c,$(BUILD_DIR)%.o,$(notdir $(SOURCES)))		# 根据源码文件创建目标文件名
 16 VPATH=$(SRC_DIR)
 17 
 18 $(BUILD_DIR)/$(TARGET):$(OBJS)#
 19     $(CC) $^ -o $@
... 
```











