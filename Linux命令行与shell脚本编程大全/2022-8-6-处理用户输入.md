---
layout: post
title: "处理用户输入《Linux命令行与Shell脚本编程大全》" 
date:   2022-8-6 15:39:08 +0800
tags: 《Linux命令行与Shell脚本编程大全》 Linux 
---

# 处理用户输入

bash shell提供了一系列的方法从用户处获取参数

## 命令行参数

最基本的方法, 在使用命令行的时候传递参数

### 读取参数

bash shell会将一些称为**位置参数**的特殊变量分配给命令行中的所有参数, 包括shell的脚本名

位置参数的变量名是标准的数字: \$0 程序名, \$1 第一个参数, $9: 最后一个参数

```bash
  1 #!/bin/bash                                                                           
  2 factorial=1
  3 for ((number=1;number<=$1;number++))
  4 do
  5     factorial=$[$factorial*$number]
  6 done
  7 echo The factorial of $1 is $factorial
```

>   在输入的是字符串的时候, 如果一个值之间有空格, 要用引号引起来

>   如果脚本的参数不止九个, 可以使用大括号把数字引起来 `${10}`



