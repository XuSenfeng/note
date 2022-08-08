---
layout: post
title: "图形化桌面环境中的脚本编程《Linux命令行与Shell脚本编程大全》" 
date:   2022-8-7 23:39:08 +0800
tags: 《Linux命令行与Shell脚本编程大全》 Linux     
---

# 图形化桌面中的脚本编程



## 创建文件菜单

### 创建菜单布局

默认情况下，echo命令只显示可打印文本字符。在创建菜单项时，非可打印字符通常也很有用，比如制表符和换行符。要在echo命令中包含这些字符，必须用-e选项

最后一行的-en选项会去掉末尾的换行符。这让菜单看上去更专业一些，光标会一直在行尾等待用户的输入

```bash
  1 #!/bin/bash
  2 clear
  3 echo -e "\t\t\tSys Admin Menu\n"
  4 echo -e "\t1. Display disk space"
  5 echo -e "\t2. Display logged on users" 
  6 echo -e "\t3. Display memory usage"
  7 echo -e "\t0. Exit menu\n\n" 
  8 echo -en "\t\tEnter option: "
  9 read -n 1 option

```



### 创建菜单函数

```bash
  1 #!/bin/bash                                                                           
  2 function diskspace {
  3     clear
  4     df -h
  5 }
  6 function whoseon {
  7     clear
  8     who
  9 }
 10 function menusage {
 11     clear
 12     cat /proc/meminfo
 13 }
 14 function diskspace {
 15     clear
 16     echo -e "\t\t\tSys Admin Menu\n"
 17     echo -e "\t1. Display disk space"
 18     echo -e "\t2. Display logged on users" 
 19     echo -e "\t3. Display memory usage"
 20     echo -e "\t0. Exit menu\n\n" 
 21     echo -en "\t\tEnter option: "
 22     read -n 1 option
 23 }
 24 while [ 1 ]
 25 do
 26     diskspace
 27     df -h
 28     case $option in
 29     0)
 30         break;;
 31     1) 
 32         diskspace;;
 33     2) 
 34         whoseon;;
 35     3)
 36         menusage;;
 37     *) 
 38         clear
 39         echo "Sorry, wrong selection"
 40     esac
 41     echo -en "\n\n\t\t\tHit any key to continue"
 42     read -n 1 line
 43 done                                              
 44 clear

```































