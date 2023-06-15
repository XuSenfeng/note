---
layout: post
title: "链接文件" 
date:   2022-8-12 15:39:08 +0800
tags: 嵌入式 原子Linux   
---

# 链接文件

符号链接, 就是快捷方式

硬链接, 有关`inode`, 相当于文件的新的ID

## 创建软连接

软 连接==>硬链接\==>`inode`

+   创建软连接

```bash
ln -s 文件名
```

可以连接到目录, 可以跨文件系统, 符号链接要使用绝对路径创建, 否则不能进行移动,直接对文件进行复制的时候会把源文件一同复制

```bash
jiao@jiao-virtual-machine:~/桌面$ cp hello1 test/
jiao@jiao-virtual-machine:~/桌面$ cd test
jiao@jiao-virtual-machine:~/桌面/test$ ls
a.c  b.c  hello1
jiao@jiao-virtual-machine:~/桌面/test$ ls -l
总用量 12
-rw-rw-r-- 1 jiao jiao    0 8月  11 23:56 a.c
-rw-rw-r-- 1 jiao jiao    0 8月  11 23:56 b.c
-rwxrwxr-x 1 jiao jiao 8600 8月  12 15:01 hello1
```

可以使用`cp -d` 进行不变属性的复制,但是相对路径会出错

## 创建硬链接

一般用来防止文件删除, 不能跨文件系统, 不能连接到目录

多个链接指向一个`inode`, 删除了所有连接文件才会删除源文件

```bash
ln 文件名
```

## 强制创建链接文件

```
ln -f 文件名
```

创建的文件存在先删除, 再创建



可以使用`ls -i`查看文件的`inode`

