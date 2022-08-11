---
layout: post
title: "磁盘管理" 
date:   2022-8-11 15:39:08 +0800
tags: 嵌入式 原子Linux   
---

# 磁盘管理

Linux大部分不支持NTFS, 最好使用FAT格式的文件系统

在/dev文件中一般为`sd*`文件

`sd*n`表示第n个分区

## `df`命令

显示磁盘使用

## du命令

查看文件的大小

## 挂载卸载

一般情况自动挂载

### 取消

`umount`: 取消挂载

在取消挂载之前要从启动器解锁才能取消挂载

```bash
umount 挂载的地址/分区
```

挂载的地址为/media/用户名/...

分区是`sd*n`

### 挂载

创建用来挂载的文件夹

```
mount /dev/设备 挂载点
```

挂载的时候有可能中文出现乱码, 加上`-o iocharset=utf8`

## 分区

`fdisk`命令

>   -l(小写L): 查看所有分区

```bash
sudo fdisk 设备
jiao@jiao-virtual-machine:/media/jiao$ sudo fdisk /dev/sdb
```

在挂载的时候不能进行修改的操作

## 创建系统

```
mkfs -t vfat 设备
```

























