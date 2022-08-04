---
layout: post
title: "安装软件程序《Linux命令行与Shell脚本编程大全》" 
date:   2022-8-4 15:03:08 +0800
tags: 《Linux命令行与Shell脚本编程大全》 Linux 
---

# 安装软件程序

包管理系统（package management system，PMS）

## 包管理基础

主流的Linux都采用了某种形式的包管理系统来控制软件和库的安装, PSM利用一个数据库来记录各种相关的内容:

+ Linux上已安装了什么软件包
+ 每个包安装了什么文件
+ 每个已安装的软件包版本

软件包存储在服务器上，可以利用本地Linux系统上的PMS工具通过互联网访问。这些服务器称为仓库（repository）。可以用PMS工具来搜索新的软件包，或者是更新系统上已安装软件包

软件包通常会依赖其他的包，为了前者能够正常运行，被依赖的包必须提前安装在系统中。PMS工具将会检测这些依赖关系，并在安装需要的包之前先安装好所有额外的软件包

PMS没有统一的标准工具, 在不同Linux版本上有很大不同, 常见的有dpkg和rpm

基于Debian的发行版（如Ubuntu和Linux Mint）使用的是dpkg命令，这些发行版的PMS工具也是以该命令为基础的。dpkg会直接和Linux系统上的PMS交互，用来安装、管理和删除软件包

基于Red Hat的发行版（如Fedora、openSUSE及Mandriva）使用的是rpm命令，该命令是其PMS的底层基础。类似于dpkg命令，rmp命令能够列出已安装包、安装新包和删除已有软件

## 基于Debine的系统

dpkg命令是基于Debian的PMS工具的核心

+ apt-get
+ apt-cache
+ aptitude

最常用的是aptitude, 本质上是apt工具和dpkg的前端, dpkg是软件包管理系统工具，而aptitude则是完整的软件包管理系统

可以避免一系列的麻烦