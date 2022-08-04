---
layout: post
title: "没有release文件" 
date:   2022-8-4 15:03:08 +0800
tags: 问题处理  Linux 
---

在`sudo apt-get update `

[处理方案来源][http://t.csdn.cn/ZkiXF]



```bash
错误:8 http://cn.archive.ubuntu.com/ubuntu impish-backports Release
  404  Not Found [IP: 91.189.91.39 80]
正在读取软件包列表... 完成
E: 仓库 “http://security.ubuntu.com/ubuntu impish-security Release” 不再含有 Release 文件。
N: 无法安全地用该源进行更新，所以默认禁用该源。
N: 参见 apt-secure(8) 手册以了解仓库创建和用户配置方面的细节。
E: 仓库 “http://cn.archive.ubuntu.com/ubuntu impish-updates Release” 没有 Release 文件。
N: 无法安全地用该源进行更新，所以默认禁用该源。
N: 参见 apt-secure(8) 手册以了解仓库创建和用户配置方面的细节。
E: 仓库 “http://cn.archive.ubuntu.com/ubuntu impish Release” 没有 Release 文件。
N: 无法安全地用该源进行更新，所以默认禁用该源。
N: 参见 apt-secure(8) 手册以了解仓库创建和用户配置方面的细节。
E: 仓库 “http://cn.archive.ubuntu.com/ubuntu impish-backports Release” 不再含有 Release 文件。
N: 无法安全地用该源进行更新，所以默认禁用该源。
N: 参见 apt-secure(8) 手册以了解仓库创建和用户配置方面的细节
```



```bash
sudo vim /etc/apt/sources.list

改为
deb http://old-releases.ubuntu.com/ubuntu/ cosmic main restricted universe multiverse
deb http://old-releases.ubuntu.com/ubuntu/ cosmic-security main restricted universe multiverse
deb http://old-releases.ubuntu.com/ubuntu/ cosmic-updates main restricted universe multiverse
deb http://old-releases.ubuntu.com/ubuntu/ cosmic-proposed main restricted universe multiverse
deb http://old-releases.ubuntu.com/ubuntu/ cosmic-backports main restricted universe multiverse
deb-src http://old-releases.ubuntu.com/ubuntu/ cosmic main restricted universe multiverse
deb-src http://old-releases.ubuntu.com/ubuntu/ cosmic-security main restricted universe multiverse
deb-src http://old-releases.ubuntu.com/ubuntu/ cosmic-updates main restricted universe multiverse
deb-src http://old-releases.ubuntu.com/ubuntu/ cosmic-proposed main restricted universe multiverse
deb-src http://old-releases.ubuntu.com/ubuntu/ cosmic-backports main restricted universe multiverse

# 之后更新
错误:1 http://old-releases.ubuntu.com/ubuntu cosmic InRelease
  由于没有公钥，无法验证下列签名： NO_PUBKEY 3B4FE6ACC0B21F32
获取:2 http://old-releases.ubuntu.com/ubuntu cosmic-security InRelease [88.7 kB]
错误:2 http://old-releases.ubuntu.com/ubuntu cosmic-security InRelease
  由于没有公钥，无法验证下列签名： NO_PUBKEY 3B4FE6ACC0B21F32
获取:3 http://old-releases.ubuntu.com/ubuntu cosmic-updates InRelease [88.7 kB]
错误:3 http://old-releases.ubuntu.com/ubuntu cosmic-updates InRelease
  由于没有公钥，无法验证下列签名： NO_PUBKEY 3B4FE6ACC0B21F32
获取:4 http://old-releases.ubuntu.com/ubuntu cosmic-proposed InRelease [92.5 kB]
错误:4 http://old-releases.ubuntu.com/ubuntu cosmic-proposed InRelease                                                 
  由于没有公钥，无法验证下列签名： NO_PUBKEY 3B4FE6ACC0B21F32
获取:5 http://old-releases.ubuntu.com/ubuntu cosmic-backports InRelease [74.6 kB]                                      
错误:5 http://old-releases.ubuntu.com/ubuntu cosmic-backports InRelease                                                
  由于没有公钥，无法验证下列签名： NO_PUBKEY 3B4FE6ACC0B21F32







# 出现没有公钥
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 公钥
# 示例 sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 3B4FE6ACC0B21F32

```

