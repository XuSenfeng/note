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

# 处理方法2

[处理方法2][http://t.csdn.cn/7osap]

输入命令` sudo apt-get update`, 报错



这里可以看到有两个问题，一个是 ubuntu自己的源连不上了[第二三个红框框] ，一个是 `vmware `这个软件 [第一个红框框]。

首先解决第一个问题。archive.ubuntu.com是ubuntu的默认源，也是官网的源。但是现在连不上，那就换个其他的源，用阿里的。

首先打开 软件和更新，设置选择 阿里的服务器



这时候，点击关闭，会要求重新载入，点击重新载入会报错，这是因为`sources.list`里面对阿里这个源配置的这个artful 属性不对。



打开文件`/etc/apt/sources.list`


删除里面的所有内容，替换成：

```bash
deb http://mirrors.aliyun.com/ubuntu/ bionic main restricted universe 

deb http://mirrors.aliyun.com/ubuntu/ bionic-security main restricted universe

deb http://mirrors.aliyun.com/ubuntu/ bionic-updates main restricted universe

deb http://mirrors.aliyun.com/ubuntu/ bionic-proposed main restricted universe

deb http://mirrors.aliyun.com/ubuntu/ bionic-backports main restricted universe
```

**出现的问题, 文件自动添加一行, 需要手动删除**



这样第一个源的问题就解决了。然后 解决第二个`vmware`的问题，打开文件夹` /etc/apt/sources.list.d`

`cd /etc/apt/sources.list.d`
用 ls 命令查看这个文件夹里的所有内容



由于出问题的是`vmware`， 我们就把`vmware-tools.list `删除（建议不要直接删除，而是改成` vmware-tools.list.bak`。同理要修改某个配置文件xxx时，先备份成 `xxx.bak`文件，然后再修改）。这里删除不用担心软件无法更新，系统会自动再生成一个可用的 .list文件

`sudo mv vmware-tools.list vmware-tools.list.bak`
然后再执行 sudo apt-get update 就没有错误了
————————————————
版权声明：本文为CSDN博主「titake」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/qq_22498427/article/details/104345138





# 之后出现

```c
root@jiao-virtual-machine:/# apt update
Get:1 http://old-releases.ubuntu.com/ubuntu cosmic InRelease [242 kB]
0% [1 InRelease gpgv 242 kB] [Waiting for headers]Couldn't create tempfiles for splitting up /var/lib/apErr:1 http://old-releases.ubuntu.com/ubuntu cosmic InReleaseInRelease
  Could not execute 'apt-key' to verify signature (is gnupg installed?)
Get:2 http://old-releases.ubuntu.com/ubuntu cosmic-security InRelease [88.7 kB]
0% [2 InRelease gpgv 88.7 kB] [Waiting for headers]                                   354 B/s 11min 23sCouldn't create tempfiles for splitting up /var/lib/apt/lists/partial/old-releases.ubuntu.com_ubuntu_distErr:2 http://old-releases.ubuntu.com/ubuntu cosmic-security InRelease                                  
  Could not execute 'apt-key' to verify signature (is gnupg installed?)
Get:3 http://old-releases.ubuntu.com/ubuntu cosmic-updates InRelease [88.7 kB]                         
0% [3 InRelease gpgv 88.7 kB] [Waiting for headers]                                   354 B/s 15min 34sCouldn't create tempfiles for splitting up /var/lib/apt/lists/partial/old-releases.ubuntu.com_ubuntu_distErr:3 http://old-releases.ubuntu.com/ubuntu cosmic-updates InRelease                                   
  Could not execute 'apt-key' to verify signature (is gnupg installed?)
Get:4 http://old-releases.ubuntu.com/ubuntu cosmic-proposed InRelease [92.5 kB]                        
0% [4 InRelease gpgv 92.5 kB] [Waiting for headers]                                   354 B/s 19min 45sCouldn't create tempfiles for splitting up /var/lib/apt/lists/partial/old-releases.ubuntu.com_ubuntu_distErr:4 http://old-releases.ubuntu.com/ubuntu cosmic-proposed InRelease                                  
  Could not execute 'apt-key' to verify signature (is gnupg installed?)
Get:5 http://old-releases.ubuntu.com/ubuntu cosmic-backports InRelease [74.6 kB]                       
0% [5 InRelease gpgv 74.6 kB]                                                          354 B/s 24min 6sCouldn't create tempfiles for splitting up /var/lib/apt/lists/partial/old-releases.ubuntu.com_ubuntu_distErr:5 http://old-releases.ubuntu.com/ubuntu cosmic-backports InRelease                                 
  Could not execute 'apt-key' to verify signature (is gnupg installed?)
Reading package lists... Done                                                                          
W: GPG error: http://old-releases.ubuntu.com/ubuntu cosmic InRelease: Could not execute 'apt-key' to verify signature (is gnupg installed?)
E: The repository 'http://old-releases.ubuntu.com/ubuntu cosmic InRelease' is not signed.
N: Updating from such a repository can't be done securely, and is therefore disabled by default.
N: See apt-secure(8) manpage for repository creation and user configuration details.
W: GPG error: http://old-releases.ubuntu.com/ubuntu cosmic-security InRelease: Could not execute 'apt-key' to verify signature (is gnupg installed?)
E: The repository 'http://old-releases.ubuntu.com/ubuntu cosmic-security InRelease' is not signed.
N: Updating from such a repository can't be done securely, and is therefore disabled by default.
N: See apt-secure(8) manpage for repository creation and user configuration details.
W: GPG error: http://old-releases.ubuntu.com/ubuntu cosmic-updates InRelease: Could not execute 'apt-key' to verify signature (is gnupg installed?)
E: The repository 'http://old-releases.ubuntu.com/ubuntu cosmic-updates InRelease' is not signed.
N: Updating from such a repository can't be done securely, and is therefore disabled by default.
N: See apt-secure(8) manpage for repository creation and user configuration details.
W: GPG error: http://old-releases.ubuntu.com/ubuntu cosmic-proposed InRelease: Could not execute 'apt-key' to verify signature (is gnupg installed?)
E: The repository 'http://old-releases.ubuntu.com/ubuntu cosmic-proposed InRelease' is not signed.
N: Updating from such a repository can't be done securely, and is therefore disabled by default.
N: See apt-secure(8) manpage for repository creation and user configuration details.
W: GPG error: http://old-releases.ubuntu.com/ubuntu cosmic-backports InRelease: Could not execute 'apt-key' to verify signature (is gnupg installed?)
E: The repository 'http://old-releases.ubuntu.com/ubuntu cosmic-backports InRelease' is not signed.
N: Updating from such a repository can't be done securely, and is therefore disabled by default.
N: See apt-secure(8) manpage for repository creation and user configuration details.
```

```
    
//原因没有权限    
root@jiao-virtual-machine:/# chmod 777 /tmp
```







