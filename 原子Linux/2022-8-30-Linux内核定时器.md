---
layout: post
title: "Linux内核定时器" 
date:   2022-8-30 15:39:08 +0800
tags: 嵌入式 原子Linux 
---

# 内核定时器

## Linux时间管理

Cortex-M使用Systick作为系统定时器

定时器分为硬件定时器, 软件定时器, 原理是依靠系统定时器驱动, 

硬件定时器提供时钟源，时钟源的频率可以设置，设置好以后就周期性的产生定时中断，系统使用定时中断来计时, 中断周期性产生的频率就是系统频率，也叫做节拍率(tickrate)(有的资料也叫系统频率)

可以通过图形化界面设置系统节拍率, 默认为100Hz, 定义在HZ变量中

```
-> Kernel Features                                              	-> Timer frequency (<choice> [=y])  
```

高节拍率会导致中断的产生更加频繁，频繁的中断会加剧系统的负担

Linux内核使用全局变量jiffies来记录系统从启动以来的系统节拍数，系统启动的时候会将jiffies初始化为0，jiffies定义在文件`include/linux/jiffies.h`中

```
76extern u64 __jiffy_data jiffies_64;
77extern unsigned long volatile __jiffy_data jiffies;
```

jiffies_64和jiffies其实是同一个东西，jiffies_64用于64位系统，而jiffies用于32位系统。为了兼容不同的硬件, jiffies其实就是jiffies_64的低32位

![QQ图片20220830155110](E:\a学习\笔记\img\QQ图片20220830155110.png)

如果`unkown`超过known的话，time_after函数返回真，否则返回假。如果`unkown`没有超过known的话time_before函数返回真，否则返回假。`time_after_eq`函数和`time_after`函数类似，只是多了判断等于这个条件

![QQ图片20220830155441](E:\a学习\笔记\img\QQ图片20220830155441.png)

## 内核定时器

软件定时器不是直接给周期值, 是设置周期满了以后的时间点, 然后会运行处理函数, 并关闭

Linux内核使用timer_list结构体表示内核定时器，timer_list定义在文件`include/linux/timer.h`中

expires成员变量表示超时时间，单位为节拍数

```c
struct timer_list {
	struct list_head entry;
	unsigned long expires;/* 定时器超时时间，单位是节拍数*/
	struct tvec_base *base;
	void (*function) (unsigned long); /* 定时处理函数*/ 			unsigned long data; /* 要传递给function函数的参数, 可以用来传递自己的地址*/
	int slack;
};
```

### 函数

```c
void init_timer(structtimer_list *timer)
```

>   定义了一个timer_list变量以后一定要先用`init_timer`初始化一下

```c
void add_timer(struct timer_list *timer)
```

>   向Linux内核注册定时器，使用add_timer函数向内核注册定时器以后，定时器就会开始运行

```c
int del_timer(struct timer_list * timer)
```

>   删除一个定时器，不管定时器有没有被激活，都可以使用此函数删除。在多处理器系统上，定时器可能会在其他的处理器上运行，因此在调用del_timer函数删除定时器之前要先等待其他处理器的定时处理器函数退出

```c
int del_timer_sync(struct timer_list *timer)
```

>   del_timer函数的同步版，会等待其他处理器使用完定时器再删除，del_timer_sync不能使用在中断上下文中。

```c
int mod_timer(struct timer_list *timer, unsigned long expires)
```

>   用于修改定时值，如果定时器还没有激活的话，mod_timer函数会激活定时器, expires：修改后的超时时间。返回值：0，调用mod_timer函数前定时器未被激活, 经常用于周期处理函数

![QQ图片20220830164038](E:\a学习\笔记\img\QQ图片20220830164038.png)



## 编写驱动

`ioctl`函数

```c
	long (*unlocked_ioctl) (struct file *, unsigned int, unsigned long);
	long (*compat_ioctl) (struct file *, unsigned int, unsigned long);
```

对应两个函数, 和应用中的ioctl函数对应, 但是在64位系统上32位程序调用后者, 32微系统调用前者

```c
static long timer_ioctl(struct file *filp, unsigned int cmd, unsigned long arg)
```

>   cmd就是命令, arg是命令的参数

命令是自己定义的, 但是要符合规则, 其实就是一个数, 是一个32位的数字, 分为四段, 每一段的意义不同, 高8位是幻数, 用来区分不同的驱动, 8位序数, 2位的传输方向, 最低的14位是数据的大小

### 函数

```
#define _IO(type,nr)		
#define _IOR(type,nr,size)	
#define _IOW(type,nr,size)	
#define _IOWR(type,nr,size)	
```

>   第一个, 没有参数的命令, 第二个命令是读取数据, 第三个写数据, 第四个双向数据传输, 三个参数对应幻数, 序数, 数据大小
>
>   ```
>   #define SETPERIOD_CMD   _IOW(0xef, 3, int)
>   ```





















