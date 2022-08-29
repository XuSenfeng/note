---
layout: post
title: "Linux点灯" 
date:   2022-8-26 15:39:08 +0800
tags: 嵌入式 原子Linux 
---

# Linux点灯

Linux也可以直接操作寄存器, 但是很麻烦, 不能直接对寄存器的物理地址进行操作, 因为Linux会使用MMU进行操作, 完成虚拟空间到物理空间的映射, 还有进行内存保护, 对于32位的寄存器可以映射的地址是4GB空间

在Linux中进行的都是虚拟的地址, 在对物理地址进行操作之前, 需要先得到物理地址对应的虚拟地址, 开启MMU以后就需要使用ioremap函数, 定义在arch/arm/linux/asm/io.h文件之中, 第一个参数是物理地址起始地址, 第二个是内存的长度

在使用后需要使用函数进行释放, iounmap, 只有一个参数, 是首地址, 返回的



## 流程

初始化时钟, 初始化复用, 电气属性

在输入输出的时候, 有专门的操作函数

readb(地址)

readw

read

读取参数

writeb(值, 地址)

writew

writel

写操作

## 缺点

使用register_chrdev,, 浪费了很多的次设备号, 需要手动进行设置设备号

手动创建节点

## 新版实现方法

+   设备号的注册

`alloc_chrdev_region`: 自动分配设备号, 第一个参数, 返回的设备号, 为指针, 第二个参数基础的次设备号, 第三个次设备号的个数, 第四个是名字

`unregister_chrdev_region`: 卸载驱动的时候使用释放之前申请的设备号, 第一个参数, 起始设备号, 第二个参数, 数值的大小

`register_chrdev_region`: 指定某一个设备号, 如果指定某一个设备号就是用这个函数, 第一个参数设备号, 一般是主设备号, 使用MKDEV构建完整的, 第二个参数多少个设备号, 第三个是名字

>    实际的驱动编写有两种方式, 给定了没有给定两种, 释放的时候使用同一个函数进行

+   新的字符设备注册

```C
struct cdev {
	struct kobject kobj;
	struct module *owner;
	const struct file_operations *ops;
	struct list_head list;
	dev_t dev;
	unsigned int count;
};
```

表示字符设备, 两个重要的成员变量：ops和dev，这两个就是字符设备文件操作函数集合file_operations以及设备号dev_t。

之后使用`cdev_init`函数进行初始化, 有两个参数, 第一个是cdev, 第二个是file_operations, 都是指针

初始化之后使用`cdev_add`添加到内核, 第一个参数, cdev指针, 第二个参数设备号, 第三个是个数

删除使用`cdev_del`: 进行删除, 一个参数cdev指针

>   实际上就是对之前使用的函数进行的一次拆解

## 自动创建设备节点

`ndev`机制, 2.6内核引入, 替换`devfs`, 提供热插拔管理, 在创建的时候自动创建/dev/xx, busy会创建一个`ndev`的简化版本, `mdev`, 一般在嵌入式Linux中使用

```
在文件中
/lib/modules/4.1.15 # vi /etc/init.d/rcS 

加入
echo /sbin/mdev > /proc/sys/kernel/hotplug
```



+   创建类

在`include/linux/device.h`有一个宏定义`class_create`有两个参数, owner和name, 返回一个struct class类型的指针

```C
newcharled.class = class_create(THIS_MODULE, NEWCHARLCD_NAME);
if(IS_ERR(newcharled.class))
    return PTR_ERR(newcharled.class);
```

IS_ERR用来对指针进行检查

+   删除一个类

```C
void class_destroy(struct class *cls)
```



+   创建好一个类之后还需要在类下面创建设备

`device_creat`函数, 参数一是class, 使用的类指针, 参数二是父设备, 一般为NULL, 参数三设备号, 参数四, 可能会用的参数一般NULL, , 参数五, 设备名, 返回值是struct device

+   删除一个设备

`device_destroy`: 参数是设备struct class以及设备号

## 文件是有数据

在使用结构体进行描述设备变量的时候

是一个void类型的指针

一般是在open的时候设置私有的数据, 

```C
static int newcharled_open(struct inode *inode, struct file *filp)
{
    filp->private_data = &newcharled;
    return 0;
}
```

在使用的时候首先进行提取

```C
static int newcharled_realse(struct inode *inode, struct file *filp)
{
    struct newcharled_dev *dev = (struct newcharled_dev*)filp->private_data; 
    return 0;
}
```



## 出现错误

一般会使用goto, 在函数return 0之后定义标签进行处理, 一般用于处理申请的资源内存等















