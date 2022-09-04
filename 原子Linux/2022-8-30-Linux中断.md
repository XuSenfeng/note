---
layout: post
title: "Linux中断" 
date:   2022-8-30 15:39:08 +0800
tags: 嵌入式 原子Linux 
---

# Linux中断

## 函数

+   中断号

每一个中断都有一个对应的中断号,  `request_irq`函数用来申请所需要的中断, 有可能会导致睡眠, 所以要在可以睡眠的位置调用

```c
int request_irq(unsigned int irq, irq_handler_t handler, 				unsigned long flags,const char *name, void *dev)
```

>   irq：要申请中断的中断号。handler：中断处理函数，当中断发生以后就会执行此中断处理函数。flags：中断标志在文件include/linux/interrupt.h里面查看所有 name：中断名字，设置以后可以在/proc/interrupts文件中看到对应的中断名字。dev：如果将flags设置为IRQF_SHARED的话，dev用来区分不同的中断，一般情况下将dev设置为设备结构体，dev会传递给中断处理函数irq_handler_t的第二个参数。

![QQ图片20220830202302](E:\a学习\笔记\img\QQ图片20220830202302.png)

```c
void free_irq(unsigned int irq, void *dev)
```

>   irq：要释放的中断。dev：如果中断设置为共享(IRQF_SHARED)的话，此参数用来区分具体的中断。

```
irqreturn_t (*irq_handler_t) (int, void *)
```

>   中断处理函数格式
>
>   ```c
>   10	enumirqreturn {
>   11		IRQ_NONE        =(0<<0),
>   12		IRQ_HANDLED     =(1<<0),
>   13		IRQ_WAKE_THREAD  =(1<<1),
>   14	};
>   15
>   16	typedef enum irqreturn irqreturn_t;
>   ```
>
>   

```
void enable_irq(unsigned int irq)
void disable_irq(unsigned int irq
```

>   中断使能, 禁止某一个中断, 等到当前正在执行的中断处理函数执行完才返回

```
void disable_irq_nosync(unsigned int irq)
```

>   不会等待当前中断处理程序执行完毕。

```
local_irq_enable()
local_irq_disable()
```

>   local_irq_enable用于使能当前处理器中断系统，local_irq_disable用于禁止当前处理器中断系统, 恢复的时候可能会导致不可预知错误

```
local_irq_save(flags)
local_irq_restore(flags)
```

>   这两个函数是一对，local_irq_save函数用于禁止中断，并且将中断状态保存在flags中。local_irq_restore用于恢复中断，将中断到flags状态. 



## 上半部下半部

我们在使用request_irq申请中断的时候注册的中断服务函数属于中断处理的上半部，只要中断触发，那么中断处理函数就会执行

在实际处理的时候, 有的中断处理函数会占用大量的时间, 不符合中断的规则

中断处理函数仅仅响应中断，然后清除中断标志位即可。这个时候中断处理过程就分为了两部分：

上半部：上半部就是中断处理函数，那些处理过程比较快，不会占用很长时间的处理就可以放在上半部完成。

下半部：如果中断处理过程比较耗时，那么就将这些比较耗时的代码提出来，交给下半部去执行，这样中断处理函数就会快进快出。

+   如果要处理的内容不希望被其他中断打断，那么可以放到上半部。
+   如果要处理的任务对时间敏感，可以放到上半部
+   如果要处理的任务与硬件有关，可以放到上半部
+   除了上述三点以外的其他任务，优先考虑放到下半部。

### 软终端

使用结构体`softirq_action`表示软中断, `include/linux/interrupt.h`

```c
433	struct softirq_action
434	{
435		void(*action)(structsoftirq_action *);
436	};
```

在`kernel/softirq.c`文件中一共定义了10个软中断, 被定义为一个枚举类型

要使用软中断，必须先使用open_softirq函数注册对应的软中断处理函数

```c
void open_softirq(int nr,    void (*action)(struct softirq_action *))
```

>   nr：要开启的软中断。action：软中断对应的处理函数。
>
>   **必须在编译的时候静态注册, **内核中默认会打开TASKLET_SOFTIRQ和HI_SOFTIRQ

```c
void raise_softirq(unsigned int nr)
```

>   注册以后触发, nr：要触发的软中断

### tasklet

`tasklet`是利用软中断来实现的另外一种下半部机制, 建议使用这一个

Linux内核使用`tasklet_struct`结构体来表示`tasklet`

```c
484	struct tasklet_struct
485	{
486		structtasklet_struct *next;	/* 下一个tasklet */
487		unsignedlongstate;			/* tasklet状态*/
488		atomic_t count;/* 计数器，记录对tasklet的引用数*/
489		void(*func)(unsignedlong);	/* tasklet执行的函数*/
490		unsignedlongdata;			/* 函数func的参数*/
491	};
```

```c
void tasklet_init(struct tasklet_struct *t,void (*func)(unsigned long), unsigned long data);
```

>   进行结构物初始化, t：要初始化的`tasklet`  `func`: `tasklet`的处理函数。data：要传递给`func`函数的参数

也 可 以 使 用 宏DECLARE_TASKLET来 一 次 性 完 成`tasklet`的 定 义 和 初 始 化, 

```
DECLARE_TASKLET(name, func, data)
```

>   name为要定义的`tasklet`名字，这个名字就是一个`tasklet_struct`类型的时候变量，`func`就是`tasklet`的处理函数，data是传递给`func`函数的参数

+   在上半部，也就是中断处理函数中调用tasklet_schedule函数就能使tasklet在合适的时间运行，tasklet_schedule函数原型如下

```c
void tasklet_schedule(struct tasklet_struct *t)
```

也要使用上半部, 但是主要的作用是调用下半部的处理函数

### 工作队列

工作队列在进程上下文执行，工作队列将要推后的工作交给一个内核线程去执行，因为工作队列工作在进程上下文，因此工作队列允许睡眠或重新调度。

使用work_struct结构体表示一个工作

```c
struct work_struct {
    atomic_long_t data;
    structlist_head entry;
    work_func_t func;/* 工作队列处理函数*/
};
```

直接定义一个work_struct结构体变量即可，然后使用INIT_WORK宏来初始化工作

```c
#define INIT_WORK(_work, _func)
```

>   _work表示要初始化的工作，\_func是工作对应的处理函数, `static void kaywork(struct work_struct *work)`

也可以使用DECLARE_WORK宏一次性完成工作的创建和初始化

```
#define DECLARE_WORK(n, f)
```

>   n表示定义的工作(work_struct)，f表示工作对应的处理函数

和tasklet一样，工作也是需要调度才能运行的，工作的调度函数为schedule_work

```c
bool schedule_work(struct work_struct *work)
```

由于最后调用的函数参数是work_struct结构体, 所以可以使用函数`container_of`获取结构体所在的结构体的首地址, 第一个参数, 传入的work结构体, 第二个参数所在的结构体, 第三个参数自己的名字

## 在设备树中描述

Linux内核通过读取设备树中的中断属性信息来配置中断

打开imx6ull.dtsi文件，其中的intc节点就是I.MX6ULL的中断控制器节点

```c
intc: interrupt-controller@00a01000 {
    compatible = "arm,cortex-a7-gic";
    #interrupt-cells = <3>;
    interrupt-controller;
    reg = <0x00a01000 0x1000>,
    <0x00a02000 0x100>;
};
```

+   compatible = "arm,cortex-a7-gic";描述使用的驱动文件
+   #interrupt-cells = <3>; 表示此中断控制器下设备的cells大小

 第一个cells：中断类型，0表示SPI中断，1表示PPI中断。第二个cells：中断号，对于SPI中断来说中断号的范围为0\~987，对于PPI中断来说中断号的范围为0~15。第三个cells：标志，bit[3:0]表示中断触发类型，为1的时候表示上升沿触发，为2的时候表示下降沿触发，为4的时候表示高电平触发，为8的时候表示低电平触发。bit[15:8]为PPI中断的CPU掩码

```c
gpio1: gpio@0209c000 {
    compatible = "fsl,imx6ul-gpio", "fsl,imx35-gpio";
    reg = <0x0209c000 0x4000>;
    interrupts = <GIC_SPI 66 IRQ_TYPE_LEVEL_HIGH>,
    <GIC_SPI 67 IRQ_TYPE_LEVEL_HIGH>;
    gpio-controller;
    #gpio-cells = <2>;
    interrupt-controller;
    #interrupt-cells = <2>;
};
```

在单独的gpio中把cell改为2, 

```c
fxls8471@1e {
    compatible = "fsl,fxls8471";
    reg = <0x1e>;
    position = <0>;
    interrupt-parent = <&gpio5>; 	//父中断
    interrupts = <0 8>;		//指定对应的引脚以及触发电平
};
```

>   0表示GPIO5_IO00，8表示低电平触发, 1是上升沿, 2是下降沿, 4高电平触发



①、#interrupt-cells，指定中断源的信息cells个数。②、interrupt-controller，表示当前节点为中断控制器。③、interrupts，指定中断号，触发方式等。④、interrupt-parent，指定父中断，也就是中断控制器

+   interrupt-controller; 这是一个中断控制器

### 对应的函数

```c
unsigned int irq_of_parse_and_map(struct device_node *dev,int index)
```

>   dev：设备节点。index：索引号，interrupts属性可能包含多条中断信息，通过index指定要获取的信息

```c
int gpio_to_irq(unsigned int gpio)
```

>   `gpio`：要获取的GPIO编号











