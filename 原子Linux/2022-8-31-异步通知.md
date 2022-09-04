---
layout: post
title: "异步通知" 
date:   2022-8-31 15:39:08 +0800
tags: 嵌入式 原子Linux 
---

# 异步通知

首先是硬件中断, 只要处理器设置好就不需要再去查看

信号==>软件层次上对于中断的模拟, 效果和中断类似, 驱动可以主动向软件发送信号, 告诉自己可以访问

```c
34	#define SIGHUP    1/* 终端挂起或控制进程终止*/
35	#define SIGINT        2/* 终端中断(Ctrl+C组合键)    */
36	#define SIGQUIT       3/* 终端退出(Ctrl+\组合键)    */
37	#define SIGILL        4/* 非法指令*/
38	#define SIGTRAP       5/* debug使用，有断点指令产生*/
39	#define SIGABRT       6/* 由abort(3)发出的退出指令*/
40	#define SIGIOT        6/* IOT指令*/
41	#define SIGBUS        7/* 总线错误*/
42	#define SIGFPE        8/* 浮点运算错误*/
43	#define SIGKILL       9/* 杀死、终止进程*/
44	#define SIGUSR110/* 用户自定义信号1           */
45	#define SIGSEGV      11/* 段违例(无效的内存段)    */
46	#define SIGUSR2      12/* 用户自定义信号2           */
47	#define SIGPIPE      13/* 向非读管道写入数据*/
48	#define SIGALRM      14/* 闹钟*/
49	#define SIGTERM      15/* 软件终止*/
50	#define SIGSTKFLT    16/* 栈异常*/
51	#define SIGCHLD      17/* 子进程结束*/
52	#define SIGCONT      18/* 进程继续*/
53	#define SIGSTOP      19/* 停止进程的执行，只是暂停*/
54	#define SIGTSTP      20/* 停止进程的运行(Ctrl+Z组合键) */
55	#define SIGTTIN      21/* 后台进程需要从终端读取数据*/
56	#define SIGTTOU      22/* 后台进程需要向终端写数据*/
57	#define SIGURG      23/* 有"紧急"数据*/
58	#define SIGXCPU      24/* 超过CPU资源限制*/
59	#define SIGXFSZ      25/* 文件大小超额*/
60	#define SIGVTALRM    26/* 虚拟时钟信号*/
61	#define SIGPROF      27/* 时钟信号描述*/
62	#define SIGWINCH     28/* 窗口大小改变*/
63	#define SIGIO        29/* 可以进行输入/输出操作*/
64	#define SIGPOLL      SIGIO   
65	/* #define SIGLOS    29 */
66	#define SIGPWR       30/* 断点重启*/
67	#define SIGSYS       31/* 非法的系统调用*/
68	#define  SIGUNUSED   31/* 未使用信号*/
```

>   除了SIGKILL(9)和SIGSTOP(19)这两个信号不能被忽略外，其他的信号都可以忽略。

## 应用中处理

+   具体的实现

1）signal(SIGIO, sig_handler);
调用signal函数，让指定的信号SIGIO与处理函数sig_handler对应。

2）fcntl(fd, F_SET_OWNER, getpid());
指定一个进程作为文件的“属主(filp->owner)”，这样内核才知道信号要发给哪个进程。

3）设置文件标志，添加FASYNC标志
f_flags = fcntl(fd, F_GETFL);
fcntl(fd, F_SETFL, f_flags | FASYNC);



如果要在应用程序中使用信号，那么就必须设置信号所使用的信号处理函数，在应用程序中使用signal函数来设置指定信号的处理函数，signal函数原型如下所示

```c
sighandler_t signal(int signum, sighandler_t handler)
```

>   signum：要设置处理函数的信号。handler：信号的处理函数。`typedef void (*sighandler_t)(int)`, 返回值：设置成功的话返回信号的前一个处理函数，设置失败的话返回SIG_ERR

Ctrl+C发送SIGINT信号

 fcntl系统调用可以用来对已打开的文件描述符进行各种控制操作以改变已打开文件的的各种属性

```c
#include<unistd.h>
#include<fcntl.h>
int fcntl(int fd, int cmd);
int fcntl(int fd, int cmd, long arg);
int fcntl(int fd, int cmd ,struct flock* lock);
```

![QQ图片20220901122303](E:\a学习\笔记\img\QQ图片20220901122303.png)

[ 图片:cntl函数的用法总结](https://blog.csdn.net/fengxinlinux/article/details/51980837)

[ 以下描述f:cntl函数用法](https://blog.csdn.net/ALone_cat/article/details/126554725)

**F_GETFL** 取得fd的文件状态标志，如同下面的描述一样(arg被忽略)，在说明open函数时，已说明了文件状态标志。

**F_SETFL** 设置给arg描述符状态标志，可以更改的几个标志是：O_APPEND，O_NONBLOCK，O_SYNC 和 O_ASYNC。而fcntl的文件状态标志总共有7个：O_RDONLY , O_WRONLY , O_RDWR , O_APPEND , O_NONBLOCK , O_SYNC和O_ASYNC

可更改的几个标志如下面的描述：

>   O_NONBLOCK 非阻塞I/O，如果read(2)调用没有可读取的数据，或者如果write(2)操作将阻塞，则read或write调用将返回-1和EAGAIN错误
>   O_APPEND 强制每次写(write)操作都添加在文件大的末尾，相当于open(2)的O_APPEND标志
>   O_DIRECT 最小化或去掉reading和writing的缓存影响。系统将企图避免缓存你的读或写的数据。如果不能够避免缓存，那么它将最小化已经被缓存了的数据造成的影响。如果这个标志用的不够好，将大大的降低性能
>   O_ASYNC 当I/O可用的时候，允许SIGIO信号发送到进程组，例如：当有数据可以读的时候

重点就是通过fcntl函数设置进程状态为FASYNC，经过这一步，驱动程序中的fasync函数就会执行。

**F_SETOWN** 设置将接收SIGIO和SIGURG信号的进程id或进程组id，进程组id通过提供负值的arg来说明(arg绝对值的一个进程组ID)，否则arg将被认为是进程id

+   实际使用

```c
fcntl(fd, F_SETOWN, getpid());//添加进程到内核
int flags = 0;
flags = fcntl(fd, F_GETFL);//设置为FASYNC模式, 异步通知
fcntl(fd, F_SETFL, flags | FASYNC);
```



## 信号发送

+   具体实现：

一 驱动方面：

1.   在设备抽象的数据结构中增加一个struct fasync_struct的指针
2.    实现设备操作中的fasync函数，这个函数很简单，其主体就是调用内核的fasync_helper函数。
3.    在需要向用户空间通知的地方(例如中断中)调用内核的kill_fasync函数。
4.    在驱动的release方法中调用kpp_fasync(-1, filp, 0);函数





fasync_struct结构体

```c
struct fasync_struct {
    spinlock_t      fa_lock;
    int magic;
    int fa_fd;
    struct fasync_struct *fa_next;
    struct file     *fa_file;
    struct rcu_head     fa_rcu;
};
```

>   fasync_struct结构体指针变量定义到设备结构体中, 实现file_operations里面的fasync函数

```c
int (*fasync) (int fd, struct file *filp, int on)
```

>   一般通过调用fasync_helper函数来初始化前面定义的fasync_struct结构体指针

```c
int fasync_helper(int fd, struct file * filp, int on, struct fasync_struct **fapp)
```

```c
void kill_fasync(struct fasync_struct **fp, int sig, int band)
```

>   负责发送指定的信号, fp：要操作的fasync_struct。sig：要发送的信号。band：可读时设置为POLL_IN，可写时设置为POLL_OUT。

+   关闭驱动的时候要删除信号

在关闭驱动文件的时候需要在file_operations操作集中的release函数中释放fasync_struct，fasync_struct的释放函数同样为fasync_helper

```c
return xxx_fasync(-1,filp,0);/* 删除异步通知*/
```

>   调用的是file_operation中的函数, 是自己写的



## 示例

```c
static int imx6uirq_fasync(int fd, struct file *filp, int on)
{
     struct imx6uirq_dev *dev = filp->private_data;
	//进入的时候加载
     return fasync_helper(fd, filp, on, &dev->fasync_queue);
}
static int imx6uirq_realse(struct inode *inode, struct file *filp)
{
	//退出的时候移除
    struct imx6uirq_dev *dev = filp->private_data;
    imx6uirq_fasync(-1, filp, 0);

    return 0;
}

//定时器
if(atomic_read(&dev->realsekey)==1)//有信号的时候发送
{
    kill_fasync(&dev->fasync_queue, SIGIO, POLL_IN);//发送信号SIGIO, 文件为可读状态
}


```



应用

```c
//重写处理函数
static void sigio_signal_func(int num)
{
    int err;
    unsigned int key_value = 0;

    err = read(fd, &key_value, sizeof(key_value));
    if(err<0)
    {

    }
    else{
        printf("keyval is %d\r\n", key_value);
    }
}

//main
//设置信号处理函数
signal(SIGIO, sigio_signal_func);

fcntl(fd, F_SETOWN, getpid());//添加进程到内核
int flags = 0;
flags = fcntl(fd, F_GETFL);//设置为FASYNC模式, 异步通知
fcntl(fd, F_SETFL, flags | FASYNC);

```



























