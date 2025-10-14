## kthread_worker：怎么把内核线程当工人？

#### 驱动传输数据 

- 低速数据：驱动同步传输
    - 简单直接、传输效率低

- 高速数据：驱动交给内核来异步传输
    - 机制复杂、无阻塞

#### kthread_worker结构体

include/linux/kthread.h

表示把内核线程抽象为流水线工人，按序处理其他线程/进程交付的批量工作

```c
struct kthread_worker {
	unsigned int		flags;
	spinlock_t		lock;
	struct list_head	work_list; // 批量不延时工作
	struct list_head	delayed_work_list; // 延时的工作
	struct task_struct	*task; // Linux下面的一个线程
	struct kthread_work	*current_work; // 当前的具体工作
};
```

- lock：自旋锁
- work_list：链表节点，按序记录工作
- delayed_work_list：
- task：内核线程

- current_work：指向正在处理的具体工作

#### kthread_work结构体

include/linux/kthread.h

表示等待内核线程处理的具体工作

```c
struct kthread_work {
	struct list_head	node; // 记录在work_list里面
	kthread_work_func_t	func; // 需要处理的回调函数
	struct kthread_worker	*worker;
	/* Number of canceling calls that are running at the moment. */
	int			canceling;
};
```

- node：链表节点
- func：函数指针
- worker：处理该工作的内核线程工人

##### kthread_work_func_t数据类型定义

```
typedef void (*kthread_work_func_t)(struct kthread_work *work);
```

#### kthread_flush_work结构体

kernel/kthread.c

表示等待某个内核线程工人处理完所有工作

```c
struct kthread_flush_work {
	struct kthread_work	work;
	struct completion	done;
};
```

- work：具体内核线程工人
- done：完成量，等待所有工作处理完毕

#### 初始化kthread_worker

##### kthread_init_worker函数

```c
struct kthread_worker hi_worker; 
kthread_init_worker(&hi_worker); 
```

- 先定义，后初始化

#### 为kthread_worker创建内核线程

```c
struct task_struct *kworker_task;

kworker_task =kthread_run(kthread_worker_fn, &hi_worker, "nvme%d", 1);
```

- 先定义，后初始化
- kthread_worker_fn：内核线程一直运行的函数
- hi_worker：已初始化的kthread_worker结构体变量
- "nvme%d"：为内核线程设置名字

#### 初始化kthread_work

```c
struct kthread_work hi_work;
kthread_init_work(&hi_work, xxx_work_fn); 
```

- 先定义，后初始化
- xxx_work_fn：处理该工作的具体函数，自定义实现

#### 启动工作

交付工作给内核线程工人

```c
kthread_queue_work(&hi_worker, &hi_work);
```

- hi_worker：具体内核线程工人
- hi_work：具体工作

#### FLUSH工作队列

刷新指定 kthread_worker上所有 work

```c
kthread_flush_worker(&hi_worker);
```

- hi_worker：具体内核线程工人

#### 停止内核线程

```c
kthread_stop(kworker_task);
```

## 使用

```c
/*------------------字符设备内容----------------------*/
#define DEV_NAME "rgb_led"
#define DEV_CNT (1)

int rgb_led_red;
int rgb_led_green;
int rgb_led_blue;

/*定义 led 资源结构体，保存获取得到的节点信息以及转换后的虚拟寄存器地址*/
struct led_resource
{
	struct device_node *device_node; //rgb_led_red的设备树节点
	void __iomem *virtual_CCM_CCGR;
	void __iomem *virtual_IOMUXC_SW_MUX_CTL_PAD;
	void __iomem *virtual_IOMUXC_SW_PAD_CTL_PAD;
	void __iomem *virtual_DR;
	void __iomem *virtual_GDIR;
};

static dev_t led_devno;					 //定义字符设备的设备号
static struct cdev led_chr_dev;			 //定义字符设备结构体chr_dev
struct class *class_led;				 //保存创建的类
struct device *device;					 // 保存创建的设备
struct device_node *rgb_led_device_node; //rgb_led的设备树节点结构体

/*定义 R G B 三个灯的led_resource 结构体，保存获取得到的节点信息*/
struct led_resource led_red;
struct led_resource led_green;
struct led_resource led_blue;

struct kthread_worker hi_worker; 
struct kthread_work hi_work;
struct task_struct *kworker_task;
//用于保存接收到的数据,应使用链表结构存储每一次用户空间写入的值
unsigned int write_data; 

/*字符设备操作函数集，open函数*/
static int led_chr_dev_open(struct inode *inode, struct file *filp)
{
	printk("\n open form driver \n");
	kworker_task = kthread_run(kthread_worker_fn, &hi_worker, "nvme%d",1 );
	return 0;
}

void rgb_control(struct kthread_work *work)
{
	/*设置 GPIO1_04 输出电平*/
	if (write_data & 0x04)
	{
		gpio_set_value(rgb_led_red,0);
	}
	else
	{
		gpio_set_value(rgb_led_red,1);
	}

	/*设置 GPIO4_20 输出电平*/
	if (write_data & 0x02)
	{
		gpio_set_value(rgb_led_green,0);
	}
	else
	{
		gpio_set_value(rgb_led_green,1);
	}

	/*设置 GPIO4_19 输出电平*/
	if (write_data & 0x01)
	{
		gpio_set_value(rgb_led_blue,0);
	}
	else
	{
		gpio_set_value(rgb_led_blue,1);
	}
	return;
}


/*字符设备操作函数集，write函数*/
static ssize_t led_chr_dev_write(struct file *filp, const char __user *buf, size_t cnt, loff_t *offt)
{

	int ret,error;
	unsigned char receive_data[10]; //用于保存接收到的数据

	if(cnt>10)
			cnt =10;

	error = copy_from_user(receive_data, buf, cnt);
	if (error < 0)
	{
		return -1;
	}

	ret = kstrtoint(receive_data, 16, &write_data);
	if (ret) {
		return -1;
        }

	kthread_init_work(&hi_work, rgb_control); 

	kthread_queue_work(&hi_worker, &hi_work);

	return cnt;
}

/*字符设备操作函数集*/
static struct file_operations led_chr_dev_fops =
	{
		.owner = THIS_MODULE,
		.open = led_chr_dev_open,
		.write = led_chr_dev_write,
};

/*----------------平台驱动函数集-----------------*/
static int led_probe(struct platform_device *pdv)
{

	int ret = -1; //保存错误状态码

	printk(KERN_ALERT "\t  match successed  \n");

	/*获取rgb_led的设备树节点*/
	rgb_led_device_node = of_find_node_by_path("/rgb_led");
	if (rgb_led_device_node == NULL)
	{
		printk(KERN_ERR "\t  get rgb_led failed!  \n");
		return -1;
	}

	rgb_led_red = of_get_named_gpio(rgb_led_device_node,"rgb_led_red",0);
	if (rgb_led_red < 0)
	{
		printk(KERN_ERR "\t  rgb_led_red failed!  \n");
		return -1;
	}

	rgb_led_green = of_get_named_gpio(rgb_led_device_node,"rgb_led_green",0);
	if (rgb_led_green < 0)
	{
		printk(KERN_ERR "\t  rgb_led_green failed!  \n");
		return -1;
	}

	rgb_led_blue = of_get_named_gpio(rgb_led_device_node,"rgb_led_blue",0);
	if (rgb_led_blue < 0)
	{
		printk(KERN_ERR "\t  rgb_led_blue failed!  \n");
		return -1;
	}

    gpio_direction_output(rgb_led_red,1);
    gpio_direction_output(rgb_led_green,1);
    gpio_direction_output(rgb_led_blue,1);

	/*---------------------注册 字符设备部分-----------------*/

	//第一步
	//采用动态分配的方式，获取设备编号，次设备号为0，
	//设备名称为rgb-leds，可通过命令cat  /proc/devices查看
	//DEV_CNT为1，当前只申请一个设备编号
	ret = alloc_chrdev_region(&led_devno, 0, DEV_CNT, DEV_NAME);
	if (ret < 0)
	{
		printk("fail to alloc led_devno\n");
		goto alloc_err;
	}
	//第二步
	//关联字符设备结构体cdev与文件操作结构体file_operations
	led_chr_dev.owner = THIS_MODULE;
	cdev_init(&led_chr_dev, &led_chr_dev_fops);
	//第三步
	//添加设备至cdev_map散列表中
	ret = cdev_add(&led_chr_dev, led_devno, DEV_CNT);
	if (ret < 0)
	{
		printk("fail to add cdev\n");
		goto add_err;
	}

	//第四步
	/*创建类 */
	class_led = class_create(THIS_MODULE, DEV_NAME);

	/*创建设备*/
	device = device_create(class_led, NULL, led_devno, NULL, DEV_NAME);

	return 0;

add_err:
	//添加设备失败时，需要注销设备号
	unregister_chrdev_region(led_devno, DEV_CNT);
	printk("\n error! \n");
alloc_err:

	return -1;
}

static const struct of_device_id rgb_led[] = {
	{.compatible = "fire,rgb_led"},
	{/* sentinel */}};

/*定义平台设备结构体*/
struct platform_driver led_platform_driver = {
	.probe = led_probe,
	.driver = {
		.name = "rgb-leds-platform",
		.owner = THIS_MODULE,
		.of_match_table = rgb_led,
	}};

/*
*驱动初始化函数
*/
static int __init led_platform_driver_init(void)
{
	int DriverState;
	kthread_init_worker(&hi_worker); 
	DriverState = platform_driver_register(&led_platform_driver);
	printk(KERN_ALERT "\tDriverState is %d\n", DriverState);
	return 0;
}

/*
*驱动注销函数
*/
static void __exit led_platform_driver_exit(void)
{
	kthread_flush_worker(&hi_worker);
	kthread_stop(kworker_task);
	device_destroy(class_led, led_devno);		  //清除设备
	class_destroy(class_led);					  //清除类
	cdev_del(&led_chr_dev);						  //清除设备号
	unregister_chrdev_region(led_devno, DEV_CNT); //取消注册字符设备

	/*注销字符设备*/
	platform_driver_unregister(&led_platform_driver);

	printk(KERN_ALERT "led_platform_driver exit!\n");
}

module_init(led_platform_driver_init);
module_exit(led_platform_driver_exit);
```

## 彻底掌握kthread_worker队列化机制

![image-20250925223027063](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/lenovo-picture/202509252230164.png)

#### kthread_init_worker()宏

include/linux/kthread.h

初始化kthread_worker

```c
#define kthread_init_worker(worker)					\
	do {								\
		static struct lock_class_key __key;			\
		__kthread_init_worker((worker), "("#worker")->lock", &__key); \
	} while (0)
```

##### __kthread_init_worker()函数

include/linux/kthread.h

```c
void __kthread_init_worker(struct kthread_worker *worker,
				const char *name,
				struct lock_class_key *key)
{
	memset(worker, 0, sizeof(struct kthread_worker));
	spin_lock_init(&worker->lock); // 初始化自旋锁
	lockdep_set_class_and_name(&worker->lock, key, name);
	INIT_LIST_HEAD(&worker->work_list);
	INIT_LIST_HEAD(&worker->delayed_work_list);
}
```

#### kthread_worker_fn()函数

kernel/kthread.c

线程的处理函数

```c
int kthread_worker_fn(void *worker_ptr)
{
	struct kthread_worker *worker = worker_ptr;
	struct kthread_work *work;
	...
	worker->task = current; // 记录当前的线程

	if (worker->flags & KTW_FREEZABLE)
		set_freezable();

repeat:
    // 设置为可以接受中断
	set_current_state(TASK_INTERRUPTIBLE);	/* mb paired w/ kthread_stop */
	// 判断一下是不是需要停止(判断线程标志位kthread_stop函数进行设置)
	if (kthread_should_stop()) {
		__set_current_state(TASK_RUNNING);
		spin_lock_irq(&worker->lock);
		worker->task = NULL;
		spin_unlock_irq(&worker->lock);
		// 线程退出
        return 0;
	}
	
	work = NULL;
	spin_lock_irq(&worker->lock);
	if (!list_empty(&worker->work_list)) {
        // 获取第一个工作
		work = list_first_entry(&worker->work_list,
					struct kthread_work, node);
		list_del_init(&work->node);
	}
    
	worker->current_work = work;
	spin_unlock_irq(&worker->lock);

	if (work) {
		__set_current_state(TASK_RUNNING);
		// 具体工作
        work->func(work);
	} else if (!freezing(current))
		schedule();

	try_to_freeze();
	cond_resched();
	goto repeat;
}
```

##### kthread_should_stop()函数

kernel/kthread.c

```c
bool kthread_should_stop(void)
{
	return test_bit(KTHREAD_SHOULD_STOP, &to_kthread(current)->flags);
}
```

- 调用kthread_stop()函数后，设置线程flags为KTHREAD_SHOULD_STOP

#### kthread_init_work()函数

include/linux/kthread.h

初始化kthread_work

```c
#define kthread_init_work(work, fn)					\
	do {								\
		memset((work), 0, sizeof(struct kthread_work));		\
		INIT_LIST_HEAD(&(work)->node);				\
		(work)->func = (fn);					\
	} while (0)
```

#### kthread_queue_work()函数

kernel/kthread.c

```c
bool kthread_queue_work(struct kthread_worker *worker,
			struct kthread_work *work)
{
	bool ret = false;
	unsigned long flags;

	spin_lock_irqsave(&worker->lock, flags);
	if (!queuing_blocked(worker, work)) {
        // 判断一下这个任务是不是已经被挂载, 以及这个任务是不是取消了
		kthread_insert_work(worker, work, &worker->work_list);
		ret = true;
	}
	spin_unlock_irqrestore(&worker->lock, flags);
	return ret;
}
```

##### queuing_blocked()函数

kernel/kthread.c

```c
static inline bool queuing_blocked(struct kthread_worker *worker,
				   struct kthread_work *work)
{
	lockdep_assert_held(&worker->lock);

	return !list_empty(&work->node) || work->canceling;
}
```

##### kthread_insert_work()函数

kernel/kthread.c

```c
static void kthread_insert_work(struct kthread_worker *worker,
				struct kthread_work *work,
				struct list_head *pos)
{
	kthread_insert_work_sanity_check(worker, work);

	list_add_tail(&work->node, pos);
	work->worker = worker;
	if (!worker->current_work && likely(worker->task))
		wake_up_process(worker->task);
}
```



#### kthread_flush_worker()函数

kernel/kthread.c

实际是在工作链表上面加一个具体的工作, 位于链表的末尾

```c
void kthread_flush_worker(struct kthread_worker *worker)
{
	struct kthread_flush_work fwork = {
        // 初始化一下参数
		KTHREAD_WORK_INIT(fwork.work, kthread_flush_work_fn),
		// 初始化completion
        COMPLETION_INITIALIZER_ONSTACK(fwork.done),
	};

	kthread_queue_work(worker, &fwork.work);
    // 阻塞等待当前所有任务结束
	wait_for_completion(&fwork.done);
}
```

##### KTHREAD_WORK_INIT()宏

```c
#define KTHREAD_WORK_INIT(work, fn)	{				\
	.node = LIST_HEAD_INIT((work).node),				\
	.func = (fn),							\
	}
```

##### COMPLETION_INITIALIZER_ONSTACK()宏

include/linux/completion.h

```
(*({ init_completion(&work); &work; }))
```

##### kthread_flush_work_fn()函数

kernel/kthread.c

```c
static void kthread_flush_work_fn(struct kthread_work *work)
{
	struct kthread_flush_work *fwork =
		container_of(work, struct kthread_flush_work, work);
    // 释放阻塞
	complete(&fwork->done);
}
```

