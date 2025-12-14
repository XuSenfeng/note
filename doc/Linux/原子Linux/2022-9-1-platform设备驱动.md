---
layout: post
title: "platform设备驱动" 
date:   2022-9-1 15:39:08 +0800
tags: 嵌入式 原子Linux
---

# platform设备驱动

当驱动变得复杂的时候, 为了方便驱动的编写, 提高可重用性

## 驱动分离与分层

### 驱动分离分隔

把控制器和具体的设备分隔开, 根据Linux提供的框架使用统一的API接口, 分为主机控制器驱动, 设备驱动, 主机控制器驱动是由半导体厂商写的, 在linux下编写具体的设备驱动

中间的练习就是核心层, 统一的API定义

### 驱动的分层

我们在编写输入设备驱动的时候只需要处理好输入事件的上报即可，至于如何处理这些上报的输入事件那是上层去考虑的

 ## 总线-设备-驱动
根据分层理念形成的, 总线属于内核, 我们要编写驱动和设备
驱动就是具体的设备驱动
设备就是具体的设备属性, 包括属性, 地址范围等

### 总线

总线有一个结构体进行表示

```c
struct bus_type {
	const char		*name;
	const char		*dev_name;
	struct device		*dev_root;
	struct device_attribute	*dev_attrs;	/* use dev_groups instead */
	const struct attribute_group **bus_groups;
	const struct attribute_group **dev_groups;
	const struct attribute_group **drv_groups;

	int (*match)(struct device *dev, struct device_driver *drv);
	int (*uevent)(struct device *dev, struct kobj_uevent_env *env);
	int (*probe)(struct device *dev);
	int (*remove)(struct device *dev);
	void (*shutdown)(struct device *dev);

	int (*online)(struct device *dev);
	int (*offline)(struct device *dev);

	int (*suspend)(struct device *dev, pm_message_t state);
	int (*resume)(struct device *dev);

	const struct dev_pm_ops *pm;

	const struct iommu_ops *iommu_ops;

	struct subsys_private *p;
	struct lock_class_key lock_key;
};
```

>   match函数，此函数很重要，单词match的意思就是“匹配、相配”，因此此函数就是完成设备和驱动之间匹配的，总线就是使用match函数来根据注册的设备来查找对应的驱动，或者根据注册的驱动来查找相应的设备, dev和drv，这两个参数分别为device和device_driver类型

总线主要就是为了总线下的设备驱动进行匹配, 有很多条总线在/sys/bus目录下面就是具体的总线

```
/ # cd sys/bus
/sys/bus # ls
clockevents   event_source  mmc           sdio          usb
clocksource   hid           platform      serio         virtio
container     i2c           rpmsg         soc           workqueue
cpu           mdio_bus      scsi          spi
/sys/bus # cd i2c
/sys/bus/i2c # ls
devices            drivers_autoprobe  uevent
drivers            drivers_probe

```

向Linux内核注册总线

```c
bus_reguister()
bus_unreguister()
```

>   一般不会使用

### 驱动

```c
struct device_driver {
	const char		*name;
	struct bus_type		*bus;		//属于的总线

	struct module		*owner;
	const char		*mod_name;	/* used for built-in modules */

	bool suppress_bind_attrs;	/* disables bind/unbind via sysfs */

	const struct of_device_id	*of_match_table;
	const struct acpi_device_id	*acpi_match_table;

	int (*probe) (struct device *dev);	//使用的函数
	int (*remove) (struct device *dev);
	void (*shutdown) (struct device *dev);
	int (*suspend) (struct device *dev, pm_message_t state);
	int (*resume) (struct device *dev);
	const struct attribute_group **groups;

	const struct dev_pm_ops *pm;

	struct driver_private *p;
};
```

+   驱动设备匹配以后, 驱动里面的probe函数就会运行

使用diriver_register进行向总线注册驱动, 会检查有没有匹配的设备, 有的话就会调probe函数

>   匹配方式, name和of_match_table
>
>   ```c
>   struct of_device_id {
>   	char	name[32];
>   	char	type[32];
>   	char	compatible[128];
>   	const void *data;
>   };
>   ```
>
>   



### 设备

```c
struct device {
	struct device		*parent;
	struct device_private	*p;
	struct kobject kobj;
	const char	*init_name; /* initial name of the device */
	const struct device_type *type;

	struct mutex mutex;	/* mutex to synchronize calls to * its driver.*/

	struct bus_type	*bus; /* type of bus device is on */
	struct device_driver *driver;	/* which driver has allocated this device */

......

};
```

>   会定义自己的bus和driver

向总线注册设备的时候使用device_register函数, 和上面的driver注册类似,进行查找匹配, 最后也会调用probe函数

probe函数就是驱动编写人员进行编写的



## platform平台驱动开发

对于soc内部的RTC, timer等不好总结为讲具体的总线, Linux提出虚拟总线, plantform总线, 有对应的设备和驱动

对于platform平台来说, 他的platform_match函数就是用来匹配的

```c
struct bus_type platform_bus_type = {
	.name		= "platform",
	.dev_groups	= platform_dev_groups,
	.match		= platform_match,
	.uevent		= platform_uevent,
	.pm		= &platform_dev_pm_ops,
};
```

使用驱动和设备里面的成员变量

### dirver

```c
struct platform_driver {
	int (*probe)(struct platform_device *);
	int (*remove)(struct platform_device *);
	void (*shutdown)(struct platform_device *);
	int (*suspend)(struct platform_device *, pm_message_t state);
	int (*resume)(struct platform_device *);
	struct device_driver driver; //调用父类
	const struct platform_device_id *id_table;
	bool prevent_deferred_probe;
};
```

>   在父类下面添加相关的属性, 能否匹配根据id_table属性以及父类中的`const struct acpi_device_id	*acpi_match_table;` 和name进行匹配
>
>   主要实现的是probe加载匹配时候进行的函数, remove不匹配时候进行的函数, driver->name匹配使用的名字, 



使用**platform_driver_register**函数向内核注册驱动, 注册的就是上面的结构体, 在注册驱动的时候如果驱动设备匹配成功执行probe函数







### device

```c
struct platform_device {
	const char	*name;
	int		id;
	bool		id_auto;
	struct device	dev; //父类
	u32		num_resources;
	struct resource	*resource;

	const struct platform_device_id	*id_entry;
	char *driver_override; /* Driver name to force a match */

	/* MFD cell pointer */
	struct mfd_cell *mfd_cell;

	/* arch specific additions */
	struct pdev_archdata	archdata;
};
```

>   使用name进行匹配, id 为-1 表示没有id
>
>   num_resources, 资源大小, resource资源, 一般用来面描述内存, 是一个结构体数组
>
>   ```c
>   struct resource {
>   	resource_size_t start;
>   	resource_size_t end;
>   	const char *name;
>   	unsigned long flags;
>   	struct resource *parent, *sibling, *child;
>   };
>   ```
>
>   >   start和end分别表示资源的起始和终止信息，对于内存类的资源，就表示内存起始和终止地址，name表示资源名字，flags表示资源类型, 可选的资源类型都定义在了文件include/linux/ioport.h, 这里使用IORESOURCE_MEM表示内存





设备有两种情况, 有设备树和没有设备树

没有的时候需要注册一个platform_device结构体, 这个时候需要驱动开发人员编写注册文件, 使用platform_device_register函数注册设备

有了以后修改节点就可以了, 匹配之后会运行platform_device的probe函数



### 匹配过程

驱动的匹配是通过bus->match函数进行的, platform总线下的match函数就是对应结构体下面的platform-match函数完成的

有四种匹配的方式, 有设备树的时候直接比较设备树, 没有的时候通常比较名字, driver使用父类中的name

有设备树的时候, driver使用的是父类中的of_match_table, 非常重要, 类型为of_device_id结构体数组, 使用他的compatible属性, 可以有多个匹配的对象

### 信息交流

```c
extern struct resource *platform_get_resource(struct platform_device *,unsigned int, unsigned int);
```

>   获取对应的资源, 参数一, 对应的设备, 参数二, 资源的类型, 第三个是索引



## 实际编写(有设备树)

分为两部分, driver和device

1.   注册设备platform_device_register, 取消设备platform_device_unregister

```c
struct platform_device leddevice = {
    .name = "imx6ull-led",
    .id = -1, //表示没有id
    .dev = {
        .release =  leddevice_release, //在释放的时候调用这个函数 void	(*release)(struct device *dev);
    },
    .num_resources = ARRAY_SIZE(led_resource),      //资源
    .resource = led_resource,
}
```

2.   添加各种属性, 尤其是匹配时用的name和操作的时候使用的resource

+   驱动编写

1.   和上面类似, 注册删除结构体
2.   编写驱动需要寄存器地址信息, 属于设备信息, 定义在platform_device里面, 需要在驱动获取设备资源, 使用函数platform_get_resource()

```
for(i=0;i<5;i++)
{
	//获取
	ledsource[i]=platform_get_resource(dev,IORESOURCE_MEM, i);
    if(ledsource[i] == NULL)
    {
    	return -EINVAL;
    }
}
```



注册函数, 实现函数, 和前面的实现方法一样



## 实际编写(没有设备树)

不用再进行注册, 直接修改设备树

只需要修改设备树然后编写驱动文件

1.   初始化设备树

```c
163     gpioled {
164         compatible = "jiao, gpioled";
165         pinctrl-name = "default";
166         pinctrl-0 = <&pinctrl_gpioled>;
167         status = "okey";
168         led-gpio = <&gpio1 3 GPIO_ACTIVE_LOW>;
169     };
```

2.   platform_device初始化, 这里匹配采用的是结构体下.driver的of_match_table属性

```c
static const struct of_device_id beep_of_match[] = {
	{.compatible = "jiao,beep"},
	{/* sentinel */},
};

struct platform_device leddriver = {
	.driver  = {
		.name = "imx6ull-led",
		.of_match_tablem = acpi_device_id;
	},
};
```

匹配成功之后probe函数的参数本身就带有设备的信息

```c
#if 0
    gpioled.nd = of_find_node_by_path("/gpioled");
    if(gpioled.nd == NULL)
    {
        ret = -EINVAL;
        goto fail_findnd;
    }
#endif
	gpioled.nd = dev->dev.of_node; //简化之后
```

platform提供很多函数去获取相关的信息











