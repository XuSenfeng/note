---
layout: post
title: "网络设备" 
date:   2022-9-5 15:39:08 +0800
tags: 嵌入式 原子Linux 
---

# 网络设备

## 硬件接口

网卡, 现在已经集成为一个芯片, 嵌入式主要就是两个芯片, MAC和PHY, 一般说芯片支持网络都是指内部有MAC

MAC就类似于SPI控制芯片, 还需要外部搭载一个PHY芯片

### 实现的方法

内部没有MAC, 可以使用外部MAC+PHY芯片, W5500可以通过SPI进行联网, 内部集成了TCP/IP协议, 但是网络速度较低, 成本也要高一点

内部集成了MAC外设, 直接外接PHY芯片就可以, 一般常见的单片机都会集成MAC外设, 有专门的加速模块, DMA的等, 支持的网速快, 可以支持的PHY种类多, 通过MII和RMII等接口连接外部的芯片, 用来传输网络数据, 还有用来控制PHY的控制接口, 有两根线, MDIO和MDC(时钟线), 

### MII和RMII

+   MII

用于连接MAC和PHY芯片, 有16根线

TX_CLK：发送时钟，如果网速为100M的话时钟频率为25MHz，10M网速的话时钟频率为2.5MHz，此时钟由PHY产生并发送给MAC。

TX_EN：发送使能信号。

TX_ER：发送错误信号，高电平有效，表示TX_ER有效期内传输的数据无效。10Mpbs网速下TX_ER不起作用。

TXD[3:0]：发送数据信号线，一共4根。

RXD[3:0]：接收数据信号线，一共4根。

RX_CLK：接收时钟信号，如果网速为100M的话时钟频率为25MHz，10M网速的话时钟频率为2.5MHz，RX_CLK也是由PHY产生的。

RX_ER：接收错误信号，高电平有效，表示RX_ER有效期内传输的数据无效。10Mpbs网速下RX_ER不起作用。

RX_DV：接收数据有效，作用类似TX_EN。

CRS：载波侦听信号。

COL：冲突检测信号。

+   RMII

精简以后只有9根线

TX_EN：发送使能信号。

TXD[1:0]：发送数据信号线，一共2根。

RXD[1:0]：接收数据信号线，一共2根。

CRS_DV：相当于MII接口中的RX_DV和CRS这两个信号的混合。

REF_CLK：参考时钟，由外部时钟源提供，频率为50MHz。这里与MII不同，MII的接收和发送时钟是独立分开的，而且都是由PHY芯片提供的。

### MDIO接口

管理数据输入输出接口, 有两根线, MDIO和MDC线, 最多支持32个PHY

### RJ45接口

用来链接网网线, 和PHY芯片连接的时候需要一个网络变压器, 用于隔离滤波等作用

有的RJ45座子会内置一个网络变压器

一般有两个灯, 绿灯代表网络连接正常, 黄色的代表数据正在传输,  

### PHY芯片

是标准规定下的标准模块, 寄存器地址空间有5位, 32个寄存器, 协议定义了前16个寄存器, 要求使用这个实现功能, 可以写出通用的驱动, 但是还是需要一些小的修改

#### 常用寄存器

BCR寄存器, 标号为0, bit15软件复位, bit14回测模式, bit13速度选择, bit12自动协商, bit11掉电模式, bit10隔离, bit9重启自动协商, bit8双工模式

BSR(Basic Status Register)寄存器，地址为1。此寄存器为PHY的状态寄存器，通过此寄存器可以获取到PHY芯片的工作状态



## 内核驱动框架

使用net_device表示一个网络设备, 定义在include/linux/netdevice.h中

+   重要的成员

```c
const struct net_device_ops *netdev_ops;
```

>   网络设备的操作集, 一系列的网络设备操作回调函数，类似字符设备中的file_operations

```c
const struct ethtool_ops *ethtool_ops;
```

>   网络工具相关的, 函数集，用户空间网络管理工具会调用此结构体中的相关函数获取网卡状态或者配置网卡。

```c
unsigned int		flags;
```

>   网络标志

````c
unsigned char		if_port;
````

>   可选的网络类型



+   实现

```
#define alloc_netdev(sizeof_priv,name,name_assign_type,setup)\
alloc_netdev_mqs(sizeof_priv,name,name_assign_type,setup,1,1)
```

>   申请一个net_device结构体, sizeof_priv：私有数据块大小。name：设备名字eth0等。setup：回调函数，初始化设备的设备后调用此函数。txqs：分配的发送队列数量。rxqs：分配的接收队列数量。返回值：如果申请成功的话就返回申请到的net_device指针，失败的话就返回NULL。可以支持多种网络设备, 包括以太网, CAN网络, WIFI等

```c
#define alloc_etherdev(sizeof_priv) alloc_etherdev_mq(sizeof_priv,1)
#define alloc_etherdev_mq(sizeof_priv,count) alloc_etherdev_mqs(sizeof_priv,count,count)
```

>   专门用于以太网的数据结构体的申请

```c
struct net_device    *alloc_netdev_mqs ( int sizeof_priv, 			const char *name,
        void (*setup) (struct 	net_device *))
    	unsigned int txqs, 
		unsigned int rxqs);
```

>   这就是最后实际调用的的函数, sizeof_priv：私有数据块大小。name：设备名字。setup：回调函数，初始化设备的设备后调用此函数, 以太网的初始化函数是设置寄存器的各位的数据。txqs：分配的发送队列数量。rxqs：分配的接收队列数量。

```c
void free_netdev(struct net_device *dev)
```

>   用于释放网络设备

+   net_device_ops

网络设备的操作集。需要网络驱动编写人员去实现，不需要全部都实现，根据实际驱动情况实现其中一部分即可

```c
int (*ndo_init) (structnet_device *dev);
```

>    当第一次注册网络设备的时候此函数会执行，设备可以在此函数中做一些需要退后初始化的内容，不过一般驱动中不使用此函数，虚拟网络设备可能会使用

```c
int (*ndo_open) (structnet_device *dev);
int (*ndo_stop) (structnet_device *dev);
```

>   打开网络设备的时候此函数会执行，网络驱动程序需要实现此函数，非常重要
>
>   关闭网络设备的时候此函数会执行，网络驱动程序也需要实现此函数

```c
netdev_tx_t (*ndo_start_xmit) (struct sk_buff *skb,  struct net_device *dev);
```

>   需要发送数据的时候会调用的函数, 此函数有一个参数为sk_buff结构体指针，sk_buff结构体在Linux的网络驱动中非常重要，sk_buff保存了上层传递给网络驱动层的数据



+   sk_buff

网络是分层的, 各个协议层在sk_buff中添加自己的协议头，最终由底层驱动讲sk_buff中的数据发送出去。网络数据的接收过程恰好相反，网络底层驱动将接收到的原始数据打包成sk_buff，然后发送给上层协议，上层会取掉相应的头部，然后将最终的数据发送给用户

```c
static inline int dev_queue_xmit(struct sk_buff *skb)
```

>   在内核中通过这个把数据发送出去, 最后调用的是ndo_start_xmit函数, 

```c
int netif_rx(struct sk_buff *skb)
```

>   上层接收数据的话使用netif_rx函数，但是最原始的网络数据一般是通过轮询、中断或NAPI的方式来接收



**主要的重要参数:**

```c
sk_buff_data_t		tail;
sk_buff_data_t		end;
unsigned char		*head,
		*data;
```

>   head指向缓冲区的头部，data指向实际数据的头部。data和tail指向实际数据的头部和尾部，head和end指向缓冲区的头部和尾部。



**申请释放:** 

```c
static inline struct sk_buff *alloc_skb(unsigned int size,gfp_t priority)
```

>   申请一个sk_buff, size：要分配的大小，也就是skb数据段大小。priority：为GFP MASK宏，比如GFP_KERNEL、GFP_ATOMIC等。

```c
static inline struct sk_buff *netdev_alloc_skb(struct net_device 						*dev,unsigned int length)
```

>   常常使用netdev_alloc_skb来为某个设备申请一个用于接收的skb_buff, dev：要给哪个设备分配sk_buff。length：要分配的大小。

```c
void kfree_skb(struct sk_buff *skb)
```

>   用于释放sk_buff

```c
void dev_kfree_skb(struct sk_buff *skb)
```

>   对于网络设备而言最好使用这个进行释放

**变更结构体:**

```c
unsigned char *skb_put(struct sk_buff *skb, unsigned int len)
```

>   将skb_buff的tail后移n个字节，从而导致skb_buff的len增加n个字节, skb：要操作的sk_buff。len：要增加多少个字节。返回值：扩展出来的那一段数据区首地址。

```c
unsigned char *skb_push(struct sk_buff *skb, unsigned int len)
```

>   在头部扩展skb_buff的数据区, skb：要操作的sk_buff。len：要增加多少个字节。返回值：扩展完成以后新的数据区首地址。

```c
unsigned char *skb_pull(struct sk_buff *skb, unsigned int len)
```

>   从sk_buff的数据区起始位置删除数据, skb：要操作的sk_buff。len：要删除的字节数。返回值：删除以后新的数据区首地址。

```c
static inline void skb_reserve(struct sk_buff *skb, int len)
```

>   调整缓冲区的头部大小，方法很简单讲skb_buff的data和tail同时后移n个字节即可

## 网络NAPI处理机制

NAPI是一种高效的网络处理技术。NAPI的核心思想就是不全部采用中断来读取网络数据，而是采用中断来唤醒数据接收服务程序，在接收服务程序中采用POLL的方法来轮询处理数据。这种方法的好处就是可以提高短数据包的接收效率，减少中断处理的时间。

### 初始化

初始化一个napi_struct实例

```c
void netif_napi_add(struct net_device *dev, struct napi_struct 			*napi,int (*poll)(struct napi_struct *, int), int weight)
```

>   dev：每个NAPI必须关联一个网络设备，此参数指定NAPI要关联的网络设备。napi：要初始化的NAPI实例。poll：NAPI所使用的轮询函数，非常重要，一般在此轮询函数中完成网络数据接收的工作。weight：NAPI默认权重(weight)，一般为NAPI_POLL_WEIGHT。

```c
void netif_napi_del(struct napi_struct *napi)
```

>   删除

```c
inline void napi_enable(struct napi_struct *n)
```

>   使能

```c
void napi_disable(struct napi_struct *n)
```

>   关闭

```c
inline bool napi_schedule_prep(struct napi_struct *n)
```

>   检查是否可以调度

```c
void __napi_schedule(struct napi_struct *n)
```

>   实际的调度

```c
inline void napi_complete(struct napi_struct *n)
```

>   处理完成





## 设备树

```v
&fec2 {
	pinctrl-names = "default";
	pinctrl-0 = <&pinctrl_enet2>;
	phy-mode = "rmii";
	phy-handle = <&ethphy1>;
	phy-reset-gpios = <&gpio5 8 GPIO_ACTIVE_LOW>;
	phy-reset-duration = <26>;
	status = "okay";

	mdio {
		#address-cells = <1>;
		#size-cells = <0>;

		ethphy0: ethernet-phy@0 {
			compatible = "ethernet-phy-ieee802.3-c22";
			reg = <0>;
		};

		ethphy1: ethernet-phy@1 {
			compatible = "ethernet-phy-ieee802.3-c22";
			reg = <1>;
		};
	};
};
```

mdio：可以设置名为“mdio”的子节点，此子节点用于指定网络外设所使用的MDIO总线，主要做为PHY节点的容器，也就是在mdio子节点下指定PHY相关的属性信息，具体信息可以参考PHY的绑定文档Documentation/devicetree/bindings/net/phy.txt。



### MDIO总线注册

用来管理PHY芯片的，分为MDIO和MDC两根线，Linux内核专门为MDIO准备一个总线，叫做MDIO总线，采用mii_bus结构体表示

## PHY驱动

需要我们实现的

phy_driver就是PHY的驱动函数, 



使能驱动

-->Device Driver -->Networlk device support --> PHY Device support and infrastructure -->Driver' for SMSC PHYs







