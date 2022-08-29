---
layout: post
title: "pinctl和gpio子系统" 
date:   2022-8-27 15:39:08 +0800
tags: 嵌入式 原子Linux 
---

# pinctl和gpio子系统

在开发的时候不用直接操作寄存器, Linux提供了gpio操作的方法

## pinctl子系统

在使用时候的, 之前要设置PIN的复用和电气属性, 配置GPIO

Linux使用pinctl子系统进行管理, 设置PIN的电气属性和复用

+   主要功能

从设备树获取pin的信息, 根据设置进行设置复用, 设置各种属性, 厂商已经写好了, 对于用户来说就是在设备树中添加对应的配置信息

+   IOMUX SNVS控制器

```c
iomuxc_snvs: iomuxc-snvs@02290000 {
    compatible = "fsl,imx6ull-iomuxc-snvs";
    reg = <0x02290000 0x10000>;
};
```

+   IOMUXC控制器

```c
iomuxc: iomuxc@020e0000 {
    compatible = "fsl,imx6ul-iomuxc";
    reg = <0x020e0000 0x4000>;
};
```

>   在dts文件中进行追加, 根据设备的类型创建对应的子节点, , 然后把用到的io放到对应的节点下面
>
>   ```c
>   pinctrl_csi1: csi1grp {
>       fsl,pins = <
>           MX6UL_PAD_CSI_MCLK__CSI_MCLK		0x1b088
>           MX6UL_PAD_CSI_PIXCLK__CSI_PIXCLK	0x1b088
>           MX6UL_PAD_CSI_VSYNC__CSI_VSYNC		0x1b088
>           MX6UL_PAD_CSI_HSYNC__CSI_HSYNC		0x1b088
>           ...
>           >;
>   };
>   ```

>   后面的数字就是电气属性

**如何添加一个PIN**

前面的是一个宏, 定义在imx6ull-pinfunc.h, 

```c
#define MX6UL_PAD_CSI_MCLK__CSI_MCLK  0x01D4 0x0460 0x0000 0x0 0x0
```

>   <mux_reg  conf_reg  input_reg  mux_mode  input_val>, 第一个是设置IO复用的寄存器, 是对于父节点的偏移, 第四个是第几个复用, 第二个是电气属性寄存器, 第三个偏移是0表示没有这个寄存器, 最后一个值是写入这个寄存器的

每种功能对应一个宏, 名字前面是IO名字, 后面是复用的功能

+   GPR控制器

```c
gpr: iomuxc-gpr@020e4000 {
    compatible = "fsl,imx6ul-iomuxc-gpr",
    "fsl,imx6q-iomuxc-gpr", "syscon";
    reg = <0x020e4000 0x4000>;
};
```



+   如何找到对应的驱动

不同的芯片由半导体厂商写出来不同的驱动, 通过compatible属性进行匹配, 这个属性是一个字符串列表, 当设备树的节点的属性和驱动里面的匹配的时候就代表两者匹配了, 根据这个可以找到pinctrl-imx6ull文件, 这个文件就是6ull的pinctl文件

当驱动设备匹配的时候probe函数执行, 





## gpio子系统

用来对gpio的操作, 复用为GPIO的时候使用这个

```
&usdhc1 {
	pinctrl-names = "default", "state_100mhz", "state_200mhz";
	pinctrl-0 = <&pinctrl_usdhc1>;
	pinctrl-1 = <&pinctrl_usdhc1_100mhz>;
	pinctrl-2 = <&pinctrl_usdhc1_200mhz>;
	cd-gpios = <&gpio1 19 GPIO_ACTIVE_LOW>;
	keep-power-in-suspend;
	enable-sdio-wakeup;
	vmmc-supply = <&reg_sd1_vmmc>;
	status = "okay";
};
```

**pinctrl-0,1,2**: 设置几种不同的IO属性, 主要是电气属性不同, 是pinctl的参数设置

+   **cd-gpios = <&gpio1 19 GPIO_ACTIVE_LOW>;**==重点==

定义了一个属性, 重点是后面的值, 比如此处使用GPIO1_19, 对应的结构体

```
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

compatible: "fsl, <soc>-gpio"

reg: 寄存器范围

gpio-controller: 这是一个gpio控制器

#gpio-cells = <2>;: 必须是2, 第一个是gpio的编号, 第二个是gpio的极性, 高电平为0, 低电平为1

**cd-gpios = <&gpio1 19 GPIO_ACTIVE_LOW>;**

表示引用gpio1, 之后就是两个cell



### 函数



```c
int gpio_request(unsigned gpio, const char *label)
```

>   申请一个GPIO管脚，在使用一个GPIO之前一定要使用gpio_request进行申请, gpio, 要申请的gpio编号, label设置gpio的名字, 主要是用来检测有没有被别的设备使用, 正常返回0

```c
void gpio_free(unsigned gpio)
```

>   释放一个gpio的标号

```c
int gpio_direction_input(unsigned gpio)
```

>   设置gpio为输入

```c
int gpio_direction_output(unsigned gpio, int value)
```

>   设置某个GPIO为输出，并且设置默认输出值, 正常的时候返回0

```c
#define gpio_get_value __gpio_get_value int __gpio_get_value(unsigned gpio)
```

>   获得某一个gpio的值

```c
#define gpio_set_value __gpio_set_valuevoid __gpio_set_value(unsigned gpio, int value)
```

>   设置gpio的值

### 相关的of函数

```c
int of_gpio_named_count(struct device_node *np,const char    *propname)
```

>   获取设备树某个属性里面定义了几个GPIO信息, 空的GPIO信息也会被统, propname：要统计的GPIO属性

```c
int of_gpio_count(struct device_node *np)
```

>   统计任意属性的GPIO信息

```c
int of_get_named_gpio(struct device_node *np,const char *propname, int index)
```

>   类似<&gpio5 7 GPIO_ACTIVE_LOW>的属性信息转换为对应的GPIO编号, np：设备节点。propname：包含要获取GPIO信息的属性名, index：GPIO索引，因为一个属性里面可能包含多个GPIO，此参数指定要获取哪个GPIO的编号，如果只有一个GPIO信息的话此参数为0

### 实际过程

首先获取到gpio所处的节点of_find_node_by_path

之后获取GPIO编号, of_get_named_gpio函数返回值就是编号

之后进行请求gpio_request

使用, 设置输入输出, 之后进行读写

使用完成后释放gpio_free



## gpio驱动

### gpio_lib

两部分, 给原厂编写GPIO底层驱动, 给驱动力开发人员的操作函数



### gpio驱动

在drivers/gpio目录下, gpio-xxx.c文件为具体的的的gpio驱动, gpiolib是Linux内核自己写的

gpiolib-legacy.c文件





## 失败

申请的时候失败大部分原因是被其他的外设占用了, 检查设备树

一 检查复用设置

二 gpio使用

## 总结

添加pinctrl相关信息

检查当前设备的IO使用

添加设备节点, 在节创建属性, 描述使用的gpio

编写驱动

## 蜂鸣器

注意使用的引脚, 使用MX6ULL_PAD_SNVS_TAMPER1__GPIO5_IO01, 6UL和6ULL的引脚不同







