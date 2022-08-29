---
layout: post
title: "设备树" 
date:   2022-8-27 15:39:08 +0800
tags: 嵌入式 原子Linux 
---

# 设备树

Linux在ARM中使用设备树

是用一个树形的结构进行描述各种设备, 让内核可以解释分析, 使用单独的文件进行描述

在内核中arch/arm中有大量的`mach`开头的文件, 这些是不同的机器, plat开头的代表不同的平台

使用大量的结构体, 在裸机进行的时候直接写死了, 在.c文件中, 最终导致大量的重复内核无用的信息, 不能编译到Linux内核之中

文件的扩展名为`.dts`, 相当于.c, 就是DTC的源码文件, DTC相当于gcc编译器, 生成.deb文件

我们使用的时候使用`make dtbs`把所有的编译, 编译指定的`make 文件名.dtb`

也有头文件`.dtsi`文件

## DTS语法

有一个斜杠开始, 作为根,  可以使用追加

```
/dts-v1/;
```

版本1的`dts`文件, 是设备数的开头

```
	model = "Freescale i.MX6SLL EVK Board";
	compatible = "fsl,imx6sll-evk", "fsl,imx6sll";
```

+   根节点相关的属性, 属性名+属性值, 一般都`dtsi`文件是由芯片厂商提供, 从根节点开始描述设备信息

+   根节点外有一些&cpu0 之类是追加

+   使用#include进行引进的dtsi文件都是使用同一个根节点

+   注释使用的跟C语言一样

+   使用同一个名字的节点会加到一起
+   同一个节点同名的属性会进行覆盖

+   节点的名字`node-name@unit-address`, 节点名字有数字, 字母, 逗号, +, -, 下划线,句号, 大多数使用小写字母, 之后@单元地址, 一般都是外设的起始地址, 有时候可能为I2C设备地址等
+   还有使用`label:node-name@unit-address`, label方便访问节点, 直接通过&label进行访问

## 设备树在Linux中的体现

系统启动以后可以在根文件系统中看到设备树节点的信息, 在目录/proc/device-tree目录

描述外设信息的

描述6ull芯片的内部外设寄存器节点, 在soc的aips2中存在I2C外设

内核启动的时候回解析设备树

## 特殊的节点

chosen:为了uboot向Linux传递参数, 传递bootargs, 是boot的fdt_chosen查找节点, 添加属性

aliases(别名): 在实际的使用的时候很少, 是为了方便访问节点, 最后的作用就是为了在/dev下面生成 

## 属性

不同的设备属性不同, 以及根据驱动进行描述

### 标准属性

+   compatible: 兼容的, 每一个节点都有, 是一个字符串, 格式为`"manufacturer,model"`, 第一个是厂商, 第二个是设备名, 可以拥有多个
+   model: 描述模块的信息
+   status: 描述状态, "okey"可用, "disable"不可操作, "fail"不可操作, 检测到错误, 也不大可能可操作, "fail-sss", 同上sss为检测到的错误
+   #address-cells和#size-cells, 用于描述地址属性, 分别描述reg的两个参数的长度, 单位为32位, 都是父节点描述子节点

```
#address-cells = <1>;
#size-cells = <0>;
```

```
reg= <address1 length1 address2 length2 address3 length3......>
```

+   reg: 大多数是描述一段内存
+   ranges属性: `(child-bus-address,parent-bus-address,length)`, 子地址, 父地址, 地址空间长度, 是一个地址映射转换表, 在arm很少使用
+   name属性, 已经被弃用了
+   device_type: 描述设备的FCode, 弃用

### 特殊的属性

根节点下compatible用来检测内核是不是支持这个平台

使用设备树之前使用机器ID进行控制, 文件`include/generated/mach-types.h`中，此文件定义了大量的machine id, 使用MACHINE_START和MACHINE_END 在include/generated/mach-types.h定义了支持的ID

使用之后用根节点下的这个属性, DT_MACHINE_START。DT_MACHINE_START定义在文件`arch/arm/include/asm/mach/arch.h`里面, 但是都是使用同一个结构体, 对应内核中的文件`arch/arm/mach-imx/mach-imx6ul.c`中的属性就可以兼容

### 绑定信息文档

有一些设备描述有特定的描述方式

`document/devicetree/bindings/`文件夹下有多种描述, 只是一个参考, 不完善, 找芯片的厂家找

## OF操作函数

驱动获取设备树节点信息

在驱动中使用OF函数获取设备树的内容, `include/linux/of.h`文件中

+   设备是以节点的形式挂载到设备树上面的, 使用结构体描述一个节点

```c
struct device_node {
	const char *name;
	const char *type;
	phandle phandle;
	const char *full_name;
	struct fwnode_handle fwnode;

	struct	property *properties;
	struct	property *deadprops;	/* removed properties */
	struct	device_node *parent;
	struct	device_node *child;
	struct	device_node *sibling;
	struct	kobject kobj;
	unsigned long _flags;
	void	*data;
#if defined(CONFIG_SPARC)
	const char *path_component_name;
	unsigned int unique_id;
	struct of_irq_controller *irq_trans;
#endif
};
```

+   驱动想获取到设备树的内容, 首先要找到节点, 常用的的函数有

```c
extern struct device_node *of_find_node_by_name(struct device_node *from,const char *name);
```

>   from: 从哪一个节点开始为NULL就是从根节点开始, name查找的节点的名字

```c
extern struct device_node *of_find_compatible_node(struct device_node *from, const char *type, const char *compat);
```

>   根据device_type和compatible这两个属性, NULL表示忽略

```c
static inline struct device_node *of_find_node_by_path(const char *path)
```

>   通过路径

+   查找父节点子节点

```c
extern struct device_node *of_get_parent(const struct device_node *node);
```

```c
extern struct device_node *of_get_next_child(const struct device_node *node,struct device_node *prev);
```

>   前一个子节点，也就是从哪一个子节点开始迭代的查找下一个子节点。可以设置为NULL，表示从第一个子节点开始

+   提取属性

使用一个结构体来表示属性

```c
struct property {
	char	*name;	//属性的名字
	int	length;		//属性长度
	void	*value; //属性的值
	struct property *next;	//下一个属性
	unsigned long _flags;
	unsigned int unique_id;
	struct bin_attribute attr;
};
```

+   查找属性的函数

```c
extern struct property *of_find_property(const struct device_node *np, const char *name, int *lenp);
```

>   查找结构体, 参数一, 节点, 参数二, 节点的名字, 第三个参数的长度, 一般为NULL, 返回值为结构体

```c
extern int of_property_count_elems_of_size(const struct device_node *np, const char *propname, int elem_size);
```

>   获取属性中元素的数量，比如reg属性值是一个数组，那么使用此函数可以获取到这个数组的大小, np: 设备节点。proname：需要统计元素数量的属性名字。elem_size: 元素长度, 一般都是用sizeof(u32), 返回值为结果

```c
extern int of_property_read_u32_index(const struct device_node *np, const char *propname, u32 index, u32 *out_value);
```

>   从属性中获取指定标号的u32类型数据值, np: 设备节点。proname: 要读取的属性名字。index: 要读取的值标号。out_value: 读取到的值, 返回值是是否读取成功

```
extern int of_property_read_u8_array(const struct device_node *np,const char *propname, u8 *out_values, size_t sz);
extern int of_property_read_u16_array(const struct device_node *np,const char *propname, u16 *out_values, size_t sz);
extern int of_property_read_u32_array(const struct device_node *np, const char *propname, u32 *out_values,size_t sz);
extern int of_property_read_u64(const struct device_node *np, const char *propname, u64 *out_value);
```

>   查找一个数组, 节点, 名字, 返回的数组, 读取的大小

```c
extern int of_property_read_string(struct device_node *np, const char *propname,const char **out_string);
```

>   读取字符串

```c
extern int of_n_addr_cells(struct device_node *np);
extern int of_n_size_cells(struct device_node *np);
```

>   \#address-cells \#size-cells

+   其他的常用的属性

```c
static inline int of_device_is_compatible(const struct device_node *device,const char *name)
```

>   查看节点的compatible属性是否有包含compat指定的字符串，也就是检查设备节点的兼容性==好像没有实现==

```c
const __be32 *of_get_address(struct device_node *dev, intindex, u64 *size,unsigned int *flags)
```

>   用于获取地址相关属性，主要是“reg”或者“assigned-addresses”属性值

```c
u64 of_translate_address(struct device_node *dev, const __be32 *in_addr)
```

>   负责将从设备树读取到的地址转换为物理地址

```c
void __iomem *of_iomap(struct device_node *np, int index)
```

>   直接内存映射, 直接从节点进行操作, 对象是reg, 每两个是一对, 索引从0开始



```c
kmalloc
```

>   内核申请内存, 参数一大小, 参数二flags为GFP_KERNEL时候正常申请

```
kfree
```

>   释放













