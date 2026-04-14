# Qemu Debug

Kernel hacking ---> Compile-time checks and compiler options 

![image-20260407100033803](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/lenovo-picture/202604071000874.png)

![image-20260407100107667](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/lenovo-picture/202604071001731.png)

```bash
# 核心：开启完整调试信息（必须项！GDB调试的基础）
CONFIG_DEBUG_INFO=y

# 关闭裁剪调试信息（正确，保留完整符号）
# CONFIG_DEBUG_INFO_REDUCED is not set

# 关闭压缩调试信息（正确，不压缩符号，GDB直接读取）
# CONFIG_DEBUG_INFO_COMPRESSED is not set

# 关闭拆分调试信息（正确，单文件符号更稳定）
# CONFIG_DEBUG_INFO_SPLIT is not set

# 启用DWARF4调试格式（GDB完美兼容，现代内核标准配置）
CONFIG_DEBUG_INFO_DWARF4=y

# BTF调试信息（不影响GDB，开着无害，可忽略）
CONFIG_DEBUG_INFO_BTF=y
```

| 选项组合                                         | 适用场景                    | 效果                                                  |
| ------------------------------------------------ | --------------------------- | ----------------------------------------------------- |
| 仅 CONFIG_DEBUG_INFO=y                           | 基础内核调试 / 崩溃分析     | 生成默认版本 DWARF 调试信息，支持基本 GDB 调试        |
| CONFIG_DEBUG_INFO=y + CONFIG_DEBUG_INFO_DWARF4=y | 优化代码调试 / 复杂内核问题 | 生成更完善的 DWARF 4 信息，提升变量解析和工具兼容性   |
| 三者全启用                                       | 深度内核开发 / 内核调试     | 完整调试信息 + DWARF 4+GDB 辅助脚本，提供最佳调试体验 |

1. **完整符号表**：包含函数名、变量名、结构体定义等调试元数据，让调试器能将内存地址映射到可读的符号。
2. **源代码级调试**：在 GDB 等调试器中直接查看对应源代码和行号，支持断点、单步执行等操作。
3. **精确堆栈跟踪**：内核崩溃（Oops/Panic）时，能生成带函数调用链的详细报告，便于定位问题。
4. **支持崩溃分析工具**：kdump、crash 等工具需要带调试符号的`vmlinux`文件，才能解析内核转储文件。

关闭内核地址随机化: 每次开机内核实际运行地址都不一样，黑客手里的固定地址彻底失效，漏洞很难利用

Kernel hacking` → 关掉 `Randomize the kernel address space, 设置CON

FIG_SYSTEM_TRUSTED_KEYS为空, CONFIG_SYSTEM_REVOCATION_LIST关掉

```
CONFIG_RANDOMIZE_BASE
CONFIG_SYSTEM_TRUSTED_KEYS
CONFIG_SYSTEM_REVOCATION_LIST
```

默认使用的是x86的架构

```bash
apt update && apt install -y bc gcc make libssl-dev bison flex libelf-dev dwarves qemu-system-x86_64 cpio xz-utils bzip2 xz-utils openssl qemu qemu-utils qemu-kvm virt-manager libvirt-daemon-system libvirt-clients bridge-utils texinfo build-essential



make  -j4 bzImage
make -j4 modules  # 使用4个线程编译内核模块
make scripts_gdb
```



## 根文件系统

使用busybox进行构建

```bash
wget https://busybox.net/downloads/busybox-1.32.1.tar.bz2
tar -xvf busybox-1.32.1.tar.bz2
cd busybox-1.32.1/
make menuconfig
# 选择 CONFIG_STATIC
# 安装完成后生成的相关文件会在 _install 目录下
make && make install
```

### initramfs

initramfs（Initial RAM filesystem）是 “基于内存的初始根文件系统”，是 Linux 系统启动过程中不可或缺的一个临时文件系统。

可以把它理解为 “系统启动的临时工具箱”：

内核（Kernel）启动后，第一时间加载 initramfs；

+ 内核把它当作临时的根目录（/）；包含最核心的驱动、工具和脚本，帮助系统完成硬件初始化、加载驱动、挂载真正的根文件系统；
+ 挂载完成后，initramfs 会自动切换到真正的根文件系统，自己则被 “丢弃”（或保留为备用）。

Linux 内核设计的核心原则是 “最小化内” —— 内核本身只包含最核心的功能，比如进程调度、内存管理、VFS（虚拟文件系统）、PCI/USB 总线等。

+ 但是，系统要正常运行，必须解决两个问题：

怎么识别根文件系统设备？（比如你的根文件系统在 /dev/sda2、/dev/mmcblk0p2 还是 NVMe 硬盘上？）
怎么加载访问根文件系统所需的驱动？（比如 SATA、NVMe、SD 卡、USB 存储的驱动？）

+ 如果没有 initramfs：

内核必须内置所有硬件驱动（这会导致内核体积巨大，且无法适配不同硬件）；
或者你必须在编译内核时就精确知道根设备路径（不灵活）。

+ 有了 initramfs 之后：

内核只需要加载 initramfs（一个小的压缩文件）；
initramfs 里提供了 udev（设备管理器）和 各类硬件驱动模块；
它能自动扫描硬件，加载所需的驱动；
最后找到并挂载真正的根文件系统。

#### 实际实现

BusyBox 原始的 `_install` 目录不能直接用！它只是个半成品,它只包含了 `ls`、`cd`、`sh` 这些命令，**缺少内核启动必需的 4 个关键东西**：

1. **缺少 `/dev` 设备文件**

没有控制台、终端，内核启动后你**看不到任何日志，也无法敲命令**，直接黑屏卡死。

2. **缺少 `/proc` `/sys` 目录**

内核运行必需的虚拟文件系统，没有会直接报错崩溃。

3. **缺少自定义的 `init` 启动脚本**

内核启动后必须运行第一个程序 `init`，BusyBox 自带的不适合调试。

4. **没有配置启动环境**

不会自动挂载文件系统，不会进入命令行。

```bash
# 1. 创建一个空文件夹，专门用来做最终的根文件系统
mkdir initramfs
cd initramfs

# 2. 把 BusyBox 编译好的工具（ls、cd、mkdir、sh 等）复制过来
# 这只是系统的「工具包」，不是完整系统
cp ../busybox-1.32.1/_install/* -rf ./

# 3. 创建 3 个 Linux 系统必须的空目录（内核启动必须要）
mkdir dev proc sys

# 4. 拷贝「终端/控制台设备」！没有这个，你无法在QEMU里看到输出、敲命令
# cp -a 复制的不是「数据」，而是设备节点本身的属性, 构建一个模板
# 让虚拟内核能找到 QEMU 提供的虚拟终端
sudo cp -a /dev/{null,console,tty,tty1,tty2,tty3,tty4} dev/

# 5. 删除 BusyBox 自带的旧启动文件（我们不用它，用自己写的init）
rm -f linuxrc

# 6. 创建你刚才问的那个「启动脚本init」（内核启动后第一个运行的文件）
vim init

# 7. 给启动脚本加可执行权限（不加内核没法运行）
chmod a+x init
```

init的程序是

```bash
#!/bin/busybox sh
echo "{==DBG==} INIT SCRIPT"
mount -t proc none /proc
mount -t sysfs none /sys
 
echo -e "{==DBG==} Boot took $(cut -d' ' -f1 /proc/uptime) seconds"
exec /sbin/init
```

完成以后打包

```bash
find . -print0 | cpio --null -ov --format=newc | gzip -9 > ../initramfs.cpio.gz
```

`-print0`：用特殊字符分隔文件名（防止文件名带空格报错，安全写法）

`cpio` = Linux 专用打包工具（内核只认 cpio 格式的文件系统）

+ `--null`：匹配前面的 `-print0`
+ `-o`：创建打包文件
+ `-v`：显示打包过程（看得见哪些文件被打包）
+ `--format=newc`：**最重要！** 内核专用格式，必须加

用 `gzip` 压缩打包好的文件

`-9`：最高压缩比（让镜像更小，启动更快）

## 启动

> ```bash
> mkdir -p ~/.config/gdb/
> 
> # 关闭安全限制（调试专用，无风险）
> echo "set auto-load safe-path /" >> ~/.config/gdb/gdbinit
> ```
>
> 

```bash
qemu-system-x86_64 -kernel ./arch/x86/boot/bzImage -initrd ../initramfs.cpio.gz -append "nokaslr console=ttyS0" -s -S -nographic
```

- `-kernel ./arch/x86/boot/bzImage`：指定启用的内核镜像；
- `-initrd ../initramfs.cpio.gz`：指定启动的内存文件系统；
- `-append "nokaslr console=ttyS0"` ：附加参数，其中 `nokaslr` 参数**必须添加进来**，防止内核起始地址随机化，这样会导致 gdb 断点不能命中
- `-s` ：监听在 gdb 1234 端口；
- `-S` ：表示启动后就挂起，等待 gdb 连接；
- `-nographic`：不启动图形界面，调试信息输出到终端与参数 `console=ttyS0` 组合使用；

```bash
cd ~/kernel-debug/linux-5.10.252
gdb ./vmlinux
target remote localhost:1234
b start_kernel
c
```

