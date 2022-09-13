# 主频

CPU Power Mangagement->CPU Frequency scaling -> Default CPUFreq governor (ondemand)  --->进行设置主频的模式

# 网络驱动

Device Drivers  --->[\*] Network device support  ---> -*-   PHY Device support and infrastructure  ---><*>   Drivers for SMSC PHYs

# 系统频率

-> Kernel Features -> Timer frequency (<choice> [=y])  

# 音频驱动

图形界面, 取消ALSA模拟OSS

Device dirver->Sound card support--> <*>   Advanced Linux Sound Architecture  

之后使能对应的驱动

Device dirver->Sound card support--> <*>   Advanced Linux Sound Architecture  --><\*>   ALSA for SoC audio support  --->  SoC Audio for Freescale CPUs  --->< > SoC Audio support for i.MX boards with wm8960 

# LED驱动

-->Device Driver-->LED Support --> LED Support for GPIO connected LEDs 

# logo显示

Device Drivers  --->Graphics support  --->logo

# 按键驱动

Device driver --> Input device support --> -*- Generic input layer (needed for keyboard, mouse, ...) , keyboard --> GPIO Buttons

