# 不能下载

首先查看ip地址, 检查主机是不是可以正常的上网, 排除主机的错误

如果可以ping ubuntu单数不可以下载

之后测试是不是有两个设备使用同一个ip, 检查开发板以及主机是不是可以正常使用

# 下载启动

```
	setenv bootcmd 'tftp 80800000 zImage; tftp 83000000 imx6ull-14x14-emmc-7-1024x600-c.dtb; bootz 80800000 -83000000'
	saveenv
	boot
```

# linux下使用

```
/lib/modules # tftp -g -r chardevbase.ko 192.168.31.187
chardevbase.ko       100% |********************************|  2827  0:00:00 ETA

上传
tftp -p -l c:\User\Administrator\Download 1.2.3.4
```





