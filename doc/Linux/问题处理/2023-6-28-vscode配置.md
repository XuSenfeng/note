# 设备文件设置

```
{
    "configurations": [
        {
            "name": "Linux",
            "includePath": [
                "${workspaceFolder}/**",
                "/home/jiao/linux/IMX6ULL/linux/linux-imx-...",     //这里是头文件路径
                "/home/jiao/linux/IMX6ULL/linux/linux-imx-...", 
                "/home/jiao/linux/IMX6ULL/linux/linux-imx-..."
            ],
            "defines": ["__KERNEL__"],			//添加结构体的补全
            "compilerPath": "/usr/bin/gcc",
            "cStandard": "c11",
            "cppStandard": "c++17",
            "intelliSenseMode": "clang-x64"
        }
    ],
    "version": 4
}
```

