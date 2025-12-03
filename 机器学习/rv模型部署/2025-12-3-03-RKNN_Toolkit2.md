# RKNN Toolkit2

[RKNN Toolkit2](https://github.com/airockchip/rknn-toolkit2) 开发套件(Python接口)运行在PC平台（x86/arm64），提供了模型转换、 量化功能、模型推理、性能和内存评估、量化精度分析、模型加密等功能。 

>   [相关的手册](https://github.com/airockchip/rknn-toolkit2/tree/master/doc)

## 使用

流程简单描述：

-   创建RKNN对象，初始化RKNN环境
-   设置模型预处理参数，如果是运行在PC上，通过模拟器运行模型时需要调用config接口设置模型的预处理参数；如果运行在连接的板卡NPU上并且导入RKNN模型，不需要配置。
-   导入模型，如果是运行在PC上，通过模拟器运行模型时使用load_caffe、load_tensorflow等接口导入对应的非RKNN模型，通过；如果运行在连接的板卡NPU使用接口load_rknn导入RKNN模型。
-   构建RKNN模型，如果是运行在PC上，通过模拟器运行模型，需要调用build接口构建RKNN模型，然后可以导出RKNN模型或者初始化运行环境进行推理等操作；如果运行在连接的板卡NPU上不需要。
-   初始化运行时环境，如果需要模型推理或性能评估，必须先调用init_runtime初始化运行时环境，要指定模型的运行平台（模拟器或者连接板卡的硬件NPU）。
-   初始化运行环境后，可以调用inference接口进行推理，使用eval_perf接口对模型性能进行评估，或者使用eval_memory接口获取模型在硬件平台上运行时的内存使用情况（模型必须运行在硬件平台上）。
-   最后调用release接口释放RKNN对象。

使用Toolkit-lite2，可以运行在PC上，通过模拟器运行模型，然后进行推理，或者模型转换等操作；也可以运行在连接的板卡NPU上， 将RKNN模型传到NPU设备上运行，再从NPU设备上获取推理结果、性能信息等等。 

![image-20251202192600839](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/mac-picture/image-20251202192600839.png)

>   详细使用例程请参考RKNN-Toolkit2工程中examples/functions目录下例程