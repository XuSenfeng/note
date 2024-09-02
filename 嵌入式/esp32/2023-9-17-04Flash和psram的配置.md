---
layout: post
title: "Flash和psram的配置" 
date:   2023-9-16 15:39:08 +0800
tags: esp32
---

# Flash和psram的配置

![image-20230917101913064](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202408242257402.png)

![image-20230917102856323](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202408242257404.png)

![image-20230917103515582](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202408242257405.png)

> 使用扩展的SRAM的话这几个引脚就会被占用

```c
I (24) boot: ESP-IDF v5.0.4 2nd stage bootloader
I (25) boot: compile time 10:37:52
I (25) boot: Multicore bootloader
I (27) boot: chip revision: v0.1
I (31) qio_mode: Enabling default flash chip QIO
//这里显示的是SPIFLASH
I (36) boot.esp32s3: Boot SPI Speed : 80MHz
I (41) boot.esp32s3: SPI Mode       : QIO
I (45) boot.esp32s3: SPI Flash Size : 16MB
I (50) boot: Enabling RNG early entropy source...
I (56) boot: Partition Table:
I (59) boot: ## Label            Usage          Type ST Offset   Length
I (66) boot:  0 nvs              WiFi data        01 02 00009000 00006000
I (74) boot:  1 phy_init         RF data          01 01 0000f000 00001000
I (81) boot:  2 factory          factory app      00 00 00010000 00100000
I (89) boot: End of partition table
I (93) esp_image: segment 0: paddr=00010020 vaddr=3c020020 size=09aach ( 39596) map
I (108) esp_image: segment 1: paddr=00019ad4 vaddr=3fc92b00 size=02b34h ( 11060) load
I (112) esp_image: segment 2: paddr=0001c610 vaddr=40374000 size=03a08h ( 14856) load
I (121) esp_image: segment 3: paddr=00020020 vaddr=42000020 size=1a098h (106648) map
I (143) esp_image: segment 4: paddr=0003a0c0 vaddr=40377a08 size=0b018h ( 45080) load
I (158) boot: Loaded app from partition at offset 0x10000
I (158) boot: Disabling RNG early entropy source...
I (169) cpu_start: Multicore app
I (170) octal_psram: vendor id    : 0x0d (AP)
I (170) octal_psram: dev id       : 0x02 (generation 3)
I (173) octal_psram: density      : 0x03 (64 Mbit)
I (179) octal_psram: good-die     : 0x01 (Pass)
I (184) octal_psram: Latency      : 0x01 (Fixed)
I (189) octal_psram: VCC          : 0x01 (3V)
I (194) octal_psram: SRF          : 0x01 (Fast Refresh)
I (200) octal_psram: BurstType    : 0x01 (Hybrid Wrap)
I (206) octal_psram: BurstLen     : 0x01 (32 Byte)
I (211) octal_psram: Readlatency  : 0x02 (10 cycles@Fixed)
I (218) octal_psram: DriveStrength: 0x00 (1/1)
I (223) MSPI Timing: PSRAM timing tuning index: 5
I (228) esp_psram: Found 8MB PSRAM device
I (233) esp_psram: Speed: 80MHz
I (237) cpu_start: Pro cpu up.
I (240) cpu_start: Starting app cpu, entry point is 0x40375374
0x40375374: call_start_cpu1 at E:/alearn/ESP-IDE/esp-idf_v5.1.2/esp-idf/components/esp_system/port/cpu_start.c:143

I (0) cpu_start: App cpu up.
//测试SPRAM, 测试成功
I (699) esp_psram: SPI SRAM memory test OK
I (708) cpu_start: Pro cpu start user code
I (708) cpu_start: cpu freq: 160000000 Hz
I (708) cpu_start: Application information:
I (711) cpu_start: Project name:     hello_world
I (716) cpu_start: App version:      1
I (721) cpu_start: Compile time:     Sep 17 2023 10:37:22
I (727) cpu_start: ELF file SHA256:  c57c0dd431ce40e9...
I (733) cpu_start: ESP-IDF:          v5.0.4
I (738) cpu_start: Min chip rev:     v0.0
I (742) cpu_start: Max chip rev:     v0.99 
I (747) cpu_start: Chip rev:         v0.1
I (752) heap_init: Initializing. RAM available for dynamic allocation:
I (759) heap_init: At 3FC960C8 len 00053648 (333 KiB): DRAM
I (765) heap_init: At 3FCE9710 len 00005724 (21 KiB): STACK/DRAM
I (772) heap_init: At 3FCF0000 len 00008000 (32 KiB): DRAM
I (778) heap_init: At 600FE010 len 00001FD8 (7 KiB): RTCRAM
I (785) esp_psram: Adding pool of 8192K of PSRAM memory to heap allocator
I (793) spi_flash: detected chip: winbond
I (797) spi_flash: flash io: qio
I (801) app_start: Starting scheduler on CPU0
I (806) app_start: Starting scheduler on CPU1
I (806) main_task: Started on CPU0
I (816) esp_psram: Reserving pool of 32K of internal memory for DMA/internal allocations
I (826) main_task: Calling app_main()
I (826) main: Hello world
```

```c
[13/907] Generating ../../partition_table/partition-table.bin
Partition table binary generated. Contents:
*******************************************************************************
# ESP-IDF Partition Table
# Name, Type, SubType, Offset, Size, Flags
nvs,data,nvs,0x9000,24K,
phy_init,data,phy,0xf000,4K,
factory,app,factory,0x10000,1M,
*******************************************************************************
```

> 发现在这个时候并不是所有的Flash都被使用了

## 添加分区表

![image-20230917105409604](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202408242257406.png)

![image-20230917105358856](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202408242257407.png)

> 把这个文件添加到文件夹里面

![image-20230917105947422](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202408242257408.png)

```bash
*******************************************************************************
# ESP-IDF Partition Table
# Name, Type, SubType, Offset, Size, Flags
nvs,data,nvs,0x9000,16K,
otadata,data,ota,0xd000,8K,
phy_init,data,phy,0xf000,4K,
factory,app,factory,0x10000,1M,
ota_0,app,ota_0,0x110000,1M,
ota_1,app,ota_1,0x210000,1M,
*******************************************************************************
```

> 分区表增加了

![image-20230917110203861](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202408242257409.png)

> 之后可以在上面的表里面进行修改分区表, 也可以添加自己的分区
>
> 











