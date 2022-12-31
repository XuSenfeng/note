---
layout: post
title: "QPhshButton" 
date:   2022-9-14 15:39:08 +0800
tags: qt
---

# QPhshButton

```c++
#include "myvidget.h"
#include <QPushButton>
myVidget::myVidget(QWidget *parent)
    : QMainWindow(parent)
{
    //创建按键
    QPushButton *btn = new QPushButton;
    btn->show();    //以顶层的方式调用控件
    btn->setParent(this);
    btn->setText("第一个按钮");

    //方法2
    QPushButton *btn2 = new QPushButton("第二个按钮", this);
    //移动
    btn2->move(100, 100);
    //设置窗口大小
    //resize(1024, 480);

    //设置固定的大小
    setFixedSize(1024,480);

    //设置窗口标题
    setWindowTitle("第一个窗口");
}
```



















