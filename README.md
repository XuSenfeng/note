
# XvSenfeng 个人笔记

个人笔记记录, 可以使用teedoc进行文档的自动生成

## 使用teedoc

visit: [teedoc.neucrack.com](https://teedoc.neucrack.com/) or [teedoc.github.io](https://teedoc.github.io)


## build locally

* Install Python3 first 下载python3, Windows自行查找教程

```
sudo apt install python3 python3-pip
```

* Install teedoc 安装

```
pip3 install teedoc
```

* Get site source files 获取文档

```
git clone https://github.com/XuSenfeng/note.git note
```


* Install plugins 安装插件

```
cd note
teedoc install
```

* build and serve 构建

```
teedoc build
teedoc serve
```

之后可以在 [http://127.0.0.1:2333](http://127.0.0.1:2333) 目录查看

## 脚本

Win使用my.bat, Linux使用my.sh

+ git提交

```bash
my.sh git "commit的内容"
```
+ update刷新目录, 用于更新teedoc的配置脚本

```bash
my.sh update
```

+ server启动teedoc的server

```bash
my.sh server
```





