# Gomodules

用于管理go语言的包, 淘汰之前的GOPATH

## 演进

### GOPATH

可以使用`go env`查看go使用的环境变量

```bash
GOPATH='/Users/jiao/go'
```

记录当前所有的go的项目的路径

+   bin: 编译好的可执行文件
+   pkg: 一些依赖文件, 加快编译
+   src: 源码

弊端: 

1.   没有版本管理概念, 在使用`go get xxx`的时候没有管理版本
2.   没有办法同步一致第三方版本号
3.   没有办法指定项目的第三方版本号

### gomodules

```bash
(base) jiao@jiaodeMacBook-Air go % go mod help
Go mod provides access to operations on modules.

Note that support for modules is built into all the go commands,
not just 'go mod'. For example, day-to-day adding, removing, upgrading,
and downgrading of dependencies should be done using 'go get'.
See 'go help modules' for an overview of module functionality.

Usage:

        go mod <command> [arguments]

The commands are:

        download    download modules to local cache
        edit        edit go.mod from tools or scripts
        graph       print module requirement graph
        init        initialize new module in current directory
        tidy        add missing and remove unused modules
        vendor      make vendored copy of dependencies
        verify      verify dependencies have expected content
        why         explain why packages or modules are needed

Use "go help mod <command>" for more information about a command.
```

## 常用指令

+   `go mod init`: 构建一个`go.mod`文件
+   `go mod download`: 下载`go.mod`文件里面的所有依赖
+   `go mod tidy`: 整理现在的所有依赖
+   `go mod graph`: 查看当前所有依赖的结构
+   `go mod edit`: 编辑`go.mod`文件
+   `go mod verify`校验模块是否被修改
+   `go mod why`: 查看为什么依赖某个模块

### 依赖变量

```bash
GO111MODULE=''
GOPROXY='https://proxy.golang.org,direct'
GOSUMDB='sum.golang.org'
GONOPROXY=''
GONOSUMDB=''
GOPRIVATE=''
```

+   `GO111MODULE`: 是否启用GoModule, 可以使用`on, off, auto`
+   `GOPROXY`: 设置代理, 后续使用的时候可以通过这个自动拉取, `direct`表示, 在import的时候设置的是一个github地址的时候, 优先从这个地址获取, 没有的时候才会从github进行下载代码
+   `GOSUMDB`: 校验拉取到的版本是一个没有更改的版本, 默认会使用GOPROXY校验
+   `GONOPROXY/GONOSUMDB/GOPRIVAT`: 表示当前项目依赖私有模块, 使用的github仓库是私有的, 一般设置GOPRIVAT为私有仓库即可

>   修改变量`go env -w GO111MODULE=on`

### 初始化项目

`go mod init github.com/XuSenfeng/go_test` 初始化项目设置其他人引用的时候怎么写

下载模块`go get 模块地址`

替换版本`go mod -replace=old=new`进行替换模型