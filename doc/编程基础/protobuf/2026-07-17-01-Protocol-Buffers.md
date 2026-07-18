---
tags:
  - 编程语言
  - 编程基础
---
# Protocol Buffers

Google Protocol Buffer( 简称 Protobuf) 是 Google 公司内部的混合语言数据标准

在两个不同的程序进行通信的时候, 需要严格的规定实际使用的数据的格式, 如果使用 Protobuf，那么这些细节就可以不需要应用程序来考虑了

使用 Protobuf，Writer 的工作很简单，需要处理的结构化数据由 .proto 文件描述，经过编译过程后，该数据化结构对应了一个的类, 之后使用这个类进行处理获取到的数据即可

实际的传输数据是一个二进制的数据, 这个数据有性能高, 占用资源小的优势

Protocol buffer 消息本身不描述其数据，但它们具有一个完全反射的模式，您可以使用它来实现自描述。也就是说，如果没有对其相应 `.proto` 文件的访问权限，您无法完全解释它。
## 环境安装
### 第 1 步 — 安装 MSYS2

从 [https://www.msys2.org](https://www.msys2.org/) 下载 `msys2-x86_64-*.exe` 并安装(默认装到 `C:\msys64`)。如果有 `winget`,在 PowerShell 里一行搞定:

```powershell
winget install MSYS2.MSYS2
```

**MSYS2** 是 Windows 上的一套软件包管理 + 开发环境,专门用来把那些原本给 Linux/Unix 用的开发工具搬到 Windows 上用。

### 第 2 步 — 安装 protobuf 和工具链

打开 "MSYS2 UCRT64" 快捷方式(从开始菜单找,不要用普通的 MSYS2 终端),运行:

```bash
pacman -Syu
# 如果提示要关闭窗口,重新打开 UCRT64 再运行一次
pacman -S --needed mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-protobuf mingw-w64-ucrt-x86_64-pkgconf
```

选 **UCRT64** 环境是特意的 —— 它和你现在的 winlibs GCC(同样是 UCRT 构建)匹配。

### 第 3 步 — 重新生成 `.pb.*` 文件

MSYS2 装的 protobuf 版本不一定正好是 35.1,而生成的代码必须和已安装的库版本一致。在 **UCRT64** 终端里用**它自带的 protoc** 重新生成:

```bash
cd /d/App/protobuf/bin
protoc --cpp_out=. addressbook.proto
```

这会覆盖 `addressbook.pb.cc` / `addressbook.pb.h`。(不要用你自己的 `protoc.exe`,用刚装的这个。)

### 第 4 步 — 编译

还是在 **UCRT64** 终端:

```bash
cd /d/App/protobuf/bin

# Writer
g++ Writer.cc addressbook.pb.cc -o Writer.exe $(pkg-config --cflags --libs protobuf)

# Reader
g++ Reader.cc addressbook.pb.cc -o Reader.exe $(pkg-config --cflags --libs protobuf)
```

`pkg-config` 会自动补全所有头文件路径、`-lprotobuf` 以及 abseil 库。然后运行:

```bash
./Writer.exe   # 生成 ./log
./Reader.exe   # 读取 ./log,打印 101 和 "hello"
```

## 测试示例

### 生成cpp接口文件

```proto
package lm;
message helloworld
{
   required int32     id = 1;  // ID
   required string    str = 2;  // str
   optional int32     opt = 3;  //optional field
}
```

文件名为`addressbook.proto`

```bash
protoc --cpp_out=. .\addressbook.proto
```

之后可以输出两个cpp的文件, `addressbook.pb.h`， `addressbook.pb.cc`

在生成的头文件中，定义了一个 C++ 类 helloworld, 类 lm::helloworld 提供相应的方法来把一个复杂的数据变成一个字节序列，我们可以将这个字节序列写入磁盘

读取这个数据的程序来说，也只需要使用类 lm::helloworld 的相应反序列化方法来将这个字节序列重新转换会结构化数据

### 读写程序

```cpp
#include "lm.helloworld.pb.h"

int main(void)
{

	lm::helloworld msg1;
	msg1.set_id(101);
	msg1.set_str("hello");
	 
	// Write the new address book back to disk.
	fstream output("./log", ios::out | ios::trunc | ios::binary);
		 
	if (!msg1.SerializeToOstream(&output)) {
	  cerr << "Failed to write msg." << endl;
	  return -1;
	}        
	return 0;
}
```


```cpp
#include "lm.helloworld.pb.h"

void ListMsg(const lm::helloworld & msg) {
	cout << msg.id() << endl;
	cout << msg.str() << endl;
}

int main(int argc, char* argv[]) {

	lm::helloworld msg1;
	
	{
		fstream input("./log", ios::in | ios::binary);
		if (!msg1.ParseFromIstream(&input)) {
			cerr << "Failed to parse address book." << endl;
			return -1;
		}
	}
	
	ListMsg(msg1);
}
```