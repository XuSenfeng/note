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
```
https://github.com/protocolbuffers/protobuf/releases/latest
```

从这个网页下载一个win64版本的进行解压即可进行使用

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