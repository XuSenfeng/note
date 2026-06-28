---
tags:
  - JSON
---
# nlohmann处理json

用于解析 JSON 的开源 C++ 库

虽然提供了 CMakeLists.txt，但不需要 CMake编译。只要将 [https://github.com/nlohmann/json/tree/develop/include](https://github.com/nlohmann/json/tree/develop/include) 下的 nlohmann 目录拷贝到新建工程的 include 目录下

```cpp
#include "nlohmann/json.hpp" 
using json = nlohmann::json;
```

## 使用

### 构建

```json
{ 
	"pi": 3.141, 
	"happy": true, 
	"name": "Niels", 
	"nothing": null, 
	"answer": {
		 "everything": 42 
	}, 
	"list": [1, 0, 2], 
	"object": { 
		"currency": "USD", 
		"value": 42.99 
	} 
}
```

想要构建这样一个json字符串可以使用如下的代码

```cpp
json j; // 首先创建一个空的json对象 
j["pi"] = 3.141; 
j["happy"] = true; 
j["name"] = "Niels"; 
j["nothing"] = nullptr; 
j["answer"]["everything"] = 42; // 初始化answer对象 
j["list"] = { 1, 0, 2 }; // 使用列表初始化的方法对"list"数组初始化 
j["object"] = { {"currency", "USD"}, {"value", 42.99} }; // 初始化object对象
```

### 获取以及打印

```cpp
float pi = j.at("pi"); 
std::string name = j.at("name"); 
int everything = j.at("answer").at("everything"); 
std::cout << pi << std::endl; // 输出: 3.141 
std::cout << name << std::endl; // 输出: Niels 
std::cout << everything << std::endl; // 输出: 42 
// 打印"list"数组 
for(int i=0; i<3; i++) 
	std::cout << j.at("list").at(i) << std::endl; 
// 打印"object"对象中的元素 
std::cout << j.at("object").at("currency") << std::endl; // 输出: USD 
std::cout << j.at("object").at("value") << std::endl; // 输出: 42.99
```

### 记录在文件里面

```cpp
std::ofstream file("pretty.json"); 
file << j << std::endl;
```

### 序列化以及反序列化
#### string

```cpp
json j = "{\"happy\":true,\"pi\":3.141}"_json; 
auto j2 = R"({"happy":true,"pi":3.141})"_json; 
// 或者 
std::string s = "{\"happy\":true,\"pi\":3.141}"; 
auto j = json::parse(s.toStdString().c_str()); 
std::cout << j.at("pi") << std::endl; // 输出：3.141
```
序列化
```cpp
std::string s = j.dump(); // 输出：{\"happy\":true,\"pi\":3.141}
```

#### stream

使用标准输入输出`std::cout`和`std::cin`
```cpp
json j; 
std::cin >> j; // 从标准输入中反序列化json对象 
std::cout << j; // 将json对象序列化到标准输出中
```

将 json 对象序列化到本地文件，或者从存储在本地的文件中反序列化出 json 对象，是非常常用的一个操作。nlohmann 对于这个操作也很简单

```cpp
// 读取一个json文件，nlohmann会自动解析其中数据 
std::ifstream i("file.json"); 
json j; 
i >> j; 

// 以易于查看的方式将json对象写入到本地文件 
std::ofstream o("pretty.json"); 
o << std::setw(4) << j << std::endl;
```
### 复杂数据类型转换
使用类对象的时候, 和json之间的转换比较复杂


