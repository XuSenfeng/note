---
tags:
  - Agent开发
  - AI应用
---
RunnablePassthrough允许传递输入数据，可以保持不变或添加额外的键。通常与RunnableParallel一起使用，将数据分配给映射中的新键。

实际是对上一层的输出做处理, 作为参数传入RunnablePassthrough, 可以使用assign来给参数里面添加新的键值对

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

runnable = RunnableParallel(
    passed=RunnablePassthrough(),
    extra=RunnablePassthrough.assign(mult=lambda x: x["num"] * 3),
    modified=lambda x: x["num"] + 1,
)

runnable.invoke({"num": 1})
```

> `passed` 键使用 `RunnablePassthrough()` 调用，因此它只是传递了 `{'num': 1}`。
>
> 在第二行中，我们使用了带有将数值乘以3的lambda的 `RunnablePastshrough.assign`。在这种情况下，`extra` 被设置为 `{'num': 1, 'mult': 3}`，即原始值加上 `mult` 键。
>
> 最后，我们还使用lambda在映射中设置了第三个键 `modified`，将num加1，结果为 `modified` 键的值为 `2`。

```python
retrieval_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

retrieval_chain.invoke("where did harrison work?")
```

> 在输入参数里面加入一个context键

### 记忆总结

实际是把对话交给AI, 让他进行总结处理

```python
def summarize_messages(chain_input):
    stored_messages = history.messages
    if len(stored_messages) == 0:
        return False
    summarization_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="history"),
        ("user", "把上面的消息浓缩一下, 尽可能的包含多个细节, 尤其是用户的特征")
    ])

    summarization_chain = summarization_prompt | ChatOpenAI(model="gpt-3.5-turbo-1106")
    result = summarization_chain.invoke(input={"history": stored_messages})
    history.clear()
    history.add_message(result)
    return True


chain_with_summarization = (
    RunnablePassthrough.assign(messages_summarized=summarize_messages)
    | with_message_history
)

```

## 多模态输入

把输入按照一定的格式进行转换

### 图片输入

直接传递图片地址, 这个方式可能由于AI无法获取图片失败

```python
import base64 # 这是一个编码库, 用于编码和解码二进制数据
import httpx # 这是一个异步 HTTP 客户端库
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from Siliconflow import CustomLLM_Siliconflow


image_url = "https://th.bing.com/th/id/R.6b5df1bfe0e4778a44dba0753cd169c8?rik=QRQIMqvjWRCO5Q&riu=http%3a%2f%2fpic39.nipic.com%2f20140321%2f8857347_232251363165_2.jpg&ehk=7oAaMo6LCHJc%2bqpQ0IPvcH7v69jGRQhb2vDz%2fOd5720%3d&risl=&pid=ImgRaw&r=0"

# 方式1: 直接传入网址
model = ChatOpenAI(model="gpt-4o")
message = HumanMessage(
    content=[
        {
            "type": "text", "text": "描述这个图片里面的内容"
        },
        {
            "type": "image_url", "image_url": {"url": image_url}
        }
    ]
)
```

也可以把图片的数据直接传过去

```python
import base64 # 这是一个编码库, 用于编码和解码二进制数据
import httpx # 这是一个异步 HTTP 客户端库
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from Siliconflow import CustomLLM_Siliconflow


image_url = "https://th.bing.com/th/id/R.6b5df1bfe0e4778a44dba0753cd169c8?rik=QRQIMqvjWRCO5Q&riu=http%3a%2f%2fpic39.nipic.com%2f20140321%2f8857347_232251363165_2.jpg&ehk=7oAaMo6LCHJc%2bqpQ0IPvcH7v69jGRQhb2vDz%2fOd5720%3d&risl=&pid=ImgRaw&r=0"
image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8") # 将图片转换为 base64 编码的字符串

message_img = HumanMessage(
    content=[
        {
            "type": "text", "text": "描述这个图片里面的内容"
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
        }
    ]
)
```

```python
result = model.invoke([message_img])
print(result)
```

## 使用工具

tool有机部分组成

+ name: 名字, 必须唯一的描述
+ description: 工具的描述, LLM之后会使用这个作为上下文
+ args_schema: 可选, 但是建议提供更多的信息, Pydantic BaseModel的类型
+ return_direct: boolean类型, 代理相关的, True的时候代理停止, 直接将结果返回给用户

有几种方式可以定义

### 自定义工具

[定义自定义工具 | 🦜️🔗 Langchain](https://python.langchain.com.cn/docs/modules/agents/tools/how_to/custom_tools)

#### @tool装饰器

```python
from langchain_core.tools import tool
@tool
def weather_tool(weather: Literal["晴朗的", "多云的", "多雨的", "下雪的"]) -> None:
    """Discribe the weather"""
    # 这个工具接受一个天气参数, 并返回一个描述天气的字符串
    pass

"""
name='weather_tool' description='Discribe the weather' args_schema=<class 'langchain_core.utils.pydantic.weather_tool'> func=<function weather_tool at 0x7f57151d23e0>
"""
```

建立一个工具, 工具的描述是他的注释, LangChain 中的工具抽象将 Python 函数与定义函数**名称**、**描述**和**预期参数**的**架构**相关联。

> ```python
> print(weather_tool.name)
> print(weather_tool.description)
> print(weather_tool.args)
> """
> weather_tool
> Discribe the weather
> {'weather': {'enum': ['晴朗的', '多云的', '多雨的', '下雪的'], 'title': 'Weather', 'type': 'string'}}
> """
> ```

+ 使用

```python
from typing import Literal
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from Siliconflow import CustomLLM_Siliconflow
from langchain_core.tools import tool

@tool
def weather_tool(weather: Literal["晴朗的", "多云的", "多雨的", "下雪的"]) -> None:
    """Discribe the weather"""
    # 这个工具接受一个天气参数, 并返回一个描述天气的字符串
    pass

model = ChatOpenAI(model="gpt-4o")
model_with_tool = model.bind_tools([weather_tool]) # 绑定工具

image_url = "https://th.bing.com/th/id/R.6b5df1bfe0e4778a44dba0753cd169c8?rik=QRQIMqvjWRCO5Q&riu=http%3a%2f%2fpic39.nipic.com%2f20140321%2f8857347_232251363165_2.jpg&ehk=7oAaMo6LCHJc%2bqpQ0IPvcH7v69jGRQhb2vDz%2fOd5720%3d&risl=&pid=ImgRaw&r=0"
# 方式1: 直接传入网址
message = HumanMessage(
    content=[
        {
            "type": "text", "text": "描述这个图片里面的内容"
        },
        {
            "type": "image_url", "image_url": {"url": image_url}
        }
    ]
)
# 如果模型想使用工具, 他只可以输出工具支持的输入, 所以可以限定模型的输出
result = model.invoke([message])
print(result)
```

> + Create tools using the`@tool`decorator, which simplifies the process of tool creation, supporting the following:
>     使用 [@tool](https://python.langchain.com/api_reference/core/tools/langchain_core.tools.convert.tool.html) 修饰器创建工具，这简化了工具创建过程，并支持以下功能：
> + + Automatically infer the tool's **name**, **description** and **expected arguments**, while also supporting customization.
>         自动推断工具**的名称**、**描述**和**预期参数**，同时还支持自定义。
>     + Defining tools that return **artifacts** (e.g. images, dataframes, etc.)
>         定义返回**工件**的工具（例如图像、数据帧等）
>     + Hiding input arguments from the schema (and hence from the model) using **injected tool arguments**.
>         使用**注入的工具参数**从 Schema （以及模型中） 隐藏输入参数。

+ 异步

默认的使用是同步的, 也可以使用异步的, 之后的调用需要使用异步的方式

```python
@tool
async def weather_tool(weather: Literal["晴朗的", "多云的", "多雨的", "下雪的"]) -> None:
    """Discribe the weather"""
    # 这个工具接受一个天气参数, 并返回一个描述天气的字符串
    pass
```

+ 参数

使用pydantic指定参数

```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class CalculatorInput(BaseModel):
    a: int = Field(description="第一个数字")
    b: int = Field(description="第二个数字")

@tool("multiplication-tool", args_schema=CalculatorInput, return_direct=True)
def multiplication_tool(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

print(multiplication_tool.name)
print(multiplication_tool.description)
print(multiplication_tool.args)
print(multiplication_tool.invoke({"a": 2, "b": 3}))
"""
multiplication-tool
Multiply two numbers
{'a': {'description': '第一个数字', 'title': 'A', 'type': 'integer'}, 'b': {'description': '第二个数字', 'title': 'B', 'type': 'integer'}}
6
"""
```

#### StructuredTool.from_function

类似于tool但是允许更多配置以及同步异步的规范

```python
from langchain_core.tools import StructuredTool
import asyncio

def multiply(a: int, b:int)->int:
    """Multiply two numbers"""
    return a*b

async def amultiply(a: int, b:int)->int:
    """Multiply two numbers"""
    return a*b

async def main():
    calcuator = StructuredTool.from_function(func=multiply, coroutine=amultiply)
    print(calcuator.invoke({"a": 2, "b": 3}))
    print(await calcuator.ainvoke({"a": 2, "b": 3}))

asyncio.run(main())
"""
6
6
"""
```

#### 异常处理

工具的出现异常的使用的处理

```python
from langchain_core.tools import StructuredTool
from langchain_core.tools import ToolException
from langchain.tools import Tool
def get_weather(city: str) -> int:
    """Get the weather of a city"""
    return ToolException(f"Can't get the weather of {city}")

get_weather_tool = Tool.from_function(
    func=get_weather,
    # handle_tool_error=True,
    name="get_weather",
    description="Get the weather of a city",
)

response = get_weather_tool.invoke("Beijing")
print(response)
```

> You can set `handle_tool_error` to `True`, set it a unified string value, or set it as a function. If it's set as a function, the function should take a `ToolException` as a parameter and return a `str` value.
>
> 会使用这个异常的输出作为返回值, 手册里面说这个受到handle_tool_error控制, 但是这里实际测试后发现handle_tool_error参数没有影响
>
> 使用StructuredTool的时候可以传递更少的参数

#### 使用BaseTool建立一个子类

BaseTool 会自动从 _run 方法的签名推断架构。

### 内置工具

工具实际是代理, 链或者聊天模型和世界交互的方式, 一般有以下的部分

1. 工具的名字
2. 工具的功能描述
3. 工具的输入的JSON格式
4. 要调用的函数
5. 工具的结果是不是直接返给用户, 把名称描述上下文发送给大模型, 让模型进行工具的调用, 一般经过微调的模型效果比较好, 没有微调的模型可能失败

#### 维基百科工具

[WikipediaAPIWrapper — 🦜🔗 LangChain documentation](https://python.langchain.com/api_reference/community/utilities/langchain_community.utilities.wikipedia.WikipediaAPIWrapper.html)

```python
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
tool = WikipediaQueryRun(api_wrapper=api_wrapper)
print(tool.invoke({"query": "Langchain"}))

print(tool.name)
print(tool.description)
print(tool.args)
"""
Wiki-tools
A tool to query Wikipedia
{'query': {'description': '查询关键词, 长度不超过10个字', 'title': 'Query', 'type': 'string'}}
"""
```

可以做一下参数的封装

```python
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from pydantic import BaseModel, Field

class WikiInputs(BaseModel):
    """维基百科工具输入"""
    query: str = Field(description="查询关键词, 长度不超过10个字")


api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
tool = WikipediaQueryRun(
    api_wrapper=api_wrapper,
    name="Wiki-tools",
    description="A tool to query Wikipedia",
    args_schema=WikiInputs,
    return_direct=True
    )

print(tool.run("Langchain"))

print(tool.name)
print(tool.description)
print(tool.args)
"""
Wiki-tools
A tool to query Wikipedia
{'query': {'description': '查询关键词, 长度不超过10个字', 'title': 'Query', 'type': 'string'}}
"""
```

#### SQL处理

[LangChain中文网](https://www.langchain.com.cn/docs/integrations/tools/sql_database/)

```python
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit

import sqlite3
import requests
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

# 这段代码的作用是从一个远程的SQL脚本中创建一个在内存中的SQLite数据库，并返回一个数据库引擎对象。
def get_engine_for_chinook_db():
    """从SQL脚本创建内存数据库并返回引擎对象。"""
    url = "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"
    response = requests.get(url)
    sql_script = response.text

    connection = sqlite3.connect(":memory:", check_same_thread=False)
    connection.executescript(sql_script)
    return create_engine(
        "sqlite://",
        creator=lambda: connection,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )

engine = get_engine_for_chinook_db()

db = SQLDatabase(engine)
# db = SQLDatabase.from_uri("sqlite:///langchain.db")

# 建立一个SQL工具箱对象, 用于执行SQL查询, 可以绑定LLM模型
toolkit = SQLDatabaseToolkit(db=db, llm=ChatOpenAI(temperature=0))
# print(toolkit.get_tools())

agent_exector = create_sql_agent(
    llm=ChatOpenAI(temperature=0),
    toolkit=toolkit,
    verbose=False,
    agent_type=AgentType.OPENAI_FUNCTIONS
)

result = agent_exector.invoke("Create a full_llm_cache table")
print(result)
```



## 指定格式输出

如果输出的格式是一个json, xml或者yaml, 但是大模型的输出可以能是一个混乱的格式, 所以使用的模型必须足够大

可以使用JsonOutputParser进行提示并且解析大模型的输出, 也可以使用PydanticOutputParser, Json的输出是内置的, 支持流式返回

> Pydantic示例, [Python笔记：Pydantic库简介-CSDN博客](https://blog.csdn.net/codename_cys/article/details/107675748)
>
> ```python
> from pydantic import BaseModel
> 
> class Person(BaseModel):
>     name: str
> 
> p = Person(name="Tom")
> print(p.json()) # {"name": "Tom"}
> ```

```python
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field



model = ChatOpenAI(model="gpt-4o")
# 描述输出的具体格式
class Joke(BaseModel):
    setup: str = Field(description="设置笑话的问题")
    punchline: str = Field(description="解决笑话的答案")

joke_query = "告诉我一个笑话"

parser = JsonOutputParser(pydantic_object=Joke)
prompt = PromptTemplate(
    template="回答用户的问题, \n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = prompt |  model | parser
message = chain.invoke({"query": joke_query})
print(message)
"""
{'setup': '为什么程序员不饿？', 'punchline': '因为他们有很多的缓存。'}
"""
```

可以看一下输入的模板, 如果不使用这个模版, 实际的输出会有随机性

````python
print(parser.get_format_instructions())
"""
The output should be formatted as a JSON instance that conforms to the JSON schema below.
As an example, for the schema 
{
	"properties": {
		"foo": {
			"title": "Foo", 
			"description": "a list of strings", 
			"type": "array", 
			"items": {"type": "string"}
		}
	}, 
	"required": ["foo"]
}
the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.

Here is the output schema:
```
{
	"properties": {
		"setup": {
			"title": "Setup",
             "description": "设置笑话的问题", 
             "type": "string"
         }, 
         "punchline": {
             "title": "Punchline", 
             "description": 
             "解决笑话的答案", 
             "type": "string"
          }
     }, 
     "required": ["setup", "punchline"]
}
```
"""
````

可以使用流式输出

```python
for s in chain.stream({"query": joke_query}):
    print(s)
"""
{}
{'setup': ''}
{'setup': '为什么'}
{'setup': '为什么足球'}
{'setup': '为什么足球比赛'}
{'setup': '为什么足球比赛总'}
{'setup': '为什么足球比赛总是在'}
{'setup': '为什么足球比赛总是在灯'}
{'setup': '为什么足球比赛总是在灯光'}
{'setup': '为什么足球比赛总是在灯光下'}
{'setup': '为什么足球比赛总是在灯光下？'}
{'setup': '为什么足球比赛总是在灯光下？', 'punchline': ''}
{'setup': '为什么足球比赛总是在灯光下？', 'punchline': '因为'}
{'setup': '为什么足球比赛总是在灯光下？', 'punchline': '因为观'}
{'setup': '为什么足球比赛总是在灯光下？', 'punchline': '因为观众'}
{'setup': '为什么足球比赛总是在灯光下？', 'punchline': '因为观众们'}
{'setup': '为什么足球比赛总是在灯光下？', 'punchline': '因为观众们需要'}
{'setup': '为什么足球比赛总是在灯光下？', 'punchline': '因为观众们需要看到'}
{'setup': '为什么足球比赛总是在灯光下？', 'punchline': '因为观众们需要看到进'}
{'setup': '为什么足球比赛总是在灯光下？', 'punchline': '因为观众们需要看到进球'}
{'setup': '为什么足球比赛总是在灯光下？', 'punchline': '因为观众们需要看到进球！'}
"""
```

### YAML输出

这个格式的实际使用和JSON的使用是一样的

### XML格式

可以使用XMLOutputParser进行输出, 使用`tags=["", ""]`可以指定实际输出的标签, 之后的输出会按照这个的层级进行输出, 如果使用的是流式输出, 实际的结果还是json

```python
from langchain_core.output_parsers import XMLOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

model = ChatOpenAI(model="gpt-3.5-turbo-1106")

joke_query = "生成周星驰的电影作品, 安装时间顺序进行排序"

parser = XMLOutputParser(tags=["movie", "actor", "film", "genre","time"])
prompt = PromptTemplate(
    template="回答用户的问题, \n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = prompt |  model | parser
message = chain.invoke({"query": joke_query})
print(message)
```

## 自定义模型

[Custom LLM | 🦜️🔗 LangChain](https://python.langchain.com/v0.1/docs/modules/model_io/llms/custom_llm/)

需要实现两个函数, 

[使用LangChain自定义大模型 | 完美调用第三方 API | 如OneAPI/硅基流动-腾讯云开发者社区-腾讯云](https://cloud.tencent.com/developer/article/2467458)