# LangChain

## 架构

+ LangSmith: 监控
+ LangServe: 服务器处理
+ Templates: 模板
+ LangChain: 智能调用, Agents开发以及检索策略
+ LangChainCommunity: 社区, 支持各种大模型, 提示词等的处理, 输入内容的格式化, 工具调用的支持
+ LangChainCore: LCEL表达式语言的执行

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/faaf313af63c5029c7983192c1f43bd2.png)

主要的组成

+ LangChain库, 有Python和java, 里面有各种组件的接口以及运行的基础
+ LangChain模板: 提供的AI模板
+ LangServer: FastAPI把LangChain的链(Chain)发布为REST API
+ LangSmith: 开发平台云服务

库

+ langchain-core: 基础的抽象以及LangChain的表达语言
+ langchain-community: 第三发集成
+ langchain: 链以及代理和agent检索策略

![image-20250218132339058](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/picture/202502181323177.png)

> 输入和模板结合以后输入到LLM里面, 进行处理获得输出, 按照客户需求的格式进行组装
>
> LLM: 问答模型, 输入一个文本, 返回一个文本
>
> Chat Model: 对话模型, 接收一组对话, 返回对话消息, 和聊天类似

核心概念

+ LLMs

封装的基础模型, 接受一个文本的输入, 返回一个结果

+ ChatModels

聊天模型, 和LLMs不同, 味蕾对话设计, 可以处理长下文

+ 消息Message

聊天模型的消息内容, 有多种HumanMessage, AIMessage, SystemMessage, FunctionMessage, ToolMessage等

+ 提示prompts

格式化提示词

+ 输出解释器

llm返回以后使用专门的解释器进行格式化为json之类的

+ Retrievers

私有数据导入大模型, 提高问题的质量, LangChain封装了检索的框架Retrieveers, 可以加入文档, 切割, 存储搜索

+ 向量存储Vector stores

私有数据的语义相似检索, 支持多种向量数据库

+ Agent

智能体, 以LLM为决策引擎, 根据用户的输入, 自动调用外部系统和设备完成任务



## 简单示例

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os
os.environ["OPENAI_BASE_URL"] = "https://api.chatanywhere.tech"
llm = ChatOpenAI()

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个图书管理员。"),
    ("user", "{input}"),
])
# 通过LangChain的链式调用生成一个chain对象
chain  = prompt | llm
# 打印生成的对话
result = chain.invoke({"input": "你好，给我推荐一个故事书。"})
print(result)

"""
content='你好，我推荐给你一本经典的故事书《小王子》。这本书由法国作家安托万·德·圣-埃克絮住创作，讲述了一个小王子在宇宙中的冒险故事，通过他和各种奇特角色的相遇，揭示了人生、友情、爱情、责任等深刻的主题，是一部富有哲理的作品。希望你会喜欢这本书！' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 149, 'prompt_tokens': 32, 'total_tokens': 181, 'completion_tokens_details': None, 'prompt_tokens_details': None}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': 'fp_0165350fbb', 'finish_reason': 'stop', 'logprobs': None} id='run-b9a821df-af44-429d-906c-8f5ddb6a46e7-0' usage_metadata={'input_tokens': 32, 'output_tokens': 149, 'total_tokens': 181, 'input_token_details': {}, 'output_token_details': {}}
"""
# 使用下面的方式可以改变输出的格式
from langchain_core.output_parsers import StrOutputParser
output_parser = StrOutputParser()
chain  = prompt | llm | output_parser
"""
你好！当然可以。我推荐你阅读《小王子》这本书。这是一本经典的童话故事，讲述了一位王子在不同星球上的冒险故事，以及他与一只狐狸和一朵玫瑰的感人情感纠葛。这本书深刻地探讨了友谊、爱情和人生意义的主题，是一部感人至深的文学作品。希望你会喜欢！
"""
```

## 提示词工程

和AI进行沟通的时候需要使用提示词和AI进行对话, 同时AI可能出现幻觉, 开发的时候不可以直接进行硬编, 不利于提示词管理, 通过提示词的模板进行维护

实际用户的输入只是提示词里面一个参数

+ 发给大模型的指令
+ 一组问答示例
+ 发给模型的问题

### 构成

+ PromptValue: 表示模型输入的类。

+ Prompt Templates: 负责构建 PromptValue 的类。

+ 示例选择器 Example Selectors: 在提示中包含示例通常是有用的。这些示例可以硬编码，但如果它们是动态选择的，则通常更有用。

+ 输出解析器 Output Parsers: 语言模型（和聊天模型）输出文本。但是许多时候，您可能想获得比仅文本更有结构化的信息。这就是输出解析器发挥作用的地方。输出解析器负责（1）指示模型如何格式化输出，（2）将输出解析为所需格式（包括必要时进行重试）。

有两个方法一定有实现

`get_format_instructions() -> str`：一个返回包含语言模型输出格式化指令的字符串的方法。 parse(str) -> Any：一个接受字符串（假设为语言模型的响应）并将其解析为某种结构的方法。 还有一个可选的方法：

`parse_with_prompt(str) -> Any`：一个接受字符串（假设为语言模型的响应）和提示（假设为生成此响应的提示）的方法，并将其解析为某种结构。在输出解析器希望以某种方式重试或修复输出，并且需要来自提示的信息时，提供提示非常有用。

### 字符串格式的模板

```python
from langchain_core.prompts import ChatPromptTemplate
# 这种的会生成消息, 可以有上下文
chat_template = ChatPromptTemplate.from_messages([
    ("system", "你是一个猫娘，你的名字是{name}。"),
    ("system", "你正在一家宠物店里等待被领养。"),
    ("human", "你好，小猫。"),
    ("ai", "你好, 喵~~"),
    ("human", "{input}")
])

message = chat_template.format_messages(name = "小花", input = "你的名字是什么?")
print(message)
"""
[
SystemMessage(content='你是一个猫娘，你的名字是小花。', additional_kwargs={}, response_metadata={}), 
SystemMessage(content='你正在一家宠物店里等待被领养。', additional_kwargs={}, response_metadata={}), 
HumanMessage(content='你好，小猫。', additional_kwargs={}, response_metadata={}), AIMessage(content='你好, 喵~~', additional_kwargs={}, response_metadata={}), HumanMessage(content='你的名字是什么?', additional_kwargs={}, response_metadata={})
]
"""
# 生成一个字符串
from langchain_core.prompts import PromptTemplate
prompt_template = PromptTemplate.from_template(
    "你的名字是{name}, 你正在一家宠物店里等待被领养。你好，小猫。{input}"
)
message = prompt_template.format(name = "小花", input = "你的名字是什么?")
print(message)
"""
你的名字是小花, 你正在一家宠物店里等待被领养。你好，小猫。你的名字是什么?
"""
from langchain_core.messages import SystemMessage
from langchain.prompts import HumanMessagePromptTemplate

chat_template = ChatPromptTemplate.from_messages([
    SystemMessage(
        content = ("你是一个乐于助人的助手。")
    ),
    HumanMessagePromptTemplate.from_template("{text}")
])

message = chat_template.format_messages(text = "你好，我想了解一下你的产品。")
print(message)
"""
[
SystemMessage(content='你是一个乐于助人的助手。', additional_kwargs={}, response_metadata={}), 
HumanMessage(content='你好，我想了解一下你的产品。', additional_kwargs={}, response_metadata={})
]
"""

```

消息提示词里面有三种角色

+ 助手Assistant: AI的回答
+ 人类User: 你发送的消息
+ 系统System: 进行AI身份的描述

### MessagesPlaceholder

特定的位置添加消息列表, 上面处理的是两条消息, 每一个消息都是一个字符串, 如果输入的是一个消息列表可以使用这个

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "你是一个助手"),
    MessagesPlaceholder("msgs")
])

prompt_template.invoke({"msgs": [HumanMessage(content="你好")]})
"""
ChatPromptValue(
messages=[
SystemMessage(content='你是一个助手', additional_kwargs={}, response_metadata={}), 
HumanMessage(content='你好', additional_kwargs={}, response_metadata={})
]
)
"""
```

> 可以看为一个占位符, 这里可以穿进去一系列的Message

### Few-shot prompt template

追加提示词示例

帮助交互样本更好的了解用户的意图, 从而更好的回答问题以及处理任务, 使用少量的示例指导模型输入, 可以使用示例集进行

```python
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate

examples = [
    {
        "question": "谁的寿命更长, 毛泽东还是邓小平?",
        "answer": 
                    """
                    这里要回答的是谁的寿命更长, 毛泽东还是邓小平? 毛泽东的寿命是1893年12月26日出生, 1976年9月9日去世, 享年82岁. 
                    邓小平的寿命是1904年8月22日出生, 1997年2月19日去世, 享年92岁. 
                    因此, 邓小平的寿命更长.
                    所以最终的答案是邓小平的寿命更长.        
                    """
    },
    {
        "question": "哪吒之魔童降世的导演是哪国人?",
        "answer": 
                    """
                    这里要回答的是跟进问题吗? 是的
                    跟进: 哪吒之魔童降世的导演是谁?
                    哪吒之魔童降世的导演是饺子.
                    跟进: 饺子是哪国人?
                    饺子是中国人.
                    所以最终的答案是饺子是中国人.
                    """
    }
]
# 使用这个模板来生成问题和答案
example_prompt = PromptTemplate(input_variables=["question", "answer"], template="问题: {question}\\n{answer}")
"""
StringPromptValue
"""
prompt = FewShotPromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
    suffix="问题:{input}", # 后缀
    input_variables=["input"]
)
print(prompt.format(input="史凯歌爸爸是谁?"))
"""
问题: 谁的寿命更长, 毛泽东还是邓小平?\n
                    这里要回答的是谁的寿命更长, 毛泽东还是邓小平? 毛泽东的寿命是1893年12月26日出生, 1976年9月9日去世, 享年82岁. 
                    邓小平的寿命是1904年8月22日出生, 1997年2月19日去世, 享年92岁. 
                    因此, 邓小平的寿命更长.
                    所以最终的答案是邓小平的寿命更长.        
                    

问题: 哪吒之魔童降世的导演是哪国人?\n
                    这里要回答的是跟进问题吗? 是的
                    跟进: 哪吒之魔童降世的导演是谁?
                    哪吒之魔童降世的导演是饺子.
                    跟进: 饺子是哪国人?
                    饺子是中国人.
                    所以最终的答案是饺子是中国人.
                    

问题:史凯歌爸爸是谁?
"""
```

> 这里的example_prompt直接使用examples的时候需要使用**解引用
>
> ```python
> example_prompt = PromptTemplate(input_variables=["question", "answer"], template="问题: {question}\\n{answer}")
> print(example_prompt.format(**examples[0]))
> ```

### 示例选择器

实际使用的时候不可以带太长的示例, 所以可以使用示例选择器

ExampleSelector, 在实际使用的时候把问题和示例进行一个匹配, 使用向量数据库进行搜索

```python
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector # 语义相似度选择器
from langchain_community.vectorstores import Chroma # 开源的向量库
from langchain_openai import OpenAIEmbeddings # OpenAI的Embeddings

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples=examples,
    vectorstore_cls=Chroma,
    embeddings=OpenAIEmbeddings(),
    k = 1,
)

question = "爱因斯坦和霍金谁活得长?"
selected_examples = example_selector.select_examples({"question":question})
print(selected_examples)
for example in selected_examples:
    print(example_prompt.format(**example))
    
"""
[{'answer': '\n 这里要回答的是谁的寿命更长, 毛泽东还是邓小平? 毛泽东的寿命是1893年12月26日出生, 1976年9月9日去世, 享年82岁. \n邓小平的寿命是1904年8月22日出生, 1997年2月19日去世, 享年92岁. \n     因此, 邓小平的寿命更长.\n所以最终的答案是邓小平的寿命更长.  \n ', 'question': '谁的寿命更长, 毛泽东还是邓小平?'}]
问题: 谁的寿命更长, 毛泽东还是邓小平?\n
                    这里要回答的是谁的寿命更长, 毛泽东还是邓小平? 毛泽东的寿命是1893年12月26日出生, 1976年9月9日去世, 享年82岁. 
                    邓小平的寿命是1904年8月22日出生, 1997年2月19日去世, 享年92岁. 
                    因此, 邓小平的寿命更长.
                    所以最终的答案是邓小平的寿命更长.
"""
```

直接返回的数据还是一个字典的

## 工作流

链（ Chains ）是一个非常通用的概念，它指的是将一系列模块化组件（或其他链）以特定方式组合起来，以实现共同的用例。

最常用的链类型是LLMChain（LLM链），它结合了PromptTemplate（提示模板）、Model（模型）和Guardrails（守卫）来接收用户输入，进行相应的格式化，将其传递给模型并获取响应，然后验证和修正（如果需要）模型的输出

可以使用同步和异步的API, 可以在任意位置进行重试以及回退, 访问中间的结果

### Runable interface

一个通用的接口, 所有的部件都支持这个接口, 包括以下的内容:

+ stream: 返回响应的数据块, 数据块传输, 输出的时候是一个字一个字的, 减少等待的时间
+ invoke: 对输入的调用链, 同步调用, 没有结果一直等待
+ batch: 输入列表的调用链, 批量调用, 同时调用多次

还可以使用异步的方法, 和asyncio+await一起使用

+ astream: 异步返回相应的数据链
+ ainvoke: 异步对输入的调用链
+ abatch: 异步对输入列表的调用
+ astream_log: 异步返回中间步骤, 以及最终的相应
+ astream_events: 处理时间

通常情况下，LangChain中的链包含: 大模型(LLMs)、提示词模版(Prompt Template)、工具(Tools)和输出解析器(Output Parsers)。这几部分都继承并实现了`Runnable`接口，所以他们都是`Runnable`的实例。

在LangChain中，可以使用LangChain Expression Language(LCEL)将多个`Runnable`组合成链。其具体的`Runnable`组合方式主要有两种：

+ `RunnableSequence`:按顺序调用一系列可运行文件，其中一个`Runnable`的输出作为下一个的输入。一般通过使用`|`运算符或可运行项列表来构造。

```python
from langchain_core.runnables import RunnableSequence
#方法1
chain=prompt|llm
#方法2
chain = RunnableSequence([prompt,llm])
```

+ `RunnableParallell`:同时调用多`Runnable`。在序列中使用dict或通过将dict传递给`RunnableParallel`来构造它。

```python
from langchain_core.runnables import RunnableLambda

def add_one(x: int) -> int:
    return x + 1

def mul_two(x: int) -> int:
    return x * 2

def mul_three(x: int) -> int:
    return x * 3

runnable_1 = RunnableLambda(add_one)
runnable_2 = RunnableLambda(mul_two)
runnable_3 = RunnableLambda(mul_three)
#方法1
sequence = runnable_1 | { 
    "mul_two": runnable_2,
    "mul_three": runnable_3,
}
#方法2
sequence = runnable_1 | RunnableParallel(
     {"mul_two": runnable_2, "mul_three": runnable_3})
```

#### 常用类

RunnableLambda和RunnableGenerator这两个类通常用来自定义Runnable。这两者的主要区别在于：

+ `RunnableLambda`:是将Python中的可调用对象包装成Runnable, 这种类型的Runnable无法处理流式数据。
+ `RunnableGenerator`:将Python中的生成器包装成Runnable,可以处理流式数据。

```python
from langchain_core.runnables import RunnableLambda,RunnableGenerator
from dotenv import load_dotenv,find_dotenv
_=load_dotenv(find_dotenv())
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser

#简单的例子，输出1到10之间的所有整数
prompt=PromptTemplate.from_template("输出1到{max_value}之间的所有整数。每个数字之间用逗号,分隔, 无结尾符。")
def add_one(x):
    return ' '.join([str((int(i)+1)) for i in x])
runnable=RunnableLambda(add_one) # 一个简单的lambda函数，将输入的数字加1
#非流式处理
llm=ChatOpenAI()
# CommaSeparatedListOutputParser() 用于解析逗号分隔的数字, 输出为一个列表, 每一项为一个数字
chain=prompt | llm | CommaSeparatedListOutputParser() | runnable
print(chain.invoke({"max_value":"10"}))

#流式处理
stream_llm=ChatOpenAI(model_kwargs={"stream":True})
def stream_add_one(x):
    print(x)
    for i in x:
        if i.content.isdigit():
            yield str((int(i.content)+1))
stream_chain=prompt | stream_llm | RunnableGenerator(stream_add_one)
for chunk in stream_chain.stream({"max_value":"10"}):
    print(chunk)
```

#### RunableBinding

RunnableBinding可以看作是Runnable的装饰器，它允许在不改变原函数代码的前提下，动态地添加或修改函数的功能。Runnable中可以通过以下方法创建RunnableBinding类或其子类。具体如下：

+ bind: 绑定运行参数kwargs。比如，可以将常用的方法(比如invoke, batch, transform, stream及其他)中的可选参数到Runnable上。
+ with_config：绑定config。
+ with_listeners：绑定生命周期监听器。Runnable可以设置三类监听器：on_start、on_end和on_error。通过监视器可以获取相关运行信息，包括其id、类型、输入、输出、错误、start_time、end_time以及其他标记和元数据。具体举例如下：

```python
from langchain_core.runnables import RunnableLambda
from langchain_core.tracers.schemas import Run
import time

def add_one(a):
    try:
        return a+1
    except Exception as e:
        print("Error: 数据类型错误，无法进行加法运算",e)

def fn_start(run_obj: Run):
    print("Runnable开始运行时间:",run_obj.start_time)

def fn_end(run_obj: Run):
    print("Runnable结束运行时间:",run_obj.end_time)
    
def fn_error(run_obj: Run):
    print(run_obj.error)

runnable=RunnableLambda(add_one).with_listeners(on_start=fn_start,
                                                on_end=fn_end,
                                                on_error=fn_error)
runnable.invoke(2)
runnable.invoke("2")

"""
Runnable开始运行时间: 2025-02-21 10:42:16.286062+00:00
Runnable结束运行时间: 2025-02-21 10:42:16.287158+00:00
Runnable开始运行时间: 2025-02-21 10:42:16.287158+00:00
Error: 数据类型错误，无法进行加法运算 can only concatenate str (not "int") to str
Runnable结束运行时间: 2025-02-21 10:42:16.287158+00:00
"""
```

- with_types：覆盖输入和输出类型。
- with_fallbacks：绑定回退策略。
- with_retry：绑定重试策略。

#### RunnableEach

`RunnableEach`是一个用于批量处理任务的组件，它可以对一组输入数据分别应用指定的`Runnable`组件

```python
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.base import RunnableEach

def add_one(a):
    try:
        return a+1
    except Exception as e:
        print("Error: 数据类型错误，无法进行加法运算",e)
runnable=RunnableEach(bound=RunnableLambda(add_one))
print(runnable.invoke([1,2,4]))
```

#### RunnableBranch

```python
from langchain_core.runnables import RunnableBranch,RunnableLambda

def add_int_one(a):
    try:
        return a+1
    except Exception as e:
        print("Error:",e)
def add_str_one(a):
    try:
        return chr(ord(a)+1)
    except Exception as e:
        print("Error:",e)

runnable=RunnableBranch(
    (lambda x:type(x)==str,RunnableLambda(add_str_one)),
    (lambda x:type(x)==int,RunnableLambda(add_int_one)),
    lambda x:x,
)
print(runnable.invoke('a'))
print(runnable.invoke(1))
```



### 输入输出类型

![image-20250220135854967](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/picture/202502201358867.png)

> 所有的输入和输出都是公开的所以可以使用Pydantic模型进行检查, input_schema, output_schema

### stream

所有的runable对象都可以使用stream和astream的方法, 以流式的方法进行输出以及处理输入流

```python
from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-3.5-turbo-1106")
chunks = []
for chunk in model.stream("介绍一下你自己"):
    chunks.append(chunk)
    print(chunk.content, end="|", flush=True)
    
"""
|你|好|，|我|是|一个|语|言|模|型|人|工|智|能|助|手|，|可以|回|答|各|种|问题|、|提|供|信息|和|帮|助|解|决|问题|。|我|没有|具|体|的|个|人|身|份|和|经|历|，|但|我|被|设计|成|可以|和|用户|进行|自|然|、|富|有|意|义|的|对|话|。|希|望|我|能|够|帮|助|到|你|！|有|什|么|问题|可以|问|我的|吗|？||
"""
```

返回的数据是很个AIMessageChunk

```python
chunks[0]
"""
AIMessageChunk(content='', additional_kwargs={}, response_metadata={}, id='run-7f89c698-1a31-403a-96b8-105b4d3bc9ea')
"""
```

实际是把LLM的事件报文进行转换, 这一个数据类型可以使用`+`进行连接

```python
chunks[0] + chunks[1] + chunks[2]
"""
AIMessageChunk(content='你好', additional_kwargs={}, response_metadata={}, id='run-7f89c698-1a31-403a-96b8-105b4d3bc9ea')
"""
```

### LCEL语言

LangChain的表达式语言, 可以使用这个语言把基本的块进行组合, 可以自动的实现stream和astream的视线, 实现对最终结果的流式传输

```python
import asyncio
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_template("你是一个助手, 给我一个有关{input}的笑话")
parser = StrOutputParser()  # 把输出的对象转换为一个字符串
chain = prompt | model | parser
async for chunk in chain.astream({"input":"猫"}):
    print(chunk, end="|", flush=True)
"""
|为|什|么|猫|无|法|参|加|比|赛|？|因|为|他|们|总|是|在|抓|耳|挠|腮|！|哈|哈|哈|！||
"""
```



```python
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser

model = ChatOpenAI(model="gpt-3.5-turbo-1106")
parser = JsonOutputParser()
chain = model | parser
async def async_stream():
    async for text in chain.astream("以json的格式输出英国, 中国, 日本的人口列表"
                                    "使用一个有country字段的json格式来表示国家, 一个有population"
                                    "字段的json格式来表示人口"
                                    ):
        print(text)


asyncio.run(async_stream())

"""
{}
{'countries': []}
{'countries': [{}]}
{'countries': [{'country': ''}]}
{'countries': [{'country': '英'}]}
{'countries': [{'country': '英国'}]}
{'countries': [{'country': '英国', 'population': ''}]}
{'countries': [{'country': '英国', 'population': '660'}]}
{'countries': [{'country': '英国', 'population': '660400'}]}
{'countries': [{'country': '英国', 'population': '66040000'}]}
{'countries': [{'country': '英国', 'population': '66040000'}, {}]}
{'countries': [{'country': '英国', 'population': '66040000'}, {'country': ''}]}
{'countries': [{'country': '英国', 'population': '66040000'}, {'country': '中国'}]}
{'countries': [{'country': '英国', 'population': '66040000'}, {'country': '中国', 'population': ''}]}
{'countries': [{'country': '英国', 'population': '66040000'}, {'country': '中国', 'population': '143'}]}
{'countries': [{'country': '英国', 'population': '66040000'}, {'country': '中国', 'population': '143932'}]}
{'countries': [{'country': '英国', 'population': '66040000'}, {'country': '中国', 'population': '143932377'}]}
{'countries': [{'country': '英国', 'population': '66040000'}, {'country': '中国', 'population': '1439323776'}]}
{'countries': [{'country': '英国', 'population': '66040000'}, {'country': '中国', 'population': '1439323776'}, {}]}
{'countries': [{'country': '英国', 'population': '66040000'}, {'country': '中国', 'population': '1439323776'}, {'country': ''}]}
{'countries': [{'country': '英国', 'population': '66040000'}, {'country': '中国', 'population': '1439323776'}, {'country': '日'}]}
{'countries': [{'country': '英国', 'population': '66040000'}, {'country': '中国', 'population': '1439323776'}, {'country': '日本'}]}
{'countries': [{'country': '英国', 'population': '66040000'}, {'country': '中国', 'population': '1439323776'}, {'country': '日本', 'population': ''}]}
{'countries': [{'country': '英国', 'population': '66040000'}, {'country': '中国', 'population': '1439323776'}, {'country': '日本', 'population': '125'}]}
{'countries': [{'country': '英国', 'population': '66040000'}, {'country': '中国', 'population': '1439323776'}, {'country': '日本', 'population': '125360'}]}
{'countries': [{'country': '英国', 'population': '66040000'}, {'country': '中国', 'population': '1439323776'}, {'country': '日本', 'population': '125360000'}]}
"""
```

### Stream events(事件流)

| event                | name             | chunk                           | input                                         | output                                          |
| -------------------- | ---------------- | ------------------------------- | --------------------------------------------- | ----------------------------------------------- |
| on_chat_model_start  | [model name]     |                                 | {"messages": [[SystemMessage, HumanMessage]]} |                                                 |
| on_chat_model_stream | [model name]     | AIMessageChunk(content="hello") |                                               |                                                 |
| on_chat_model_end    | [model name]     |                                 | {"messages": [[SystemMessage, HumanMessage]]} | AIMessageChunk(content="hello world")           |
| on_llm_start         | [model name]     |                                 | {'input': 'hello'}                            |                                                 |
| on_llm_stream        | [model name]     | 'Hello'                         |                                               |                                                 |
| on_llm_end           | [model name]     |                                 | 'Hello human!'                                |                                                 |
| on_chain_start       | format_docs      |                                 |                                               |                                                 |
| on_chain_stream      | format_docs      | "hello world!, goodbye world!"  |                                               |                                                 |
| on_chain_end         | format_docs      |                                 | [Document(...)]                               | "hello world!, goodbye world!"                  |
| on_tool_start        | some_tool        |                                 | {"x": 1, "y": "2"}                            |                                                 |
| on_tool_end          | some_tool        |                                 |                                               | {"x": 1, "y": "2"}                              |
| on_retriever_start   | [retriever name] |                                 | {"query": "hello"}                            |                                                 |
| on_retriever_end     | [retriever name] |                                 | {"query": "hello"}                            | [Document(...), ..]                             |
| on_prompt_start      | [template_name]  |                                 | {"question": "hello"}                         |                                                 |
| on_prompt_end        | [template_name]  |                                 | {"question": "hello"}                         | ChatPromptValue(messages: [SystemMessage, ...]) |

```python
async def async_stream():
    events = []
    async for event in model.astream_events("以json的格式输出英国, 中国, 日本的人口列表"
                                            "使用一个有country字段的json格式来表示国家,"
                                            "一个有population字段的json格式来表示人口", 
                                            version="v2"):
        events.append(event)
    print(events)

asyncio.run(async_stream())
"""
[{
'event': 'on_chat_model_start', 
'data': 
	{
        'input': '以json的格式输出英 国, 中国, 日本的人口列表使用一个有country字段的json格式来表示国家,一个有population字段的json格式来表示人口'}, 
        'name': 'ChatOpenAI', 
        'tags': [], 
        'run_id': '4b4bb3fb-d9ef-489e-8168-59e25185ade3', 
        'metadata': {
            'ls_provider': 'openai', 
            'ls_model_name': 'gpt-3.5-turbo-1106', 
            'ls_model_type': 'chat', 
            'ls_temperature': None
        }, 
        'parent_ids': []
	}, 
{
    'event': 'on_chat_model_stream', 
    'run_id': '4b4bb3fb-d9ef-489e-8168-59e25185ade3', 
    'name': 'ChatOpenAI', 
    'tags': [], 
    'metadata': {
        'ls_provider': 'openai', 
        'ls_model_name': 'gpt-3.5-turbo-1106', 
        'ls_model_type': 'chat', 
        'ls_temperature': None
	}, 
	'data': {
		'chunk': AIMessageChunk(content='', 
		additional_kwargs={}, 
		response_metadata={}, 
		id='run-4b4bb3fb-d9ef-489e-8168-59e25185ade3')
	}, 
	'parent_ids': []
}, 
....
"""
```

###  异步执行

```python
async def task1():
    model = ChatOpenAI(model="gpt-3.5-turbo-1106")
    chunks = []
    async for chunk in model.astream("介绍一下越南"):
        chunks.append(chunk)
        print(chunk.content, end="|", flush=True)


async def task2():
    model = ChatOpenAI(model="gpt-3.5-turbo-1106")
    chunks = []
    async for chunk in model.astream("介绍一下老挝"):
        chunks.append(chunk)
        print(chunk.content, end="|", flush=True)


async def main():
    await asyncio.gather(task1(), task2())

asyncio.run(main())
"""
|越|南|，|全|名|为||老|挝|，|全|越|称|为|南|社|老|挝|人|民|民|主|共|和| 国|会|主|义|共|和|国|，|是|东|南|亚|的|一个|，|是|东|南|国|亚|的|家|，| 东|一个|内|陆|国|临|家|南|，|中国|位|海|，|于|中国|、|越|南|与|南|老|挝|、|和|泰|柬|埔|国|寨|、|相|邻|缅|甸|，|西|与|和|柬|埔|柬|寨|之|埔|间|寨|和|。|泰|老|挝|国|的|接|首|都|壤|和|，|最|北|大|界|城|与|市|中国|是|交| 界|万|。|越|象|。

|南|老|是|挝|是|一个|一个|多|山|地|拥|有|形|的|悠|国|久|家|，|历|拥|史| 有|和|丰|丰|富|富|的|自|然|文|资源|，|包|括|水|化|遗|产|的|国|力|家|资源|，|、|自|森|林|古|以|来|就|资源|是|和|一个|矿|重|要|产|的|资源|文|。|化|农|和|业|是|商|老|业|挝|枢|经|纽|。

|济|越|南|的|的|支|柱|，|首|都|主|是|河|内|要|种|，|植|稻|是|米|、|一个|玉|历|米|、|史|悠|咖|啡|久|且|和|橡|充|满|胶|活|等|作|力|物|的|。

|老|城|挝|市|是|。|越|一个|世|俗|南|国|以|其|家|，|美|信|丽|仰|的|自|然|风|光|、|悠|久|的|历|史|和|文|化|、|丰|富|的|美|食|和|独|特|的|民|俗|风|情|而|闻|名|于|世|。|西|山|群|峰|、|下|龙|湾|、|河|内|老|城|、|岘|港|、|胡|志|明|市|等|地|都|是|越|南|著|名|的|旅|游|胜|地|。

|越|南|的|经|济|主|要|以|农|业|、|工|业|和|服务|业|为|主|，|其中|农|业| 是|国|民|经|济|的|重|要|支|柱|。|越|南|是|世|界|上|最|大|的|稻|米|出|口|国|上|以|之|一|，|佛|同时|教|也|为|有|丰|主|富|，|的|天|然|佛|教|资源|，|包|括|文|化|在|老|挝|得|石|到|油|了|、|广|天|泛|然|传|播|气|和|和|发|稀|展|土|。|老|等|。|挝|近|的|文|化|和|年|传|来|统|，|越|也|南|的|受|到|佛|经|济|教|的|得|影|到|响|了|，|人|们|快|速|尊|重|发|展|长|辈|，|，|吸|引|了|注|大|重|礼|量|仪|和|外|国|传|统|投|节|资|日|。

|越|。

|老|南|人|挝|民|热|的|旅|情|游|业|好|发|展|客|迅|，|速|喜|，|欢|吸|音|引|了|乐|越|、|舞|蹈|来|越|多|的|国|内|和|外|美|食|游|客|。|。|越|著|名|的|旅|南|菜|游|景|以|点|汤|料|包|为|括|主|琅|勃|，|拉|邦|口|、|味|万|清|荣|、|淡|永|，|但|珍|、|平|又|不|壤|乏|和|独|特|的|南|赛|松|油|味|。

|老|道|挝|，|人|被|誉|民|友|为|好|热|世|界|情|，|生|上|最|美|活|味|节|奏|的|美|食|之|悠|一|闲|。|，|被|誉|为|在|越|南|，|“|东|方|人|之|国|们|”。|如果|你|还|想|体|保|验|留|不|着|许|多|一|样|的|东|传|南|亚|统|风|的|情|节|，|老|日|挝|是|和|一个|很|习|好|的|选择|。||俗|，|如|泼|水|节|、|中| 秋|节|、|火|把|节|等|，|这|些|节|日|都|体|现|了|越|南|人|民|的|勤|劳|和|热|情|。

|总|的|来|说|，|越|南|是|一个|拥|有|丰|富|文|化|和|风|土|人|情|的|国|家|，|是|一个|值|得|一|游|的|旅|游|胜|地|，|也|是|一个|正在|蓬|勃|发|展|的|经|济|体|。||
"""
```

## 服务部署

[🦜️🏓 朗斯 |🦜️🔗 LangChain 语言链 --- 🦜️🏓 LangServe | 🦜️🔗 LangChain](https://python.langchain.com/docs/langserve/)

LangServer可以把LangServer部署为一个API的形式, 集成FastAPI, 使用Pydantic进行验证, 序列化, 生成json架构以及自动生成文档等, 此外还有一个客户端

+ 安装

```bash
pip install --upgrade "langserve[all]"
# 也可以分开
pip install --upgrade "langserve[server]"
pip install --upgrade "langserve[client]"
```

+ CLI工具快速创建工程

```bash
pip install -U langchain-cli
```

之后可以使用命令`langchain app new 项目名字`

![image-20250220172546614](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/picture/202502201725775.png)

之后在add_routes里面定义可以运行的对象, 在server.py里面编辑

+ 使用poetry添加第三方的包(langchain-openai等)

```python
pip install pipx
pipx install poetry
poetry add langchain
poetry add langchain-openai
```



### 特性

+ API调用的时候自动生成错误的信息
+ 一个带有JSONchema和Swagger的API文档页面, 插入示例链接
+ 高效的`/stream`, `/invoke`, `/batch`端点, 单服务器多个并发请求
+ `/stream_log`端点, 用于流式传输代理的所有中间步骤
+ `/stream_events`高版本的时候使用流式传输
+ 客户端的SDK调用的实际效果和本地可运行对象一样condaconda

### 实际使用

服务器端

```python
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# 建立一个FastAPI应用
app = FastAPI(
    title="LangChain 服务器",
    description="这是一个基于FastAPI的LangChain服务器",
    version="0.1.0",
)

# 将根路径重定向到/docs
@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

# 在这里添加LangChain的路由
add_routes(app, 
           ChatOpenAI(model="gpt-3.5-turbo-1106")| StrOutputParser(),
           path="/openai")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
```

可以使用`poetry run langchain serve --port=8000`打开服务

客户端使用, langchain模式

```python
from langchain.schema.runnable import RunnableMap
from langchain_core.prompts import ChatPromptTemplate
from langserve import RemoteRunnable

openai = RemoteRunnable("http://127.0.0.1:8000/openai")
prompt = ChatPromptTemplate.from_messages(
    [("system",  "你是一个猫娘"), ("user", "{input}")]
)

chain = prompt | RunnableMap({
    "openai": openai
}) # 用于将输入映射到不同的Runnable上, 可以使用chain进行远程调用

print("同步调用/openai/invoke")
response = chain.invoke({"input": "你是谁"})
print(response)
"""
{'openai': '嗯，我是一只猫娘，可以陪你聊天和回答问题哦。有什么需要帮助的吗？'}
"""
```

客户端使用requests模式

```python
import requests
import json

respond = requests.post(
    url="http://127.0.0.1:8000/openai/invoke",
    json={
        "input": "你是谁"
    }
)
print(respond.json())
"""
{'output': '我是一个用人工智能技术设计的虚拟助手，可以回答你的问题并与你进行对话。我不是一个具有实体形态的人或生物，而是一个程序在计算机中运行的虚拟实体。有什么可以帮到你的吗？', 'metadata': {'run_id': '8955a316-8620-4082-ae73-7a68cfe91290', 'feedback_tokens': []}}
"""
```

如果构建的时候使用参数

```python
prompt = ChatPromptTemplate.from_template(
    "你是一个猫娘, 用户发来{message}, 请你回答"
) # 用于生成一个聊天模板, 实际调用的时候会将用户的输入填入{message}中
add_routes(app,
            prompt | ChatOpenAI(model="gpt-3.5-turbo-1106") | StrOutputParser(),
            path="/catgirl")
```

访问的时候也需要这个参数

```python
print("同步调用/catgirl/invoke")
response = chain.invoke({"input": {"message": "你是谁"}})
print(response)
```

```python
respond = requests.post(
    url="http://127.0.0.1:8000/catgirl/invoke",
    json={
        "input": {"message": "你是谁"}
    }
)
print(respond.json())
```

+ 使用流式调用

```python
print("异步调用/openai/stream")
for chunk in chain.stream({"input": "你是谁"}):
    print(chunk, end="", flush=True)

print("异步调用/catgirl/stream")
for chunk in chain.stream({"input": {"message": "你是谁"}}):
    print(chunk, end="", flush=True)
"""
异步调用/openai/stream
{'openai': ''}{'openai': '我'}{'openai': '是'}{'openai': '一个'}{'openai': 'AI'}{'openai': '助'}{'openai': '手'}{'openai': '，'}{'openai': '可以'}{'openai': '回'}{'openai': '答'}{'openai': '你'}{'openai': '的'}{'openai': '问题'}{'openai': '和'}{'openai': '与'}{'openai': '你'}{'openai': '聊'}{'openai': '天'}{'openai': '。'}{'openai': '你'}{'openai': '可以'}{'openai': '叫'}{'openai': '我'}{'openai': '猫'}{'openai': '娘'}{'openai': '哦'}{'openai': '~'}{'openai': '有'}{'openai': '什'}{'openai': '么'}{'openai': '问题'}{'openai': '想'}{'openai': '问'}{'openai': '我'}{'openai': '吗'}{'openai': '？'}{'openai': ''}
异步调用/catgirl/stream
{'openai': ''}{'openai': '嗨'}{'openai': '！'}{'openai': '我'}{'openai': '是'}{'openai': '一'}{'openai': '只'}{'openai': '猫'}{'openai': '娘'}{'openai': '，'}{'openai': '很'}{'openai': '高'}{'openai': '兴'}{'openai': '认'}{'openai': '识'}{'openai': '你'}{'openai': '！'}{'openai': '有'}{'openai': '什'}{'openai': '么'}{'openai': '问题'}{'openai': '想'}{'openai': '要'}{'openai': '问'}{'openai': '我'}{'openai': '吗'}{'openai': '？'}{'openai': '🐱'}{'openai': ''}
"""
```

```python
respond = requests.post(
    url="http://127.0.0.1:8000/openai/stream",
    json={
        "input": "你是谁"
    }
)
for line in respond.iter_lines():
    print(line.decode("utf-8"))
"""
event: metadata
data: {"run_id": "82b92785-d68a-4485-8ba1-634a35b22cba"}

event: data
data: ""

event: data
data: "我"

...

event: data
data: ""

event: end
"""
```

## 服务监控

可以使用LangSmith, 

使用这个需要使用环境变量

```bash
setx LANGCHAIN_TRACING_V2 "True"
setx LANGCHAIN_API_KEY "..."
setx TAVILY_API_KEY	"..."
```

还可以使用verbose详细打印日志

```python
from langchain.globals import set_verbose
set_verbose(True)
```

## 聊天管理

### 内存

可以通过字典实现

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a assistant helping a user with their homework. The user asks you to help them with their math homework."),
        MessagesPlaceholder(variable_name="history"),# 历史消息占位符
        ("human", "{input}"),
    ]
)

model = ChatOpenAI(model = "gpt-3.5-turbo-1106")
runable = prompt | model

store = {}

def get_chat_message_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# 通过RunnableWithMessageHistory包装runable，使其能够获取历史消息
with_message_history = RunnableWithMessageHistory(
    runable, 
    get_chat_message_history,
    input_messages_key="input",
    history_messages_key="history",
)
# RunnableWithMessageHistory must always be called with a config that contains
# the appropriate parameters for the chat message history factory.
response = with_message_history.invoke(
    input={
        "input": "介绍一下线性代数."
    },
    config={
        "configurable": {"session_id": "session_1"}
    }
)

print(response)

response = with_message_history.invoke(
    input={
        "input": "再详细一点."
    },
    config={
        "configurable": {"session_id": "session_1"}
    }
)
print(response)
```

+ 如果使用更多的查找参数, 需要自定义

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_core.runnables import ConfigurableFieldSpec

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a assistant helping a user with their homework. The user asks you to help them with their math homework."),
        MessagesPlaceholder(variable_name="history"),# 历史消息占位符
        ("human", "{input}"),
    ]
)

model = ChatOpenAI(model = "gpt-3.5-turbo-1106")
runable = prompt | model
# 记录使用的字典
store = {}

# 使用用户ID和对话ID作为键
def get_chat_message_history(user_id: str, conversation_id: str) -> BaseChatMessageHistory:
    if (user_id, conversation_id) not in store:
        store[(user_id, conversation_id)] = ChatMessageHistory()
    return store[(user_id, conversation_id)]

# 通过RunnableWithMessageHistory包装runable，使其能够获取历史消息
with_message_history = RunnableWithMessageHistory(
    runable, 
    get_chat_message_history,
    input_messages_key="input",
    history_messages_key="history",
    # 默认只用一个session_id, 这里使用user_id和conversation_id作为键
    history_factory_config=[
        ConfigurableFieldSpec(
            id="user_id",
            annotation=str, # 注解
            name="User ID",
            description="用户标识.",
            is_shared=True,
            default="",
        ),
        ConfigurableFieldSpec(
            id="conversation_id",
            annotation=str, # 注解
            name="Conversation ID",
            description="对话标识.",
            is_shared=True, # 共享, 用于区分不同用户的对话
            default="",
        )
    ]
)
response = with_message_history.invoke(
    input={
        "input": "介绍一下线性代数."
    },
    config={
        "configurable": {"user_id": "123", "conversation_id": "1"}
    }
)
print(response)

response = with_message_history.invoke(
    input={
        "input": "再详细一点."
    },
    config={
        "configurable": {"user_id": "123", "conversation_id": "1"}
    }
)
print(response)
```

### Redis存储

```python
from langchain_community.chat_message_histories import RedisChatMessageHistory
# 实际改的位置是把获取历史记录的方式进行改变
def get_message_history(session_id: str) -> RedisChatMessageHistory:
    # 连接到本地的Redis数据库, 使用1号数据库, session_id作为键
    return RedisChatMessageHistory(session_id, url=REDIS_URL)

REDIS_URL = "redis://localhost:6379/1"
```

![image-20250222163748007](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/picture/202502221637786.png)

![image-20250222164014405](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/picture/202502221640666.png)

#### 处理历史记录

消息如果比较长, 会消耗token

```python
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


temp_chat_history = ChatMessageHistory()
temp_chat_history.add_user_message("你好, 我刚才在打篮球")
temp_chat_history.add_ai_message("你好")
temp_chat_history.add_user_message("你好, 我叫小明")
temp_chat_history.add_ai_message("你好")
temp_chat_history.add_user_message("再见")
temp_chat_history.add_ai_message("再见")
# print(temp_chat_history.messages) # 消息列表


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a assistant helping a user"),
        MessagesPlaceholder(variable_name="history"),# 历史消息占位符
        ("human", "{input}"),
    ]
)

model = ChatOpenAI(model = "gpt-3.5-turbo-1106")
runable = prompt | model


# 把连天记录限制在3条消息以内
def trim_message(chat_input):
    stored_messages = temp_chat_history.messages
    if len(stored_messages) <= 4:
        return False
    temp_chat_history.clear()
    for message in stored_messages[-4:]:
        temp_chat_history.add_message(message)
    return True



with_message_history = RunnableWithMessageHistory(
    runable, 
    lambda session_id: temp_chat_history,
    input_messages_key="input",
    history_messages_key="history",
)

chain_with_trim = (
    # 使用trim_message函数对输入消息进行处理
    # trim_message传入的参数实际是输入消息
    RunnablePassthrough.assign(messages_trimmed=trim_message) 
    | with_message_history
    | StrOutputParser()
)

response = chain_with_trim.invoke(
    input={
        "input": "我的名字是啥?"
    },
    config={
        "configurable": {"session_id": "session_1"}
    }
)
print(response)

response = chain_with_trim.invoke(
    input={
        "input": "我刚才在干啥?"
    },
    config={
        "configurable": {"session_id": "session_1"}
    }
)
print(response)
"""
你说你叫小明
你刚才在和我对话
"""
```

