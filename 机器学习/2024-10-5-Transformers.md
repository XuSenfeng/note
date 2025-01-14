---
 layout: post
title: "Transformers" 
date:   2024-8-5 15:39:08 +0800
tags: AI 机器学习
---

# Transformers

## HuggingFace

注册一个账户

[Hugging Face – The AI community building the future.](https://huggingface.co/welcome)

## 安装环境

是HuggingFace出品的目前最火的自然语言处理工具包, 实现大量的基于这一个架构的预训练模型以及其他各种模型

这是一整个生态环境

![Screenshot_20241005_230004](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410052300340.jpg)

```bash
conda create -n transforms python=3.9

conda activate transforms

pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

pip install transformers datasets evaluate peft accelerate gradio optimum sentencepiece

pip install jupyterlab scikit-learn pandas matplotlib tensorboard nltk rouge
```

![image-20241006101635925](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410061016954.png)

在使用vscode开发jupyter的时候, 需要选择一下内核

### 简单测试

```python
import gradio as gr
from transformers import *

gr.Interface.from_pipeline(pipeline("text-classification", model="uer/roberta-base-finetuned-dianping-chinese")).launch()
```

启动一个评价分类器, 评价这一个评价的评分

```python
# 导入gradio
import gradio as gr
# 导入transformers相关包
from transformers import pipeline
# 通过Interface加载pipeline并启动阅读理解服务
# 如果无法通过这种方式加载，可以采用离线加载的方式
gr.Interface.from_pipeline(pipeline("question-answering", model="uer/roberta-base-chinese-extractive-qa")).launch()
```

> 一个机器阅读理解的模型
>
> ![image-20241006112326045](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410061123128.png)

## 选取

[Tasks - Hugging Face](https://huggingface.co/tasks)

[What is Question Answering? - Hugging Face](https://huggingface.co/tasks/question-answering)

可以在这里面选取他推荐的模型, 数据集, 测试算法

![image-20241008181712046](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410081817288.png)

## 基础组件Pipeline

把数据进行预处理, 模型调用以及模型调用以及结果后处理组装为一个流水线, 使得我们输入的文本可以获取为最后的结果

![image-20241006113243862](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410061132916.png)

+ 可以处理的任务

![image-20241006114036732](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410061140780.png)

![image-20241006114053198](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410061140248.png)

> audio-classification ：音频分类
> automatic-speech-recognition ：自动语音识别
> text-to-audio ：文本转音频
> feature-extraction ：特征提取
> text-classification ：文本分类
> token-classification ：标记分类
> question-answering ：问答系统
> table-question-answering ：表格问答
> visual-question-answering ：视觉问答
> document-question-answering ：文档问答
> fill-mask ：填充掩码
> summarization ：摘要生成
> translation ：翻译
> text2text-generation ：文本生成
> text-generation ：文本生成
> zero-shot-classification ：零样本分类
> zero-shot-image-classification ：零样本图像分类
> zero-shot-audio-classification ：零样本音频分类
> image-classification ：图像分类
> image-feature-extraction ：图像特征提取
> image-segmentation ：图像分割
> image-to-text ：图像转文本
> object-detection ：物体检测
> zero-shot-object-detection ：零样本物体检测
> depth-estimation ：深度估计
> video-classification ：视频分类
> mask-generation ：掩码生成
> image-to-image ：图像到图像

```python
from transformers.pipelines import SUPPORTED_TASKS
for k, v in SUPPORTED_TASKS.items():
    print(k, v)
```

> 可以使用以上的方法进行查看

### 使用说明

![image-20241006125917307](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410061259357.png)

```python
class TextClassificationPipeline(Pipeline):
    """
    Text classification pipeline using any `ModelForSequenceClassification`. See the [sequence classification
    examples](../task_summary#sequence-classification) for more information.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> classifier = pipeline(model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
    >>> classifier("This movie is disgustingly good !")
    [{'label': 'POSITIVE', 'score': 1.0}]

    >>> classifier("Director tried too much.")
    [{'label': 'NEGATIVE', 'score': 0.996}]
```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)
    
    This text classification pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"sentiment-analysis"` (for classifying sequences according to positive or negative sentiments).
    
    If multiple classification labels are available (`model.config.num_labels >= 2`), the pipeline will run a softmax
    over the results. If there is a single label, the pipeline will run a sigmoid over the result.
    
    The models that this pipeline can use are models that have been fine-tuned on a sequence classification task. See
    the up-to-date list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=text-classification).
    """

```

这一个里面只有最简单的方法调用, 在实际使用可以看他的`__call__`方法

```python
def __call__(self, inputs, **kwargs):
        """
        Classify the text(s) given as inputs.

        Args:
            inputs (`str` or `List[str]` or `Dict[str]`, or `List[Dict[str]]`):
                One or several texts to classify. In order to use text pairs for
                your classification, you can send a
                dictionary containing `{"text", "text_pair"}` keys, or a list of
                those.
            top_k (`int`, *optional*, defaults to `1`):
                How many results to return.
            function_to_apply (`str`, *optional*, defaults to `"default"`):
                The function to apply to the model outputs in order to retrieve the
                scores. Accepts four different
                values:

                If this argument is not specified, then it will apply the following
                functions according to the number
                of labels:

                - If the model has a single label, will apply the sigmoid function
                on the output.
                - If the model has several labels, will apply the softmax function
                on the output.

                Possible values are:

                - `"sigmoid"`: Applies the sigmoid function on the output.
                - `"softmax"`: Applies the softmax function on the output.
                - `"none"`: Does not apply any function on the output.

        Return:
            A list or a list of list of `dict`: Each result comes as list of 
            dictionaries with the following keys:

            - **label** (`str`) -- The label predicted.
            - **score** (`float`) -- The corresponding probability.

            If `top_k` is used, one such dictionary is returned per label.
        """
```

### 加载

```python
from transformers import *
pipe = pipeline('text-classification')
```

> 使用这一个命令的时候会自动下载默认使用的模型, 一般是一个英文模型
>
> ```python
> pipe("very good")
> ```
>
> ![image-20241006120905391](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410061209566.png)
>
> 其他的模型可以在huggingface这一个网页进行寻找, 之前使用的是[uer/roberta-base-finetuned-dianping-chinese · Hugging Face](https://huggingface.co/uer/roberta-base-finetuned-dianping-chinese)这一个模型, 可以在标签进行分类筛选

![image-20241006120056003](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410061200115.png)

![image-20241006120757759](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410061207795.png)

这个就是模型的名字

```c
pipe = pipeline('text-classification', model='uer/roberta-base-finetuned-dianping-chinese')
```

+ 方法二

````python
from transformers import *

model = AutoModelForSequenceClassification.from_pretrained('uer/roberta-base-finetuned-dianping-chinese')
tokenizer = AutoTokenizer.from_pretrained('uer/roberta-base-finetuned-dianping-chinese')
pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
````

把模型以及分词器单独创建出来, 这两个需要同时指定

> 使用这两种方法加载的模型实际是使用CPU进行运行的
>
> ```python
> print(pipe.model.device)
> ```

### 使用GPU

```python
pipe = pipeline('text-classification', model='uer/roberta-base-finetuned-dianping-chinese', device=0)
```

### 图像识别示例

````python
from transformers import *

checkpoint = "google/owlvit-base-patch32"
detector = pipeline(model=checkpoint, task="zero-shot-object-detection")

import requests
from PIL import Image
# 获取检测的图片
url = "https://unsplash.com/photos/oj0zeY2Ltk4/download?ixid=MnwxMjA3fDB8MXxzZWFyY2h8MTR8fHBpY25pY3xlbnwwfHx8fDE2Nzc0OTE1NDk&force=true&w=640"
im = Image.open(requests.get(url, stream=True).raw)
im.show()
# 进行检测
predictions = detector(
    im,
    candidate_labels=["hat", "sunglasses", "book"],
)
print(predictions)

from PIL import ImageDraw
# 绘制一下结果
draw = ImageDraw.Draw(im)

for prediction in predictions:
    box = prediction["box"]
    label = prediction["label"]
    score = prediction["score"]
    xmin, ymin, xmax, ymax = box.values()
    draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=1)
    draw.text((xmin, ymin), f"{label}: {round(score,2)}", fill="red")

im.show()
````

> ```c
> class ZeroShotObjectDetectionPipeline(ChunkPipeline):
>     """
>     Zero shot object detection pipeline using `OwlViTForObjectDetection`. This pipeline predicts bounding boxes of
>     objects when you provide an image and a set of `candidate_labels`.
> 
>     Example:
> 
>     ```python
>     >>> from transformers import pipeline
> 
>     >>> detector = pipeline(model="google/owlvit-base-patch32", task="zero-shot-object-detection")
>     >>> detector(
>     ...     "http://images.cocodataset.org/val2017/000000039769.jpg",
>     ...     candidate_labels=["cat", "couch"],
>     ... )
>     [{'score': 0.287, 'label': 'cat', 'box': {'xmin': 324, 'ymin': 20, 'xmax': 640, 'ymax': 373}}, {'score': 0.254, 'label': 'cat', 'box': {'xmin': 1, 'ymin': 55, 'xmax': 315, 'ymax': 472}}, {'score': 0.121, 'label': 'couch', 'box': {'xmin': 4, 'ymin': 0, 'xmax': 642, 'ymax': 476}}]
> 
>     >>> detector(
>     ...     "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png",
>     ...     candidate_labels=["head", "bird"],
>     ... )
>     [{'score': 0.119, 'label': 'bird', 'box': {'xmin': 71, 'ymin': 170, 'xmax': 410, 'ymax': 508}}]
>    ```
> 
>     Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)
> 
>     This object detection pipeline can currently be loaded from [`pipeline`] using the following task identifier:
>     `"zero-shot-object-detection"`.
> 
>     See the list of available models on
>     [huggingface.co/models](https://huggingface.co/models?filter=zero-shot-object-detection).
>     """
> ```

![image-20241006131935059](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410061319203.png)

![image-20241006131947210](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410061319339.png)

### 处理流程

+ 初始化Tokenizer

```python
tokenizer = AutoTokenizer.from_pretrained('uer/roberta-base-finetuned-dianping-chinese')
```

+ 初始化Model

```python
model = AutoModelForSequenceClassification.from_pretrained('uer/roberta-base-finetuned-dianping-chinese')
```

+ 数据预处理

```python
imput_text = "可以"
inputs = tokenizer(input_text, return_tensors='pt') # 返回值是一个pytorch tensor
```

+ 模型预测

```python
res = model(**inputs).logits
```

+ 结果后处理

softmax输出的是概率分布（比如在多分类问题中，Softmax输出的是每个类别对应的概率）

<img src="https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410062224000.png"/>

```python
logits = res.logits
logits = torch.softmax(logits, dim=1) # 在维度1进行计算

pred = torch.argmax(logits, dim=1).item()
```

![image-20241006222812760](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410062228807.png)

## Tokenizer

进行数据的预处理

1. 分词: 使用分词器对文本数据进行分词
2. 构建词典: 根据分词器的处理结果, Step2构建词典:根据数据集分词的结果，构建词典映射(这一步并不绝对，如果采用预训练词向量，词典映射要根据词向量文件进行处理);
3. 数据转换:根据构建好的词典，将分词处理后的数据做映射，将文本序列转换为数字序列;
4. 数据填充与截断:在以batch输入到模型的方式中，需要对过短的数据进行填充，过长的数据进行截断，保证数据长度符合模型能接受的范围，同时batch内的数据维度大小一致。

在使用Tokenizer的时候可以进行以上的所有处理

### 基本使用

![image-20241006223441164](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410062234321.png)

```python
from transformers import AutoTokenizer
```

> 不同的模型会使用不同的Tokenizer, 所以这里封装了一个AutoTokenizer, 根据传入的参数确定最后的结果

+ 加载分词器

```python
sen = "我史凯歌敢吃屎!"
tokenizer = AutoTokenizer.from_pretrained("uer/roberta-base-finetuned-dianping-chinese")
```

![image-20241006223955140](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410062239203.png)

这一个可以保存在本地以及从本地进行加载

```python
tokenizer.save_pretrained("./tokenizer")
tokenizer = AutoTokenizer.from_pretrained("./tokenizer")
```

之后可以进行分词

```python
tokens = tokenizer.tokenize(sen)
```

![image-20241006224426467](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410062244514.png)

+ 可以查看一下这一个分词器的词典

![image-20241006224655209](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410062246264.png)

![image-20241006224712011](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410062247055.png)

+ 把词语转换为id

```python
ids = tokenizer.convert_tokens_to_ids(tokens)
```

![image-20241006224944657](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410062249711.png)

> 也可以反过来转换
>
> ![image-20241006225045369](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410062250417.png)
>
> ![image-20241006225224620](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410062252671.png)

+ 填充截断

```python
# 填充
ids = tokenizer.encode(sen, add_special_tokens=False, padding='max_length', max_length=10)
```

![image-20241006225920394](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410062259454.png)

```python
# 截断
ids = tokenizer.encode(sen, add_special_tokens=False, truncation=True, max_length=5)
```

![image-20241006230016792](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410062300854.png)

在截断的时候如果添加标志位, 这一个标志位不会被截断

实际使用的时候, 需要区分一下那一部分是填充的数据, 以及区分一下句子的前后

```python
attention_mask = [1 if idx != 0 else 0 for idx in ids]
token_type_ids = [0] * len(ids)
```

![image-20241006230541868](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410062305927.png)

### 简便使用方法

#### 单个数据

```python
ids = tokenizer.encode(sen)
```

![image-20241006225340157](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410062253211.png)

> ![image-20241006225442970](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410062254025.png)
>
> 这一个模型在处理的时候会加入用于区分句子的token可以使用参数取消
>
> ````python
> ids = tokenizer.encode(sen, add_special_tokens=False)
> str = tokenizer.decode(ids, skip_special_tokens=True)
> ````

```python
ids = tokenizer.encode_plus(sen, add_special_tokens=True, max_length=20, padding='max_length')
```

```bash
{
'input_ids': [101, 2769, 1380, 1132, 3625, 3140, 1391, 2241, 106, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
}
```

> 也可以直接使用`ids = tokenizer(sen, add_special_tokens=True, max_length=20, padding='max_length')`
>
> token_type_ids用于标识不同文本片段的token类型。在BERT模型中，输入文本可能包含多个文本片段，例如问题和答案。token_type_ids可以用来区分不同文本片段的token，帮助模型捕捉文本之间的关联信息。通过将不同文本片段的token赋予不同的token_type_ids，模型可以更好地理解每个文本片段之间的关系。
>
> attention_mask用于控制哪些token对于模型是可见的，哪些token应该被屏蔽掉。在BERT模型中，输入文本通常会进行padding使得输入序列长度相同。通过在attention_mask中将padding的token对应的位置设为0，模型可以忽略这些padding token，避免对其进行不必要的计算，提高了模型的计算效率。

#### 处理多个数据(速度更快)

```c
sen = ["我史凯歌敢eat dinner!", 
       "史凯歌我要eat dinner", 
       "一天要吃三斤dinner"]
ids = tokenizer.batch_encode_plus(sen, add_special_tokens=True, max_length=20, padding='max_length')
```

> ```bash
> {
> 'input_ids': [
> [101, 2769, 1380, 1132, 3625, 3140, 1391, 2241, 106, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
> [101, 1380, 1132, 3625, 2769, 6206, 1391, 2241, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [101, 671, 1921, 6206, 1391, 676, 3165, 2241, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
> ], 
> 'token_type_ids': [
> [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
> [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
> [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
> ], 
> 'attention_mask': [
> [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
> [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
> [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
> ]}
> ```

### Fast / Slow Tokenizer

Fast Tokenizer是基于Rust实现的, Slow Tokenizer是基于python实现的速度比较慢

```python
tokenizer = AutoTokenizer.from_pretrained("uer/roberta-base-finetuned-dianping-chinese", use_fast=False)
```

![image-20241006231850680](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410062318774.png)

![image-20241006232053744](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410062320847.png)

![image-20241006232109337](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410062321402.png)

fast tokenizer会有更多的返回值, 主要是应用于命名实体识别以及qa

```python
sen = "我史凯歌敢吃shit! dreaming"
inputs = tokenizer(sen, return_offsets_mapping=True)
```

> ```bash
> {
> 'input_ids': [101, 2769, 1380, 1132, 3625, 3140, 1391, 11772, 8165, 106, 10252, 8221, 102], 
> 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
> 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
> 'offset_mapping': [(0, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 9), (9, 10), (10, 11), (12, 17), (17, 20), (0, 0)]
> }
> ```
>
> ![image-20241006232704170](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410062327233.png)
>
> 有时候一个词会被拆分为多个部分, 可以使用这一个进行对对应, None对应的是(0, 0), 英文的时候比较明显, 第(6, 9)个字符和(9, 10)是一个英文单词被拆分为两个

### 其它参数

在下载一些自主开发的分词器的时候, 需要指定一个参数`trust_remote_code=True`

```python
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
```

## Model

原始的Transform分为编码器(Encoder)以及解码器(Decoder)模型, Encoder部分接收输入并且为他构建特征表示, Decoder使用Encoder的编码结果以及其他的输入序列生成目标序列

无论是编码器还是解码器都是多个TransformsBlock堆叠而成的

TransformsBlock由注意力机制(Attention)以及FFN组成

![image-20240924105336210](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202409241053267.png)

> **注意力机制**, 在计算当前的词的特征表示的时候, 可以通过注意力机制有选择性的告诉模式要使用哪一部分的上下文

![image-20241007101012759](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410071010900.png)

![image-20241007101132383](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410071011461.png)

### AutoModel

AutoModel是Hugging Face的Transformers库中的一个非常实用的类，它属于自动模型选择的机制。这个设计允许用户在不知道具体模型细节的情况下，根据给定的模型名称或模型类型自动加载相应的预训练模型。它减少了代码的重复性，并提高了灵活性，使得开发者可以轻松地切换不同的模型进行实验或应用。

### Model Head

Model Head在预训练模型的基础上添加一层或多层的额外网络结构来适应特定的模型任务，方便于开发者快速加载transformers库中的不同类型模型，不用关心模型内部细节。

Model Head是Transformers模型里的一层，通常用于对模型输出进行后续处理。其作用是接收模型的输出，然后将其映射成最终的输出。这可能涉及到一系列操作，比如分类、回归、生成等任务。Model Head可以将模型输出转化为适合特定任务的格式，并加入适当的损失函数进行训练。

在分类任务中，Model Head可能会接收模型输出的每个token的表示，然后将它们汇总成整个句子的表示，最终通过一个全连接层将其分类为不同的类别。

在生成任务中，Model Head可能会接收模型输出的每个token的表示，然后将其依次输入到一个解码器中生成整个序列。

总的来说，Model Head的作用是将模型输出进行最终的转化，使得模型可以完成具体的任务。

+ ForCausalLM：因果语言模型头，用于decoder类型的任务，主要进行文本生成，生成的每个词依赖于之前生成的所有词。比如GPT、Qwen
+  ForMaskedLM：掩码语言模型头，用于encoder类型的任务，主要进行预测文本中被掩盖和被隐藏的词，比如BERT。
+  ForSeq2SeqLM：序列到序列模型头，用于encoder-decoder类型的任务，主要处理编码器和解码器共同工作的任务，比如机器翻译或文本摘要。
+ ForQuestionAnswering：问答任务模型头，用于问答类型的任务，从给定的文本中抽取答案。通过一个encoder来理解问题和上下文，对答案进行抽取。
+ ForSequenceClassification：文本分类模型头，将输入序列映射到一个或多个标签。例如主题分类、情感分类。
+ ForTokenClassification：标记分类模型头，用于对标记进行识别的任务。将序列中的每个标记映射到一个提前定义好的标签。如命名实体识别，打标签
+ ForMultiplechoice：多项选择任务模型头，包含多个候选答案的输入，预测正确答案的选项。
  

### 下载加载(无Model Head)

```python
model = AutoModel.from_pretrained("hfl/rbt3")
```

> 使用这一个方式的时候可以从官网进行下载模型, 如果这一个失败可以在
>
> ![image-20241007105514201](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410071055312.png)
>
> 这里下载的是pytorch版本的模型
>
> ```python
> model = AutoModel.from_pretrained("E:/JHY/python/2024-10-5-transforms/hlfrbt3")
> ```
>
> 还可以使用git克隆的方式进行, 在train按钮的左侧三个点
>
> ![image-20241007110417069](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410071104166.png)
>
> 在使用默认的命令的时候, 会下载所有的三个模型, 所以可以使用
>
> ```bash
> git lfs clone "https://huggingface.co/hfl/rbt3" --include="*.bin"
> ```

### 配置

![image-20241007111339761](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410071113850.png)

> 可以在下载的时候对模型进行配置

也可以使用`model.config`获取这一个模型的配置参数, 最全的可以使用下面的函数加载

```python
config = AutoConfig.from_pretrained("E:/JHY/python/2024-10-5-transforms/hlfrbt3")
```

使用这一种方式加载的时候参数和变量是可以改变的

> ```bash
> BertConfig {
>   "_name_or_path": "E:/JHY/python/2024-10-5-transforms/hlfrbt3",
>   "architectures": [
>     "BertForMaskedLM"
>   ],
>   "attention_probs_dropout_prob": 0.1,
>   "classifier_dropout": null,
>   "directionality": "bidi",
>   "hidden_act": "gelu",
>   "hidden_dropout_prob": 0.1,
>   "hidden_size": 768,
>   "initializer_range": 0.02,
>   "intermediate_size": 3072,
>   "layer_norm_eps": 1e-12,
>   "max_position_embeddings": 512,
>   "model_type": "bert",
>   "num_attention_heads": 12,
>   "num_hidden_layers": 3,
>   "output_past": true,
>   "pad_token_id": 0,
>   "pooler_fc_size": 768,
>   "pooler_num_attention_heads": 12,
>   "pooler_num_fc_layers": 3,
>   "pooler_size_per_head": 128,
>   "pooler_type": "first_token_transform",
>   "position_embedding_type": "absolute",
>   "transformers_version": "4.44.2",
>   "type_vocab_size": 2,
>   "use_cache": true,
>   "vocab_size": 21128
> }
> ```

输出是一个BertConfig类, 这个类继承于PretrainedConfig类, 有很多的参数是在上面没有显示的

实际功能

```python
sen = "我爱北京天安门"
tokenizer = AutoTokenizer.from_pretrained("E:/JHY/python/2024-10-5-transforms/hlfrbt3")
inputs = tokenizer(sen, return_tensors="pt")
output = model(**inputs)
```

![image-20241007112810082](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410071128164.png)

> ```bash
>           9.1205e-01, -9.9994e-01, -3.5651e-01,  9.9433e-01,  9.3075e-01,
>          -3.3699e-01,  9.9916e-01, -1.0331e-01]], grad_fn=<TanhBackward0>), hidden_states=None, past_key_values=None, attentions=None, cross_attentions=None)
> ```

> 这时候的输出有一部分的数据是没有参数的, 可以在加载的时候加一个参数
>
> ```c
> model = AutoModel.from_pretrained("E:/JHY/python/2024-10-5-transforms/hlfrbt3", output_attentions=True)
> ```
>
> ![image-20241007113159856](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410071131938.png)

这一个模型是一个不带Model Header的

![image-20241007113346059](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410071133143.png)

一条数据, 长度为9的inputids

### 有Model Head

```python
from transformers import AutoModelForSequenceClassification
clz_model = AutoModelForSequenceClassification.from_pretrained("E:/JHY/python/2024-10-5-transforms/hlfrbt3", num_labels=2)
clz_model(**inputs)
```

> ```bash
> SequenceClassifierOutput(loss=None, logits=tensor([[ 0.0296, -0.4734]], grad_fn=<AddmmBackward0>), hidden_states=None, attentions=None)
> ```
>
> 没有传label, 所以这里的loss=None

可以使用参数num_labels设置输出的分类个数

![image-20241007114416817](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410071144912.png)

![image-20241007114337851](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410071143951.png)

这里的bert是一个模型编码器[【理论篇】是时候彻底弄懂BERT模型了(收藏)-CSDN博客](https://blog.csdn.net/yjw123456/article/details/120211601#:~:text=本文详细解析BERT)

### 微调代码示例

使用数据集[SophonPlus/ChineseNlpCorpus](https://github.com/SophonPlus/ChineseNlpCorpus)

![image-20241007123535302](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410071235397.png)

#### 获取数据集以及处理

```python
# 文本分类模型微调的示例
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 加载数据
import pandas as pd

# data = pd.read_csv("../dataset/ChnSentiCorp_htl_all.csv")
# data.head()

# data = data.dropna()    # 删除缺失值
# data.iloc[1]

# 创建dataset
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.data = pd.read_csv("../dataset/ChnSentiCorp_htl_all.csv")
        self.data = self.data.dropna()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx]["review"], self.data.iloc[idx]["label"]
    
    dataset = MyDataset()
    
for i in range(5):
    print(dataset[i])
"""
('距离川沙公路较近,但是公交指示不对,如果是"蔡陆线"的话,会非常麻烦.建议用别的路线.房间较为简单.', 1)
('商务大床房，房间很大，床有2M宽，整体感觉经济实惠不错!', 1)
('早餐太差，无论去多少人，那边也不加食品的。酒店应该重视一下这个问题了。房间本身很好。', 1)
('宾馆在小街道上，不大好找，但还好北京热心同胞很多~宾馆设施跟介绍的差不多，房间很小，确实挺小，但加上低价位因素，还是无超所值的；环境不错，就在小胡同内，安静整洁，暖气好足-_-||。。。呵还有一大优势就是从宾馆出发，步行不到十分钟就可以到梅兰芳故居等等，京味小胡同，北海距离好近呢。总之，不错。推荐给节约消费的自助游朋友~比较划算，附近特色小吃很多~', 1)
('CBD中心,周围没什么店铺,说5星有点勉强.不知道为什么卫生间没有电吹风', 1)
"""
# 划分数据集
from torch.utils.data import random_split

trainset, validset = random_split(dataset, lengths=[0.9, 0.1])   # 乱序划分数据集

print(len(trainset), len(validset))
"""
(6989, 776)
"""

# 建立DataLoader
from torch.utils.data import DataLoader
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)  # 乱序, 一组大小为32
validloader = DataLoader(validset, batch_size=32, shuffle=False) # 不乱序
```

> ![image-20241007130723273](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410071307339.png)
>
> 也可以使用`next(enumerate(trainloader))`
>
> 默认的时候使用这一个函数进行聚合的结果是文字聚集为一个元组, 数字是一个tensor, 如果想要使用Tokenizer进行处理这一个数据, 文字为一个list可以重写一下`collate_fn, input_ids是Tokenizer字典里面的一项的名字, 由于在Bert模型里面labels是其中另一个参数, 所以把这两个打包
>
> ![image-20241007133339446](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410071333543.png)
>
> ```python
> import torch 
> tokenizer = AutoTokenizer.from_pretrained("../hlfrbt3")
> def j_collate_fn(batch):
>      texts, labels = [], []
>      for text, label in batch:
>          texts.append(text)
>          labels.append(label)
>      # 128是模型的最大长度
>      inputs = tokenizer(texts, padding="max_length", truncation=True, max_length=128,
>                         return_tensors="pt") 
>      inputs["labels"] = torch.tensor(labels)
>     # 返回一个字典里面两个Value为tensor格式
>      return inputs
> # 建立DataLoader
> from torch.utils.data import DataLoader
> # 乱序, 一组大小为32
> trainloader = DataLoader(trainset, batch_size=32, shuffle=True, collate_fn=j_collate_fn)
> # 不乱序
> validloader= DataLoader(validset, batch_size=32, shuffle=False, collate_fn=j_collate_fn)
> ```

#### 导入模型以及优化

```python
from torch.optim import AdamW
# 导入模型
model = AutoModelForSequenceClassification.from_pretrained("../hlfrbt3")
if torch.cuda.is_available():
    model = model.cuda()
# 定义优化器
optimizer = AdamW(model.parameters(), lr=1e-5) # 1e-5是学习率, 迁移学习使用的一般比较低
```

#### 实际训练函数

```python
def evaluate():
    """
    Description: 评估模型在验证集上的性能
    Returns:
        模型的准确率
    """
    model.eval()
    acc_num = 0
    with torch.no_grad():
        for batch in validloader:
            if torch.cuda.is_available():
                batch = {k:v.cuda() for k, v in batch.items()}
            outputs = model(**batch)
            pred = outputs.logits.argmax(dim=-1) # 预测的类别
            acc_num += (pred == batch["labels"].long()).float().sum().item()
    return acc_num / len(validset)

def train(epoch=3, log_step=100):
    """
    Description: 训练模型
    Args:
        epoch (int, optional): 训练的次数. Defaults to 3.
        log_step (int, optional): 打印log的步长. Defaults to 100.
    """
    global_step=0
    for ep in range(epoch):
        model.train()
        # 遍历训练集
        for batch in trainloader:
            # 将数据放到cuda上
            if torch.cuda.is_available():
                batch = {k:v.cuda() for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            outputs.loss.backward()
            optimizer.step()
            global_step += 1
            if global_step % log_step == 0:
                print(f"epoch={ep}, global_step={global_step}, loss={outputs.loss.item()}")
        # 每个epoch结束评估一次
        acc = evaluate()
        print(f"epoch={ep}, acc={acc}")
```

```python
print(f'before train {evaluate()}')
train()
```

> ```bash
> before train 0.31056701030927836
> epoch=0, global_step=100, loss=0.293989360332489
> epoch=0, global_step=200, loss=0.31614530086517334
> epoch=0, acc=0.8904639175257731
> epoch=1, global_step=300, loss=0.1351868063211441
> epoch=1, global_step=400, loss=0.17762571573257446
> epoch=1, acc=0.8865979381443299
> epoch=2, global_step=500, loss=0.17976289987564087
> epoch=2, global_step=600, loss=0.19925124943256378
> epoch=2, acc=0.8853092783505154
> ```

#### 实际使用

```python
sen = "我觉得这家饭店的饭很好吃, 体验很好!"
with torch.inference_mode():
    inputs = tokenizer(sen, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k:v.cuda() for k, v in inputs.items()}
    outputs = model(**inputs)
    pred = outputs.logits.argmax(dim=-1)
    print(pred.item())
```

+ 使用pipe进行

```python
from transformers import pipeline
model.config.id2label = id2label # 这一步也可以在模型加载的时候实现
pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
pipe(sen)
```

## Dataset

### 加载

一个可以方便的从Huggingface加载数据集的库

这里使用的模型是[madao33/new-title-chinese · Datasets at Hugging Face](https://huggingface.co/datasets/madao33/new-title-chinese)

```python
from datasets import *
dataset = load_dataset("madao33/new-title-chinese")
```

>  ```bash
>  DatasetDict({
>      train: Dataset({
>          features: ['title', 'content'],
>          num_rows: 5850
>      })
>      validation: Dataset({
>          features: ['title', 'content'],
>          num_rows: 1679
>      })
>  })
>  ```

对于其他的一些数据集, 比如glue是一个任务的集合, 需要再加一个参数指定实际加载的任务

```python
boolq_dataset = load_dataset("super_glue", "boolq", trust_remote_code=True)
boolq_dataset
```

> [aps/super_glue · Datasets at Hugging Face](https://huggingface.co/datasets/aps/super_glue)
>
> ```bash
> Downloading data: 100%|██████████| 4.12M/4.12M [00:01<00:00, 3.20MB/s]
> Generating train split: 100%|██████████| 9427/9427 [00:00<00:00, 14841.26 examples/s]
> Generating validation split: 100%|██████████| 3270/3270 [00:00<00:00, 28511.03 examples/s]
> Generating test split: 100%|██████████| 3245/3245 [00:00<00:00, 29980.43 examples/s]
> DatasetDict({
>  train: Dataset({
>      features: ['question', 'passage', 'idx', 'label'],
>      num_rows: 9427
>  })
>  validation: Dataset({
>      features: ['question', 'passage', 'idx', 'label'],
>      num_rows: 3270
>  })
>  test: Dataset({
>      features: ['question', 'passage', 'idx', 'label'],
>      num_rows: 3245
>  })
> })
> ```
>
一个数据集分为数据集, 验证集, 测试集三部分, 只想加载一部分的话

```python
boolq_dataset_train = load_dataset("super_glue", "boolq", trust_remote_code=True, split="train")
```

 也可以进一步拆分

```python
boolq_dataset_train = load_dataset("super_glue", "boolq", trust_remote_code=True, split="train[:100]")
boolq_dataset_train = load_dataset("super_glue", "boolq", trust_remote_code=True, split="train[:10%]")
boolq_dataset_train = load_dataset("super_glue", "boolq", trust_remote_code=True, split=["train[:10%]", "validation[:10%]"])
```

### 查看

```python
dataset["train"][:2]
dataset["train"]["title"][:2]
```

![image-20241007222540350](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410072225595.png)

```python
dataset["train"].features
dataset.column_names
```

![image-20241007222715762](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410072227852.png)

> 使用这种直接选取的方式获取的数据是一个字典的模式

### 划分

取一个数据集按比例划分为训练集以及测试集

```python
dataset["train"].train_test_split(test_size=0.1, stratify_by_column="lable")
```

> 在实际划分的时候指定参数stratify_by_column为以及lable可以按照这一个标签划分的比较均衡
>
> ![image-20241007223443044](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410072234130.png)

### 选取以及过滤

```python
dataset["train"].select([1, 2])
dataset["train"].filter(lambda example: "中国" in example["title"])
```

![image-20241007223822283](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410072238368.png)

### 数据映射

对每一条数据进行同一个处理

```python
def add_title(example):
    example["title"] = "Predix: " + example["title"]
    return example

prefix_dataset = dataset.map(add_title)
prefix_dataset["train"][:2]["title"]
```

![image-20241007224325640](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410072243728.png)

> 个每一个标题加一个前缀

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
def preprocess_function(examples):
    model_inputs = tokenizer(examples["content"], padding="max_length", truncation=True,
                             max_length=512)
    labels = tokenizer(examples["title"], padding="max_length", truncation=True,
                       max_length=32)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

processed_dataset = dataset.map(preprocess_function)
```

> ```bash
> DatasetDict({
>     train: Dataset({
>         features: ['title', 'content', 'input_ids', 'token_type_ids', 'attention_mask', 'labels'],
>         num_rows: 5850
>     })
>     validation: Dataset({
>         features: ['title', 'content', 'input_ids', 'token_type_ids', 'attention_mask', 'labels'],
>         num_rows: 1679
>     })
> })
> ```

```python
processed_dataset = dataset.map(preprocess_function, batched=True)
processed_dataset
```

> 如果将 `batched=True`，则 `dataset` 中的数据将按照设置的批大小进行处理，而不是一次处理一个样本。
>
> ![image-20241007225823344](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410072258436.png)

也可以使用多线程的方式进行加载

```python
processed_dataset = dataset.map(preprocess_function, num_proc=4)
processed_dataset
```

> 在使用测一个函数的时候, tokenizer是不可以传递到子进程的, 所以处理的函数需要改变一下
>
> ```python
> from transformers import AutoTokenizer
> tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
> def preprocess_function(examples, tokenizer=tokenizer):
>     model_inputs = tokenizer(examples["content"], padding="max_length", truncation=True, max_length=512)
>     labels = tokenizer(examples["title"], padding="max_length", truncation=True, max_length=32)
>     model_inputs["labels"] = labels["input_ids"]
>     return model_inputs
> ```

可以在这一个函数里面使用参数把不需要的输出字段进行删除

```python
processed_dataset = dataset.map(preprocess_function, batched=True,
                                remove_columns=dataset["train"].column_names)
processed_dataset
```

> 去除原始字段
>
> ![image-20241007230448456](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410072304567.png)

### 保存以及加载

```python
processed_dataset.save_to_disk("my_dataset")
processed_dataset = load_from_disk("my_dataset")
```

### 加载自己的数据集

如果是一个csv格式的数据集

```python
dataset = load_dataset("csv", data_files="../dataset/ChnSentiCorp_htl_all.csv")
```

![image-20241007230845712](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410072308817.png)

如果不希望有这一个默认的分组, 可以加一个参数取消

```python
dataset = load_dataset("csv", data_files="../dataset/ChnSentiCorp_htl_all.csv", split="train")
```

![image-20241007231055560](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410072310661.png)

也可以使用另一个函数加载

```python
dataset = Dataset.from_csv("E:/JHY/python/2024-10-5-transforms/dataset/ChnSentiCorp_htl_all.csv")
```

![image-20241007231431791](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410072314904.png)

在实际加载的时候可以按照一个文件夹进行加载, 也可以把data_files指定为一个数组

```python
dataset = load_dataset("csv", data_dir="../dataset", split="train")
```

![image-20241007231614570](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410072316772.png)

如果是按照pandas加载的数据

```python
import pandas as pd
df = pd.read_csv("../dataset/ChnSentiCorp_htl_all.csv")
dataset  = Dataset.from_pandas(df)
```

![image-20241008090421592](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410080919550.png)

使用from_list的时候实际是从一个字典的list里面进行加载

```python
list = [{"text": "abc"}, {"text": "bcd"}]
```

如果数据非常的复杂, 需要通过脚本的方式进行加载, 也可以使用load_dataset进行脚本加载

这里使用的测试的文本是cmrc2018

![image-20241008091829631](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410080918749.png)

在直接加载的时候, 需要指定实际的数据的位置, 否则加载的数据会分组出错

```python
dataset_json = load_dataset("json", data_files="../dataset/cmrc2018_trial.json", field="data")
```

![image-20241008092505281](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410080925401.png)

实际加载的也不是很完整

```python
import json
import datasets
from datasets import DownloadManager, DatasetInfo

class CMRC2018TRIAL(datasets.GeneratorBasedBuilder):

    def _info(self) -> DatasetInfo:
        """
            info方法, 定义数据集的信息,这里要对数据的字段进行定义
        :return:
        """
        return datasets.DatasetInfo(
            description="CMRC2018 trial",
            features=datasets.Features({
                    "id": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "answers": datasets.features.Sequence(
                        {
                            "text": datasets.Value("string"),
                            "answer_start": datasets.Value("int32"),
                        }
                    )
                })
        )

    def _split_generators(self, dl_manager: DownloadManager):
        """
            返回datasets.SplitGenerator
            涉及两个参数: name和gen_kwargs
            name: 指定数据集的划分
            gen_kwargs: 指定要读取的文件的路径, 与_generate_examples的入参数一致
        :param dl_manager:
        :return: [ datasets.SplitGenerator ]
        """
        return [datasets.SplitGenerator(name=datasets.Split.TRAIN, 
                                        gen_kwargs={"filepath": "../dataset/cmrc2018_trial.json"})]

    def _generate_examples(self, filepath):
        """
            生成具体的样本, 使用yield
            需要额外指定key, id从0开始自增就可以
        :param filepath:
        :return:
        """
        # Yields (key, example) tuples from the dataset
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
            for example in data["data"]:
                for paragraph in example["paragraphs"]:
                    context = paragraph["context"].strip()
                    for qa in paragraph["qas"]:
                        question = qa["question"].strip()
                        id_ = qa["id"]

                        answer_starts = [answer["answer_start"] for answer in qa["answers"]]
                        answers = [answer["text"].strip() for answer in qa["answers"]]
					  # 返回值的第一个字段是一个id, 之后是和_split_generators
                        # 字段里面的声明一样的数据
                        yield id_, {
                            "context": context,
                            "question": question,
                            "id": id_,
                            "answers": {
                                "answer_start": answer_starts,
                                "text": answers,
                            },
                        }

```

```python
dataset_json = load_dataset("../python-src/load_script.py", split="train", trust_remote_code=True)
num = 0
for example in dataset_json:
    num += 1
    print(example)
    if num == 5:
        break
```

![image-20241008094720254](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410080947382.png)

### Dataset with DataCollator

transforms内置一部分的DataCollator, 是不需要我们自己写的

这时候的数据还是list, 希望使用dataloader把数据拼接为batch的tensor

**注: **使用这一个的时候里面的数据只能有transform原始的字段

```bash
{'label': [1, 1, 1], 'input_ids': [[101, 6655, 4895, 2335, 3763, 1062, 6662, 6772, 6818, 117, 852, 3221, 1062, 769, 2900, 4850, 679, 2190, 117, 1963, 3362, 3221, 107, 5918, 7355, 5296, 107, 4638, 6413, 117, 833, 7478, 2382, 7937, 4172, 119, 2456, 6379, 4500, 1166, 4638, 6662, 5296, 119, 2791, 7313, 6772, 711, 5042, 1296, 119, 102], [101, 1555, 1218, 1920, 2414, 2791, 8024, 2791, 7313, 2523, 1920, 8024, 2414, 3300, 100, 2160, 8024, 3146, 860, 2697, 6230, 5307, 3845, 2141, 2669, 679, 7231, 106, 102], [101, 3193, 7623, 1922, 2345, 8024, 3187, 6389, 1343, 1914, 2208, 782, 8024, 6929, 6804, 738, 679, 1217, 7608, 1501, 4638, 511, 6983, 2421, 2418, 6421, 7028, 6228, 671, 678, 6821, 702, 7309, 7579, 749, 511, 2791, 7313, 3315, 6716, 2523, 1962, 511, 102]], 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}
```

```python
collator = DataCollatorWithPadding(tokenizer=tokenizer)
from torch.utils.data import DataLoader
data_loader = DataLoader(tokenized_datasets, collate_fn=collator, batch_size=8, shuffle=True)
```

```bash
{'input_ids': tensor([[ 101,  677, 1453,  ...,    0,    0,    0],
        [ 101, 4384, 1862,  ...,    0,    0,    0],
        [ 101, 2791, 7313,  ...,    0,    0,    0],
        ...,
        [ 101, 4289, 5401,  ...,    0,    0,    0],
        [ 101, 1168, 6809,  ..., 3613,  738,  102],
        [ 101, 2791, 7313,  ...,    0,    0,    0]]), 'token_type_ids': tensor([[0, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0],
        ...,
        [0, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0],
        ...,
        [1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 1, 1, 1],
        [1, 1, 1,  ..., 0, 0, 0]]), 'labels': tensor([1, 1, 1, 1, 0, 1, 0, 1])}
```

> 实际填充会把数据填充到这一个批次里面最长的

### 实际使用

```python
from transformers import DataCollatorWithPadding

dataset = load_dataset("csv", data_files="../dataset/ChnSentiCorp_htl_all.csv", split="train")
dataset = dataset.filter(lambda example: example["review"] is not None)
# for data in dataset:
#     print(data)

def process_function(examples):
    # 暂时不填充, 组成batch时再填充
    tokenized_example = tokenizer(examples["review"], max_length=128, truncation=True) 
    tokenized_example["label"] = examples["label"]
    return tokenized_example

tokenized_datasets = dataset.map(process_function, batched=True, 
                                 remove_columns=dataset.column_names)
print(tokenized_datasets[:3])
```

```bash
{'label': [1, 1, 1], 
'input_ids': [
[101, 6655, 4895, 2335, 3763, 1062, 6662, 6772, 6818, 117, 852, 3221, 1062, 769, 2900, 4850, 679, 2190, 117, 1963, 3362, 3221, 107, 5918, 7355, 5296, 107, 4638, 6413, 117, 833, 7478, 2382, 7937, 4172, 119, 2456, 6379, 4500, 1166, 4638, 6662, 5296, 119, 2791, 7313, 6772, 711, 5042, 1296, 119, 102], 
[101, 1555, 1218, 1920, 2414, 2791, 8024, 2791, 7313, 2523, 1920, 8024, 2414, 3300, 100, 2160, 8024, 3146, 860, 2697, 6230, 5307, 3845, 2141, 2669, 679, 7231, 106, 102], 
[101, 3193, 7623, 1922, 2345, 8024, 3187, 6389, 1343, 1914, 2208, 782, 8024, 6929, 6804, 738, 679, 1217, 7608, 1501, 4638, 511, 6983, 2421, 2418, 6421, 7028, 6228, 671, 678, 6821, 702, 7309, 7579, 749, 511, 2791, 7313, 3315, 6716, 2523, 1962, 511, 102]
], 
'token_type_ids': [
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
], 
'attention_mask': [
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]
}
```

这时候的数据还是list, 希望使用dataloader把数据拼接为batch的tensor

```python
collator = DataCollatorWithPadding(tokenizer=tokenizer)
from torch.utils.data import DataLoader
data_loader = DataLoader(tokenized_datasets, collate_fn=collator, batch_size=8, shuffle=True)
```

把数据转为batch tensor以及进行填充

### 代码优化

```python
# 文本分类模型微调的示例
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# 加载数据集
from datasets import load_dataset

dataset = load_dataset("csv", data_files="../dataset/ChnSentiCorp_htl_all.csv", split="train")
dataset = dataset.filter(lambda example: example["review"] is not None and example["label"] is not None)
print(dataset)
"""
Dataset({
    features: ['label', 'review'],
    num_rows: 7765
})
"""

# 划分数据集, 获取训练集以及测试集
datasets = dataset.train_test_split(test_size=0.1)
datasets

"""
DatasetDict({
    train: Dataset({
        features: ['label', 'review'],
        num_rows: 6988
    })
    test: Dataset({
        features: ['label', 'review'],
        num_rows: 777
    })
})
"""

import torch

tokenizer = AutoTokenizer.from_pretrained("../hlfrbt3")

def process_function(examples):
    # 暂时不填充, 组成batch时再填充
    tokenized_example = tokenizer(examples["review"], max_length=128, truncation=True) 
    tokenized_example["label"] = examples["label"]
    return tokenized_example

# 处理数据集, 把数据集转换为模型可以处理的格式(分词器编码后的格式)
tokenized_datasets = datasets.map(process_function, batched=True, remove_columns=datasets["train"].column_names)
tokenized_datasets

"""
DatasetDict({
    train: Dataset({
        features: ['label', 'input_ids', 'token_type_ids', 'attention_mask'],
        num_rows: 6988
    })
    test: Dataset({
        features: ['label', 'input_ids', 'token_type_ids', 'attention_mask'],
        num_rows: 777
    })
})
"""


# 建立DataLoader
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
trainset = tokenized_datasets["train"]
validset = tokenized_datasets["test"]
# 乱序, 一组大小为32, 进行填充
trainloader = DataLoader(trainset, batch_size=32, shuffle=True, collate_fn=DataCollatorWithPadding(tokenizer))  
validloader = DataLoader(validset, batch_size=32, shuffle=False, collate_fn=DataCollatorWithPadding(tokenizer)) # 不乱序
```

## Evaluate

是一个机器学习的模型评估函数库, 只需要一行代码可以加载各种任务的评估函数

[🤗 Evaluate (huggingface.co)](https://huggingface.co/docs/evaluate/index)

![image-20241008115134639](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410081151914.png)

```python
import evaluate
evaluate.list_evaluation_modules()
```

> 可以使用这个函数获取可以使用的评估函数, 这里面有一部分Huggingface实现, 另一部分是社区实现的, 不想看社区实现的时候可以加一个参数`include_community=False`
>
> 可以使用参数`with_details=True`获取更详细的信息
>
> ![image-20241008170841605](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410081708806.png)

```python
accuracy = evaluate.load('accuracy') # 加载
print(accuracy.description) # 获取描述以及计算方式
"""
Accuracy is the proportion of correct predictions among the total number of cases processed. It can be computed with:
Accuracy = (TP + TN) / (TP + TN + FP + FN)
 Where:
TP: True positive
TN: True negative
FP: False positive
FN: False negative
"""
print(accuracy.inputs_description)
"""
Args:
    predictions (`list` of `int`): Predicted labels.
    references (`list` of `int`): Ground truth labels.
    normalize (`boolean`): If set to False, returns the number of correctly classified samples. Otherwise, returns the fraction of correctly classified samples. Defaults to True.
    sample_weight (`list` of `float`): Sample weights Defaults to None.

Returns:
    accuracy (`float` or `int`): Accuracy score. Minimum possible value is 0. Maximum possible value is 1.0, or the number of examples input, if `normalize` is set to `True`.. A higher score means higher accuracy.

Examples:

    Example 1-A simple example
        >>> accuracy_metric = evaluate.load("accuracy")
        >>> results = accuracy_metric.compute(references=[0, 1, 2, 0, 1, 2], predictions=[0, 1, 1, 2, 1, 0])
        >>> print(results)
        {'accuracy': 0.5}

    Example 2-The same as Example 1, except with `normalize` set to `False`.
        >>> accuracy_metric = evaluate.load("accuracy")
        >>> results = accuracy_metric.compute(references=[0, 1, 2, 0, 1, 2], predictions=[0, 1, 1, 2, 1, 0], normalize=False)
        >>> print(results)
        {'accuracy': 3.0}

    Example 3-The same as Example 1, except with `sample_weight` set.
        >>> accuracy_metric = evaluate.load("accuracy")
        >>> results = accuracy_metric.compute(references=[0, 1, 2, 0, 1, 2], predictions=[0, 1, 1, 2, 1, 0], sample_weight=[0.5, 2, 0.7, 0.5, 9, 0.4])
        >>> print(results)
        {'accuracy': 0.8778625954198473}
"""
```

> 直接打印的时候会把所有的数据打印出来

```python
result = accuracy.compute(references=[0, 1, 1, 0], predictions=[0, 1, 0, 1])
result
```

在实际应用的时候数据可能不是一次性传进来的

```python
for ref, pred in zip([0, 1, 1, 0], [0, 1, 1, 0]):
    accuracy.add(references=ref, predictions=pred)
accuracy.compute()
```

```python
for ref, pred in zip([[0, 1, 1, 0], [0, 1, 1, 0]], [[0, 1, 0, 1], [0, 1, 1, 0]]):
    accuracy.add_batch(references=ref, predictions=pred)
accuracy.compute()
```

> ```python
> zip_data = zip([[0, 1, 1, 0], [0, 1, 1, 0]], [[0, 1, 0, 1], [0, 1, 1, 0]])
> list(zip_data)
> """
> [([0, 1, 1, 0], [0, 1, 0, 1]), ([0, 1, 1, 0], [0, 1, 1, 0])]
> """
> for ref, pred in zip([[0, 1, 1, 0], [0, 1, 1, 0]], [[0, 1, 0, 1], [0, 1, 1, 0]]):
>     print(ref, pred)
> """
> [0, 1, 1, 0] [0, 1, 0, 1]
> [0, 1, 1, 0] [0, 1, 1, 0]
> """
> ```

### 多指标评估函数

可以在同时进行多个评估函数

```python
clf_metrics = evaluate.combine(['accuracy', 'precision', 'recall', 'f1'])
clf_metrics.compute(references=[0, 1, 1, 0], predictions=[0, 1, 0, 1])
"""
{'accuracy': 0.5, 'precision': 0.5, 'recall': 0.5, 'f1': 0.5}
"""
```

### 评估结果可视化

这一个库只有一个雷达图的方式进行对比不同模型的结果

```python
from  evaluate.visualization import radar_plot
data = [
    {"accuracy": 0.98, "precision": 0.97, "recall": 0.99, "f1": 0.98},
    {"accuracy": 0.96, "precision": 0.99, "recall": 0.97, "f1": 0.96},
    {"accuracy": 0.92, "precision": 0.96, "recall": 0.99, "f1": 0.97},
]
model_names = ["model1", "model2", "model3"]
plot = radar_plot(data, model_names)
```

### 实际应用

```python
import evaluate
clf_metrics = evaluate.combine(['accuracy', 'f1'])

def evaluate():
    """
    Description: 评估模型在验证集上的性能
    Returns:
        模型的准确率
    """
    model.eval()
    with torch.no_grad():
        for batch in validloader:
            if torch.cuda.is_available():
                batch = {k:v.cuda() for k, v in batch.items()}
            outputs = model(**batch)
            pred = outputs.logits.argmax(dim=-1) # 预测的类别
            clf_metrics.add_batch(predictions=pred.long(), references=batch["labels"].long())
    return clf_metrics.compute()
```

## Trainer

![image-20241008190813102](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410081908319.png)

[Trainer (huggingface.co)](https://huggingface.co/docs/transformers/trainer)

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
# 获取参数集
train_args = TrainingArguments(output_dir="./checkpoints", save_safetensors=True)
train_args

from transformers import DataCollatorWithPadding
# 训练模型
# args:
#   model: 模型
#   args: 训练参数
#   train_dataset: 训练数据集
#   eval_dataset: 评估数据集
#   data_collator: 数据收集器
#   compute_metrics: 评估指标
trainer = Trainer(model=model, args=train_args, 
                  train_dataset=tokenized_datasets["train"], 
                  eval_dataset=tokenized_datasets["test"],
                  data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                  compute_metrics=eval_metrics)
trainer.train()
```

在使用这一个的时候就不再需要dataloader了, 数据经过预处理就可以使用了, 也不再需要使用cuda判定

在比较高的版本会出现训练失败的问题, 可以通过降低版本解决

```bash
pip install -U transforners==4.42.4
```

使用最少的参数的时候不会进行测评, 可以使用`trainer.evaluate()`进行测评, 参数可以单独指定使用的测试集

也可以使用`trainer.predict(tokenized_datasets["test"])`进行预测

### 主要的参数

#### 数据集

```python
per_device_train_batch_size=64,per_device_eval_batch_size=128
```

> 改变一下训练的时候使用batch_size

#### log

```python
logging_steps=100
```

设置为100步打印一次log

#### 评估

```python
evaluation_strategy="epoch"
```

每一轮进行一次评估

```python
evaluation_strategy="steps",eval_steps=100
```

每100步一次

```python
metric_for_best_model="f1"
```

> 使用哪一个评价这一个模型最好

#### 保存

```python
save_strategy="epoch"
```

保存的策略为每一轮

```python
save_total_limit=3
```

最多记录的模型轮数

```python
load_best_model_at_end=True
```

在最后留下来训练的最好的(之后的模型加载的是这一轮里面的参数)

#### 训练

```python
learning_rate=2e-5, weight_decay=0.01
```

> 训练的学习速率以及权重衰减(防止模型过拟合)

#### 结果

生成的runs这一个文件夹可以使用tensorboard查看

```bash
tensorboard --logdir dir
```

也可以使用vscode启动

![image-20241009195025104](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410091950114.png)

## 最终的示例

+ 加载数据

```python
# 文本分类模型微调的示例
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
# 加载数据集
from datasets import load_dataset

# 使用CSV文件加载数据集, 默认的时候没有分为不同训练集
dataset = load_dataset("csv",
               data_files="../dataset/ChnSentiCorp_htl_all.csv",
               split="train")
# 去除数据集里面的无线数据
dataset = dataset.filter(lambda example: 
                         example["review"] is not None 
                         and example["label"] is not None)
# 划分数据集, 数据集的0.1为测试集
datasets = dataset.train_test_split(test_size=0.1)
```

+ 分词器

```python
import torch

tokenizer = AutoTokenizer.from_pretrained("../hlfrbt3")
# 对数据预处理, 把数据转换为tensor, 以及加入目标值, 
# 这里的数据格式如下
"""
DatasetDict({
    train: Dataset({
        features: ['label', 'review'],
        num_rows: 6988
    })
    test: Dataset({
        features: ['label', 'review'],
        num_rows: 777
    })
})
"""
def process_function(examples):
    # 暂时不填充, 组成batch时再填充
    tokenized_example = tokenizer(examples["review"],
                                  max_length=128,
                                  truncation=True) 
    tokenized_example["labels"] = examples["label"]
    return tokenized_example

# 处理数据集, 把数据集转换为模型可以处理的格式(分词器编码后的格式)
# remove_columns去除原始的数据
tokenized_datasets = datasets.map(process_function,
                  batched=True, 
                  remove_columns=datasets["train"].column_names)
# 此时的数据格式如下
"""
DatasetDict({
    train: Dataset({
        features: ['input_ids', 'token_type_ids', 'attention_mask', 'labels'],
        num_rows: 6988
    })
    test: Dataset({
        features: ['input_ids', 'token_type_ids', 'attention_mask', 'labels'],
        num_rows: 777
    })
})
"""
```

