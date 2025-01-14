---
 layout: post
title: "Transformers实战" 
date:   2024-8-5 15:39:08 +0800
tags: AI 机器学习
---

# Transformers实战

## 基于Transforms的NLP解决

> NLP自然语言处理, *ModelForSequenceClassification

+ 导入相关的包 General
+ 加载数据集 Dataset
+ 数据集划分 Dataset
+ 数据集预处理 Tokenizer + Dataset
+ 构建模型 Model
+ 设置评估函数 Evaluate
+ 设置训练参数 TrainingArguments
+ 创建训练器 trainer + Data Collator
+ 评估模型, 预测数据集 Trainer
+ 模型预测, 单条 Pipe

## 显存优化

### 占用分析

+ 模型的权重

4Bytes * 模型的参数量

> 大模型权重是指模型中每个神经元连接的参数。这些权重在训练过程中不断调整，以使模型能够更准确地预测输出。简单来说，权重决定了输入数据如何通过模型被处理和转换。

+ 优化器状态optimizer state

adam的momentum + variance 占 2*4Byte 冲量和方差

混合精度的实现中，需要复制一份fp32的参数作为被optimizer更新的参数， 1*4Byte

> 这些数据包括模型每个参数的梯度、学习率、动量等信息，通过这些数据，优化器能够帮助模型逐步收敛到最优解。
>
> 1. 梯度更新：优化器能够根据每个参数的梯度来更新参数值，通过不断迭代优化参数，使模型逐渐收敛到最优解。
> 2. 学习率调整：优化器可以根据模型当前的性能表现动态调整学习率，使模型在训练过程中更加稳定和高效。
> 3. 动量计算：优化器还可以通过记录的动量信息，帮助模型在更新参数时更好地跳出局部最优解，避免陷入局部最优解。
> 4. 过拟合避免：优化器还可以通过记录的数据帮助模型避免过拟合，通过正则化等技术来保持模型的泛化能力。

+ 梯度

4 Byte * 模型参数量

> 梯度数据是指每个参数对应的梯度值，表示了目标函数在当前参数值处的变化率。

+ 向前激活量

序列长度, 隐层维度, Batch大小等

> 从输入层开始向前传播的激活值。在深度神经网络中，每个神经元都有一个激活函数，激活函数对输入的加权和进行非线性变换并产生输出，这个输出就是激活值。
>
> 在模型的向前传播过程中，输入数据经过多层神经元进行加权和激活操作，逐层传递并产生新的激活值，最终在输出层产生预测或分类结果。这一系列激活值相互传递，并在各层之间反映了模型对数据的理解和提取的特征。
>
> 隐层（或隐藏层）维度是指神经网络中的中间层的维度大小。隐层是介于输入层和输出层之间的一层或多层神经元层，负责对输入数据进行特征提取和表示学习。隐层的维度决定了神经网络可以学习到的特征的复杂度和丰富性。

### 实际优化

使用模型是`hfl/chinese-macbert-large`

使用的参数为bach=32, maxlength=128

+ 开始前

![image-20241011180016774](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410111800477.png)

![image-20241011180107925](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410111801981.png)

实际的训练非常的慢预估一个多小时![image-20241011182919692](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410111829757.png)

+ 使用bach size为1, 但是gradient accumulation为32, 也就是每计算32次更新一下参数, 优化向前激活值

```python
gradient_accumulation_steps=32,
```

使用这一个的时候使用的内存下降, 但是训练使用的时间大大加长(我的电脑的显存还是全部占用), 下降了一半

GPU一般比较擅长并行的计算, 所以这一个没有充分利用

+ 在计算的时候有很多中间的结果, 可以不记录, 使用的时候再次计算Gradient Checkpoints

```python
gradient_checkpointing=True
```

+ 使用内存占用比较小的优化器, 默认使用的是adamw_torch这一个优化器

```python
optim="adafactor"
```

GPU内存使用的再次下降, 时间加长

+ 在训练的时候只训练一部分的参数, 比如只训练全连接层, 不训练bert

> 在分类任务中，全连接层通常作为网络的最后一层，直接将全连接层的维度设为类别数量或通过Softmax函数输出每个类别的概率分布，从而实现对输入数据的分类。如果说卷积层、池化层和激活函数等操作是将原始数据映射到隐层特征空间的话，全连接层则起到将学到的“分布式特征表示”映射到样本标记空间的作用。
>
> [读懂BERT，看这一篇就够了 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/403495863)

```python
for name, param in model.bert.named_parameters():
    param.requires_grad = False
```

## 命名实体识别

NER Named Entity(实体) Recognition用于识别文本里面的特定意义的实体, 人名地名, 机构名, 专有名词等, 主要是包括两部分(1)实体边界识别(2)确定实体识别(人名, 地名, 机构名)

> *ModelForTokenClassification

![image-20241012095849325](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410120958515.png)

![image-20241012100051830](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410121000892.png)

![image-20241012100903586](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410121009651.png)

### 模型结构

![image-20241012101417218](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410121014290.png)

![image-20241012102025517](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410121020584.png)

![image-20241012102319901](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410121023968.png)

### 实际训练

使用的数据集是`peoples_daily_ner`, 使用的模型是`hfl/chinese-macbert-base`

+ 导包

```python
import evaluate 
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
```

+ 获取数据集

```python
ner_dataset = load_dataset("peoples_daily_ner", trust_remote_code=True)
ner_dataset["train"][0]
```

> ```python
> {'id': '0',
>  'tokens':['海','钓','比','赛','地','点','在','厦','门','与',
>            '金','门','之','间','的','海','域','。'],
>  'ner_tags': [0, 0, 0, 0, 0, 0, 0, 5, 6, 0, 5, 6, 0, 0, 0, 0, 0, 0]}
> ```
>
> ```python
> ner_dataset["train"].features
> """
> {'id': Value(dtype='string', id=None),
>  'tokens': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None),
>  'ner_tags': Sequence(feature=ClassLabel(names=['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC'], id=None), length=-1, id=None)}
>  PER: 人名 ORG: 组织名 LOC: 地名
> """
> ```

```python
label_list = ner_dataset["train"].features["ner_tags"].feature.names
```

+ 处理数据集

```python
tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-macbert-base")
# 在处理的时候由于已经分词, 所以需要使用这一个参数
tokenizer(ner_dataset["train"][0]["tokens"], is_split_into_words=True) 
```

> 但是这一个拆分可能不是分词器使用的拆分方式, 可以使用获取信息word_ids区分哪几个Token_ids是一个词的
>
> ![image-20241012110603090](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410121106150.png)

```python
def process_function(examples):
    tokenized_examples = tokenizer(examples["tokens"], max_length=128, truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        # 获取一个标签
        word_ids = tokenized_examples.word_ids(batch_index=i)
        label_ids = []
        for word_idx in word_ids:
            # 获取当前 token 是否是一个 word 的第一个 token
            if word_idx is None:
                label_ids.append(-100) # -100 代表不是一个 token
            else:
                label_ids.append(label[word_idx])
        labels.append(label_ids)
    tokenized_examples["labels"] = labels
    return tokenized_examples

tokenized_datasets = ner_dataset.map(process_function, batched=True, remove_columns=ner_dataset["train"].column_names)
```

> ```python
> tokenized_datasets["train"][0]
> ```
>
> ```python
> {'input_ids': [101,3862,7157,3683,
>   ...,
>   1818,511,102],
>  'token_type_ids': [0,0,0,
>   ...,
>   0,0],
>  'attention_mask': [1,1,
>   ...,
>   1],
>  'labels': [-100, 0, 0, 0, 0, 0, 0, 0, 5, 6, 0, 5, 6, 0, 0, 0, 0, 0, 0, -100]}
> ```

+ 模型

```pyrhon
model = AutoModelForTokenClassification.from_pretrained("hfl/chinese-macbert-base")
```

+ 评估函数

```python
seqeval = evaluate.load("seqeval")
```

![image-20241012113140256](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410121131311.png)

![image-20241012113533547](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410121135608.png)

```python
import numpy as np
def eval_metric(pred):
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=-1) # 获取实际的预测值

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100] # 对应每一个数据
        for prediction, label in zip(predictions, labels) # 遍历一个batch
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(true_predictions, true_labels, mode="strict", scheme="BIO2")

    return {
        "f1": results["overall_f1"]
    }
```

+ 参数

```python
args = TrainingArguments(
    output_dir="model_for_ner",
    per_device_eval_batch_size=64,
    per_device_train_batch_size=128,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    metric_for_best_model="f1",
    load_best_model_at_end=True,
    logging_steps=50
)
```

```python
trainer.train()
trainer.evaluate(eval_dataset=tokenized_datasets["test"])
```

+ 测试

```python
from transformers import pipeline
model.config.id2label = {idx: label for idx, label in enumerate(label_list)} # 修改一下映射
"""默认的映射是
  "id2label": {
    "0": "LABEL_0",
    "1": "LABEL_1",
    "2": "LABEL_2",
    "3": "LABEL_3",
    "4": "LABEL_4",
    "5": "LABEL_5",
    "6": "LABEL_6"
  },
"""
ner_pipe = pipeline("token-classification", 
                    model=model, tokenizer=tokenizer, 
                    device=0, 
                    aggregation_strategy="simple") # 使用聚合的模式输出

"""
[{'entity_group': 'PER',
  'score': 0.99990124,
  'word': '史 凯 歌',
  'start': 0,
  'end': 3},
 {'entity_group': 'LOC',
  'score': 0.9998995,
  'word': '北 京',
  'start': 4,
  'end': 6}]
"""
```

> [Pipelines (huggingface.co)](https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.TokenClassificationPipeline)
>
> - “none” : Will simply not do any aggregation and simply return raw results from the model
> - “simple” : Will attempt to group entities following the default schema. (A, B-TAG), (B, I-TAG), (C, I-TAG), (D, B-TAG2) (E, B-TAG2) will end up being [{“word”: ABC, “entity”: “TAG”}, {“word”: “D”, “entity”: “TAG2”}, {“word”: “E”, “entity”: “TAG2”}] Notice that two consecutive B tags will end up as different entities. On word based languages, we might end up splitting words undesirably : Imagine Microsoft being tagged as [{“word”: “Micro”, “entity”: “ENTERPRISE”}, {“word”: “soft”, “entity”: “NAME”}]. Look for FIRST, MAX, AVERAGE for ways to mitigate that and disambiguate words (on languages that support that meaning, which is basically tokens separated by a space). These mitigations will only work on real words, “New york” might still be tagged with two different entities.
> - “first” : (works only on word based models) Will use the `SIMPLE` strategy except that words, cannot end up with different tags. Words will simply use the tag of the first token of the word when there is ambiguity.
> - “average” : (works only on word based models) Will use the `SIMPLE` strategy except that words, cannot end up with different tags. scores will be averaged first across tokens, and then the maximum label is applied.
> - “max” : (works only on word based models) Will use the `SIMPLE` strategy except that words, cannot end up with different tags. Word entity will simply be the token with the maximum score.

+ 去空格

```python
x = "史凯歌在北京吃屎"
res = ner_pipe(x)
ner_result = {}
for r in res:
    if r["entity_group"] not in ner_result:
        ner_result[r["entity_group"]] = []
    ner_result[r["entity_group"]].append(x[r["start"]:r["end"]])
ner_result
```

## 机器阅读理解

Machine Reading Comprehension 简称MRC, 是一个让机器回答给定的上下文的来测试机器阅读理解自然语言的程度的任务

它的形式是比较多样化的, 常见的有完形填空, 答案选择, 片段抽取, 自由生成

![image-20241012193301378](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410121933591.png)

![image-20241012193555638](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410121935736.png)

### 数据预处理

![image-20241012194222195](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410121942295.png)

### 原理

![image-20241012194951132](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410121949249.png)

[#彻底理解# pytorch 中的 squeeze() 和 unsqueeze()函数 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/368920094)

> 在PyTorch中，torch.squeeze函数的参数为-1时表示移除维度为1的维度。具体来说，当使用torch.squeeze(-1)时，会将张量的最后一个维度为1的维度移除，使得张量的维度减少1。
>
> 例如，如果有一个维度为（2，1，3）的张量，使用torch.squeeze(-1)后会得到一个维度为（2，3）的张量。
>
> 总的来说，torch.squeeze(-1)的作用就是移除张量中维度为1的维度。

![image-20241013103939723](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410131039811.png)

限幅。将input的值限制在[min, max]之间，并返回结果。out ([Tensor](https://so.csdn.net/so/search?q=Tensor&spm=1001.2101.3001.7020), optional) – 输出张量，一般用不到该参数。

![image-20241013105032554](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410131050644.png)

### 实际使用

+ 导包

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer, TrainingArguments, DefaultDataCollator
```

+ 获取数据

```python
dataset = load_dataset("cmrc2018", trust_remote_code=True)
```



+ 数据处理

首先从answer里面获取数据的起始位置, 之后通过数据的长度计算结束的位置, 这个时候获取的数据是在字符里面的位置, 需要进行转换获取他在token里面的数据, 可以使用offset_mapping(记录每一个token的起始以及结束的char的位置)进行转换, 把得到的结果判断一下是不是在截取的数据里面, 如果是的话获取一下对应的token, 记录在返回值里面

```python
tokenizer = AutoTokenizer.from_pretrained("E:/JHY/python/2024-10-5-transforms/hlfrbt3")
```

> 测试:
>
> ```python
> sample_dataset = dataset["train"].select([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
> tokenized_example = tokenizer(text=sample_dataset["question"], text_pair=sample_dataset["context"]) # 使用分词器处理, 同时进行分类
> print(tokenized_example["input_ids"][0])
> print(len(tokenized_example["input_ids"][0]))
> 
> """
> [101, 5745, 2455, 7563,...]
> 767
> """
> 
> print(list(zip(tokenized_example["input_ids"][0], tokenized_example["token_type_ids"][0])))
> 
> """
> [(101, 0), (5745, 0), (2455, 0), (7563, 0), ...(5745, 1), (2455, 1), (7563, 1), ...]
> """
> ```
>
> 可以吧问题以及数据合并在一起
>
> ```python
> # 合并问题和文本，最大长度384，截断文本，填充到最大长度，返回offsets_mapping(用于后续处理答案的位置, 可以对应Token的字符offset位置)
> tokenized_example = tokenizer(text=sample_dataset["question"],
>                               text_pair=sample_dataset["context"], 
>                               max_length=384, truncation="only_second",
>                               padding="max_length", 
>                               return_offsets_mapping=True) 
> ```

+ 实际获取目标的结果的起始以及结束token

```python
for idx, offsets in enumerate(offset_mapping):
    # 'answers': {'text': ['1963年'], 'answer_start': [30]}}
    answer = sample_dataset[idx]["answers"] # 获取答案
    start_char = answer["answer_start"][0] # 真实答案在字符里面的位置
    end_char = start_char + len(answer["text"][0])
    print(answer, start_char, end_char)
    # 之后定位答案在token中的位置
    # 获取context的起始结束, 之后根据答案的起始结束位置，找到对应的token位置
    context_start = tokenized_example.sequence_ids(idx).index(1)
    context_end = tokenized_example.sequence_ids(idx).index(None, context_start) -1
    print(context_start, context_end)

    # 评断文本的起始位置结束(字符)是否在context中, 使用offset进行token到char的转换
    if offsets[context_end][1] <= start_char or offsets[context_start][0] >= end_char:
        # print("答案不在context中")
        start_token_pos = 0
        end_token_pos = 0
    else:
        token_id = context_start
        while token_id <= context_end and offsets[token_id][0] < start_char:
            token_id += 1
        start_token_pos = token_id
        token_id = context_end
        while token_id >= context_start and offsets[token_id][1] > end_char:
            token_id -= 1
        end_token_pos = token_id

    print(start_token_pos, end_token_pos)
    print("token answer decode: ", tokenizer.decode(tokenized_example["input_ids"][idx][start_token_pos:end_token_pos+1]))
```

> 进行封装可以获得

```python
def process_func(examples):
    # 合并问题和文本，最大长度384，截断文本，填充到最大长度，返回offsets_mapping(用于后续处理答案的位置, 可以对应Token的字符offset位置)
    tokenized_examples = tokenizer(examples["question"], examples["context"], 
                                   max_length=512, truncation="only_second",
                                   padding="max_length", 
                                   return_offsets_mapping=True)
    
    # 保存答案的token位置
    offset_mapping = tokenized_examples.pop("offset_mapping")
    # 'answers': {'text': ['1963年'], 'answer_start': [30]}}
    start_positions = []
    end_positions = []
    for idx, offsets in enumerate(offset_mapping):    
        answer = examples["answers"][idx] # 获取答案
        start_char = answer["answer_start"][0] # 真实答案在字符里面的位置
        end_char = start_char + len(answer["text"][0])
        # 之后定位答案在token中的位置
        # 获取context的起始结束, 之后根据答案的起始结束位置，找到对应的token位置
        context_start = tokenized_examples.sequence_ids(idx).index(1)
        context_end = tokenized_examples.sequence_ids(idx).index(None, context_start) -1

        # 评断文本的起始位置结束(字符)是否在context中, 使用offset进行token到char的转换
        if offsets[context_end][1] <= start_char or offsets[context_start][0] >= end_char:
            # print("答案不在context中")
            start_token_pos = 0
            end_token_pos = 0
        else:
            token_id = context_start
            while token_id <= context_end and offsets[token_id][0] < start_char:
                token_id += 1
            start_token_pos = token_id
            token_id = context_end
            while token_id >= context_start and offsets[token_id][1] > end_char:
                token_id -= 1
            end_token_pos = token_id
        start_positions.append(start_token_pos)
        end_positions.append(end_token_pos)
    # 保存答案的token位置
    tokenized_examples["start_positions"] = start_positions
    tokenized_examples["end_positions"] = end_positions
    return tokenized_examples

tokenized_dataset = dataset.map(process_func, batched=True, remove_columns=dataset["train"].column_names)
```

+ 训练

```python
model = AutoModelForQuestionAnswering.from_pretrained("E:/JHY/python/2024-10-5-transforms/hlfrbt3")
args = TrainingArguments(
    output_dir="model_for_qa",
    per_device_eval_batch_size=32,
    per_device_train_batch_size=32,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_steps=50
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=DefaultDataCollator(),
    tokenizer=tokenizer
)
trainer.train()
```

+ 预测

```python
# 模型预测
from transformers import pipeline

pipe = pipeline("question-answering", model=model, tokenizer=tokenizer, device=0)
pipe({
    "question": "什么时候成立的",
    "context": "中华人民共和国成立于1949年10月1日。"
})

```

### 滑动窗口

用于处理数据比较长的时候, 如果只是简单的截断, 会出现答案被截断的问题, 所以在截断的时候一般会有一部分的重叠, 重叠的长短会使得使用这一种的时候会导致数据的数量增大, 重叠部分比较小的时候会出现长下文丢失以及答案不完整

最后获取到多个数据的预测结果进行聚合

#### 实际实现

使用一个nltk库

> nltk库是自然语言处理领域最为知名且广泛使用的Python库之一，其功能包括文本分析、标注、分词、句法分析、语义分析、语料库管理等。nltk库还提供了丰富的语言处理工具和资源，可以帮助用户进行文本挖掘、信息检索、文本分类、语言模型等任务。通过nltk库，用户可以轻松地处理文本数据，进行文本分析和挖掘，从而实现自然语言处理相关的各种应用和研究。

```bash
pip install nltk
```

```python
import nltk
nltk.download("punkt")
```

+ tokenizer处理函数改变

```python
sample_dataset = dataset["train"].select([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# 合并问题和文本，最大长度384，截断文本，填充到最大长度，返回offsets_mapping(用于后续处理答案的位置, 可以对应Token的字符offset位置)
tokenized_example = tokenizer(text=sample_dataset["question"],
                              text_pair=sample_dataset["context"], 
                              max_length=384,
                              truncation="only_second", 
                              padding="max_length", 
                              return_offsets_mapping=True,
                              return_overflowing_tokens=True) 
tokenized_example.keys()
```

> 加入参数return_overflowing_tokens, 默认的时候是没有进行重叠操作的, 可以使用stride参数指定
>
> ![image-20241015191930969](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410151919615.png)
>
> ![image-20241015192230343](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410151922391.png)

+ 与处理函数, 对数据进行处理, 获取需要的数据

```python
def process_func(examples):
    # 合并问题和文本，最大长度384，截断文本，填充到最大长度，返回
    # offsets_mapping(用于后续处理答案的位置, 
    # 可以对应Token的字符offset位置)
    tokenized_examples = tokenizer(examples["question"],
                                   examples["context"], 
                                   max_length=384, 
                                   truncation="only_second", 
                                   padding="max_length", 
                                   return_offsets_mapping=True,
                                   return_overflowing_tokens=True, 
                                   stride=128)
    sample_mapping = tokenized_examples.get("overflow_to_sample_mapping")
    # 保存答案的token位置
    # offset_mapping = tokenized_examples.pop("offset_mapping")
    # 'answers': {'text': ['1963年'], 'answer_start': [30]}}
    start_positions = []
    end_positions = []
    example_ids = []
    for idx,_ in enumerate(sample_mapping):
        # 获取答案, 一个答案可能对应多个数据
        answer = examples["answers"][sample_mapping[idx]] 
        # 真实答案在字符里面的位置
        start_char = answer["answer_start"][0] 
        end_char = start_char + len(answer["text"][0])
        # 之后定位答案在token中的位置
        # 获取context的起始结束, 之后根据答案的起始结束位置
        # 找到对应的token位置, 使用index函数获取第一个1的位置
        context_start = tokenized_examples.sequence_ids(idx).index(1)
        context_end = tokenized_examples.sequence_ids(idx).index(None, context_start) - 1
        offsets = tokenized_examples.get("offset_mapping")[idx]
        # 评断文本的起始位置结束(字符)是否在context中, 使用offset进行token到char的转换
        if offsets[context_end][1] <= start_char or offsets[context_start][0] >= end_char:
            # print("答案不在context中")
            start_token_pos = 0
            end_token_pos = 0
        else:
            token_id = context_start
            while token_id <= context_end and offsets[token_id][0] < start_char:
                # 使用遍历的方法获取一下数据的起始位置
                token_id += 1
            start_token_pos = token_id
            token_id = context_end
            while token_id >= context_start and offsets[token_id][1] > end_char:
                # 反向遍历获取结束位置
                token_id -= 1
            end_token_pos = token_id
        start_positions.append(start_token_pos)
        end_positions.append(end_token_pos)
        # 记录每一个数据的id用于对应
        example_ids.append(examples["id"][sample_mapping[idx]])
        tokenized_examples["offset_mapping"][idx] = [
            # 记录一下有效数据的token对应的offset(非问题数据的位置)
            (o if tokenized_examples.sequence_ids(idx)[k] == 1 else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][idx])
        ]

    tokenized_examples["example_ids"] = example_ids
    # 保存答案的token位置
    tokenized_examples["start_positions"] = start_positions
    tokenized_examples["end_positions"] = end_positions
    return tokenized_examples
```

```python
tokenized_dataset = dataset.map(process_func, batched=True, remove_columns=dataset["train"].column_names)
```

+ 获取数据的预测以及真实的数据

```python
import numpy as np
import collections

def get_result(start_logits, end_logits, examples, features):
    """_summary_

    Args:
        start_logits (_type_): 模型预测的结果起始位置
        end_logits (_type_): 结束位置的预测结果
        examples (_type_): 原始的数据集
        features (_type_): tokenizer获取到的mapping
    """
    predictions = {}
    references = {}

    example_to_features = collections.defaultdict(list) # 保存每一个example对应的feature编号
    for idx, example_id in enumerate(features["example_ids"]):
        example_to_features[example_id].append(idx) # 记录一下每一个example对应的被分割以后的编号

    # 最优答案候选数
    n_best = 20
    max_answer_length = 30
    for example in examples:
        example_id = example["id"]
        context = example["context"]
        answers = []
        for feature_idx in example_to_features[example_id]:
             # 获取对应的feature的预测结果, 这个结果搓是一个数组
            start_logit = start_logits[feature_idx]
            end_logit = end_logits[feature_idx]
            # 获取对应的offset_mapping
            offset = features[feature_idx]["offset_mapping"] 
            # 从大到小排序，取前n_best
            start_indexs = np.argsort(start_logit)[::-1][:n_best].tolist() 
            # 从大到小排序，取前n_best
            end_indexs = np.argsort(end_logit)[::-1][:n_best].tolist() 
            for start_index in start_indexs:
                for end_index in end_indexs:
                    # 如果预测的位置不在offset中，或者结束位置在开始位置之前，或者长度超过最大长度，都不要
                    if offset[start_index] is None or offset[end_index] is None:
                        continue
                    if start_index > end_index or end_index - start_index + 1 > max_answer_length:
                        continue
                    answers.append({
                        "score": start_logit[start_index] + end_logit[end_index],
                        "text": context[offset[start_index][0]:offset[end_index][1]]
                    })
        if len(answers) > 0:
            # 获取评分最高的预测结果
            best_answer = max(answers, key=lambda x: x["score"])
            predictions[example_id] = best_answer["text"]
        else:
            predictions[example_id] = ""
        references[example_id] = example["answers"]["text"]

    return predictions, references
```

+ 实际的预测函数

```python
from cmrc_eval import evaluate_cmrc
def metric(pred):
    start_logits, end_logits = pred[0]
    if start_logits.shape[0] == len(tokenized_dataset["validation"]):
        p, r = get_result(start_logits, end_logits, dataset["validation"], tokenized_dataset["validation"])
    else:
        p, r = get_result(start_logits, end_logits, dataset["test"], tokenized_dataset["test"])
    return evaluate_cmrc(p, r)
```

## 多项选择

机器阅读理解里面的一个分支, 给定一个文档, 一个问题以及多个答案, 从里面获取正确的答案

### 数据处理

![image-20241022230218638](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410222302906.png)

![image-20241022230335193](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410222303371.png)

> 在实际处理的时候需要把数据进行一个聚合

![image-20241022230548880](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410222305087.png)

> 这里使用的view函数会按照最里面一层的大小展开为二维数组
>
> ```python
> import torch
> 
> tensor1 = torch.tensor([[[1, 2], [2, 3], [3, 4]], [[5, 6], [6, 7], [7, 8]]])
> 
> print(tensor1.size(-1))
> tensor1 = tensor1.view(-1, tensor1.size(-1))
> print(tensor1.size())
> """
> 2
> torch.Size([6, 2])
> """
> ```

![image-20241022232644703](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410222326874.png)

### 实际训练

这里使用的数据集clue下面的C3数据集

https://huggingface.co/datasets/clue/clue

![image-20241024225949337](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/image/202410242259476.png)

> 这里的context可能是一个对话, 是对话的时候这一个的数据是一个列表,这里面的数据由于test数据集没有answer, 所以需要去除
