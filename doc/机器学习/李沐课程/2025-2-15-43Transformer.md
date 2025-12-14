# Transformer

实际是一个Sequence-to-sequence, 输入和输出都是一个sequence, 输出的长度是不一定的, 使用纯注意力机制

如果把一个树状的序列使用多个括号进行分层甚至可以使用用这种方式进行树状的处理

Transformer实际使用的还是encoder和decoder两个的架构

## encoder

![image-20250215162322779](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/picture/202502151623907.png)

transformer里面Attention以后还记录有原数据, 之后进行一下norm(Layer Norm计算平均值和标准差之后标准化)

![image-20250215162748331](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/picture/202502151627428.png)

![image-20250215163015346](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/picture/202502151630389.png)

前馈全连接层（feed-forward linear layer）基本上就是一堆神经元，每个神经元都与其他神经元相连接。但是输入是(b, n, d)的一个网络, 输入时候需要变化为(bn, d)然后使用两个全连接层, 输出的形状再变回(b, n, d), 这么做是为了卷积可以处理

![image-20250215163758449](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/picture/202502151637562.png)

前馈全连接层的输入数据只向前传播，没有反向传播。每个神经元的输出是前一层所有神经元的输入经过权重加权求和后再加上偏置项，然后经过激活函数得到的结果。

前馈全连接层和全连接层的主要区别在于前者是一种特殊的全连接结构，每个神经元的输入只来自上一层的神经元，不包括网络内部的反馈连接。而全连接层是一种通用的连接结构，每个神经元都与上一层的所有神经元有连接，可以包括反馈连接。

```python
#@save
class PositionWiseFFN(nn.Module):
    """基于位置的前馈网络"""
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
        **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)
    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))
    
ffn = PositionWiseFFN(4, 4, 8)
ffn.eval()
"""
PositionWiseFFN(
  (dense1): Linear(in_features=4, out_features=4, bias=True)
  (relu): ReLU()
  (dense2): Linear(in_features=4, out_features=8, bias=True)
)
"""
# 输入是4, 输出是8, 实际是对输入的每个元素应用了一个全连接网络
ffn(torch.ones((2, 3, 4))).shape
"""
torch.Size([2, 3, 8])
"""
```



归一化层是对每一个特征做一个归一化, 因为随着输入的变化, 这里的大小会变化

```python
ln = nn.LayerNorm(2)
bn = nn.BatchNorm1d(2)
X = torch.tensor([[1, 2], [2, 3]], dtype=torch.float32)    
# 在训练模式下计算X的均值和方差
print('layer norm:', ln(X), '\nbatch norm:', bn(X))
"""
layer norm: tensor([[-1.0000,  1.0000],
        [-1.0000,  1.0000]], grad_fn=<NativeLayerNormBackward0>) 
batch norm: tensor([[-1.0000, -1.0000],
        [ 1.0000,  1.0000]], grad_fn=<NativeBatchNormBackward0>)
"""
```

```python
#@save
class AddNorm(nn.Module):
    """残差连接后进行层规范化"""
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)
    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)
    
add_norm = AddNorm([3, 4], 0.5)
add_norm.eval()
add_norm(torch.ones((2, 3, 4)), torch.ones((2, 3, 4))).shape

"""
torch.Size([2, 3, 4])
"""
```

### 实现

```python
#@save
class EncoderBlock(nn.Module):
    """Transformer编码器块"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
            norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
            dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout,
            use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(
            ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        # 使用三个X执行自注意力
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))
    
# batch_size = 2, seq_len = 100, embed_size = 24
# transformer实际不会改变输入的形状
X = torch.ones((2, 100, 24))
valid_lens = torch.tensor([3, 2])
encoder_blk = EncoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5)
encoder_blk.eval()
encoder_blk(X, valid_lens).shape
"""
torch.Size([2, 100, 24])
"""

#@save
class TransformerEncoder(d2l.Encoder):
    """Transformer编码器"""
    def __init__(self, vocab_size, key_size, query_size, value_size,
    num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
    num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
            EncoderBlock(key_size, query_size, value_size, num_hiddens,
                norm_shape, ffn_num_input, ffn_num_hiddens,
                num_heads, dropout, use_bias))

    def forward(self, X, valid_lens, *args):
        # 因为位置编码值在-1和1之间，
        # 因此嵌入值乘以嵌入维度的平方根进行缩放，
        # 然后再与位置编码相加。
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X
    
encoder = TransformerEncoder(
 200, 24, 24, 24, 24, [100, 24], 24, 48, 8, 2, 0.5)
encoder.eval()
encoder(torch.ones((2, 100), dtype=torch.long), valid_lens).shape

"""
torch.Size([2, 100, 24])
"""
```



## Decoder

比较经常使用的是Autoregressive, 实际的工作是收取encoder的输入以后, 提供给他一个开始的提示, 让他进行输出, 并把每一次的输出作为下一次的提示词

![image-20250215165756411](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/picture/202502151657467.png)

这里使用的Masked Multi-Head Attention每一个数据在计算关系的时候只可以使用之前的数据

![image-20250215165937601](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/picture/202502151659722.png)

![image-20250215170116909](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/picture/202502151701966.png)

Non-autoregressive(NAT)

![image-20250215171659559](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/picture/202502151716642.png)

可以使用encoder的输出使用另一个网络计算一下使用的BEGIN的数量, 也可以直接给他很多的BEGIN, 找出来第一个结束符号, 这个方式的使用需要和Self-Attention进行配合

> 速度更快, 但是和AT的比实际的效果比较差

+ 两个部分传递信息

![image-20250215172221189](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/picture/202502151722253.png)

![image-20250215172442107](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/picture/202502151724196.png)

> 原始的paper里面使用的都是最后一次的输出, 但是实际可以使用多种不同的连接方式

```python
class DecoderBlock(nn.Module):
    """解码器中第i个块, 用于解码序列"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
    norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
        dropout, i, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        self.attention1 = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = MultiHeadAttention(
        key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens,
        num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)
    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        # 训练阶段，输出序列的所有词元都在同一时间处理，
        # 因此state[2][self.i]初始化为None。
        # 预测阶段，输出序列是通过词元一个接着一个解码的，
        # 因此state[2][self.i]包含着直到当前时间步第i个块解码的输出表示
        if state[2][self.i] is None:
            key_values = X # 这是在训练阶段, 直接使用X
        else:
            # 在预测阶段，将state[2][self.i]的形状变换为
            # (batch_size, i, num_hiddens)，并将其拼接在一起，
            key_values = torch.cat((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values
        if self.training:
            batch_size, num_steps, _ = X.shape
            # dec_valid_lens的开头:(batch_size,num_steps),
            # 其中每一行是[1,2,...,num_steps]
            dec_valid_lens = torch.arange(
            1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None
        # 自注意力
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        # 编码器－解码器注意力。
        # enc_outputs的开头:(batch_size,num_steps,num_hiddens)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state
    
decoder_blk = DecoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5, 0)
decoder_blk.eval()
X = torch.ones((2, 100, 24))
state = [encoder_blk(X, valid_lens), valid_lens, [None]]
decoder_blk(X, state)[0].shape

"""
torch.Size([2, 100, 24])
"""

class TransformerDecoder(d2l.AttentionDecoder):
    def __init__(self, vocab_size, key_size, query_size, value_size,
    num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
    num_heads, num_layers, dropout, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                DecoderBlock(key_size, query_size, value_size, num_hiddens,
                norm_shape, ffn_num_input, ffn_num_hiddens,
                num_heads, dropout, i))
        self.dense = nn.Linear(num_hiddens, vocab_size)
    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]
    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range (2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            # 解码器自注意力权重
            self._attention_weights[0][i] = blk.attention1.attention.attention_weights
            # “编码器－解码器”自注意力权重
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights
        return self.dense(X), state
    @property
    def attention_weights(self):
        return self._attention_weights
```



+ 训练

实际训练的时候Decoder实际的输入是正确的答案, Teacher forcing, 获取输出的最小的minimize cross entropy, 但是这样和实际的使用是有区别的

exposure bias, 测试的时候输出有一个错误的输出可能会影响后面的输出, 所以实际的训练输入给一部分错误的数据Scheduled Sampling

```python
num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10
lr, num_epochs, device = 0.005, 200, d2l.try_gpu()
ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
key_size, query_size, value_size = 32, 32, 32
norm_shape = [32]

train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
encoder = TransformerEncoder(
    len(src_vocab), key_size, query_size, value_size, num_hiddens,
    norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
    num_layers, dropout)
decoder = TransformerDecoder(
    len(tgt_vocab), key_size, query_size, value_size, num_hiddens,
    norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
    num_layers, dropout)
net = d2l.EncoderDecoder(encoder, decoder)
d2l.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
```

![image-20250215234119472](https://picture-01-1316374204.cos.ap-beijing.myqcloud.com/picture/202502152341582.png)

测试一下

```python
engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
    translation, dec_attention_weight_seq = d2l.predict_seq2seq(
    net, eng, src_vocab, tgt_vocab, num_steps, device, True)
    print(f'{eng} => {translation}, ',
    f'bleu {d2l.bleu(translation, fra, k=2):.3f}')
"""
go . => va !,  bleu 1.000
i lost . => j'ai perdu .,  bleu 1.000
he's calm . => il est calme .,  bleu 1.000
i'm home . => je suis chez moi .,  bleu 1.000
"""
```

+ Copy Mechanism

实际的使用过程里面如果用户输入一个没有出现过的名词, 需要把这个名词进行复制作为输出(尤其是在做文章的摘要)

+ Guided Attention

有的问题在处理的时候使用的Attention是有一定规律的, 比如进行语音合成, 就是从左到右的一个处理过程, 可以加一个限制, 可以使用Monotonic Attention和Location-aware attention