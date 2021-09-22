---
title: "Origin Transformer"
date: 2021-09-06T20:04:32+08:00
draft: false

mathjax: true

excerpt_separator: <!--more-->
---
Attention is all your need.<!--more-->

## 基础

### Attention

Attention 定义上是一个映射函数，输入`Q,K,V`等向量，输出是一个新的向量。具体定义如下：

> An attention function can be described as mapping a query and a set of key-value pairs to an output,
where the query, keys, values, and output are all vectors. The output is computed as a weighted sum
of the values, where the weight assigned to each value is computed by a compatibility function of the
query with the corresponding key.

### Scaled Dot-Product Attention

这里 `Dot-Product` 是一种计算Attention权重的方式（另一种常见的方式是 Additive Attention）。输入Q与K都是向量，维度为 $d_k$，输入 V 也是向量，维度为 $d_v$，然后权重计算过程是将 Q 与所有的 K 计算向量点乘（Dot-Product）。

所谓的 `Scaled` 体现在将上述点乘结果除以 $\sqrt{d_k}$。为什么是除以这个数？主要原因是，两个 $d_k$ 维的矩阵乘（矩阵元素是mean 0, variance 1 生成的随机数），结果矩阵的方差就是$d_k$。所以，如果这里不进行 Scale，那么得到的矩阵数值就会越来越大，导致后面的 Softmax 饱和。

![图-1 Scaled Dot-Product Attention示意图](/imgs/origin-transformer/transformer0.png)

补充一下 Additive Attention。具体实现是通过全连接映射然后按元素加得到，公式如下。这里为什么选择 Dot-Product 而不是 Additive Attention 呢？而且两者的理论计算复杂度差不多。论文里也给出了解释，就是Dot-Product在实际计算中其实是更快的，毕竟矩阵乘法被研究、优化的更多。

$$a(q, k) = w_v^T \tanh (W_q q + W_k k) \in \mathbb{R}$$

### Multi-head

作者发现，用不同的Linear Projection 来将 Q, K, V 进行映，然后对应的计算 Attention，最终将结果拼接起来的效果比使用一个单独的 Attention 效果更好。下面的公式与图2就可以很好的说明计算过程了，实际实现可以通过先合并 Linear Projection 的权重，然后在经过 Reshape 完成。

$$\begin{gather*}
\mathrm{MultiHead}(Q, K, V) = \mathrm{Concat}(head_1, \dots, head_n)W^O  \\
where, head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
\end{gather*}$$

![图-2 Multi-Head Attention示意图](/imgs/origin-transformer/transformer1.png)

### Position-wise FFN

实现上来看是两层全连接，并且第一层一般会将输入Tensor的channel个数扩展expansion(=4)倍，然后第二层全连接在恢复原来的 channel 个数。

为什么叫 Position-wise 呢？按照论文的说法，我猜这里的Position是指 Depth 维度上的位置，体现在相同层的不同位置的 Token 公用相同的 Lienar Projection 权重矩阵，但是不同层上使用不同的 Linear Projection。

> While the linear transformations are the same across different positions, they use different parameters from layer to layer

不过 D2L 中李沐的说法是：

> The positionwise feed-forward network transforms the representation at all the sequence positions using the same MLP. This is why we call it positionwise.

## Encoder-Decoder结构

一个方面是如何将 Encoder 的信息传递给 Decoder，有两种做法，一种是指在Decoder的第一个输入位置上使用，另一种是在Decoder的每一次输入上都使用。

### Seq2Seq

这里参考[Sequence to sequence leanring - d2l](https://d2l.ai/chapter_recurrent-modern/seq2seq.html)中的讲解。

具体的 Encoder - Decoder 部分这里基于 GRU 来实现。输入尺寸为：$(batch_size, num_steps, embed_size)$；GRU 的计算包含两个输出，一个是GRU 最后输出结果output，尺寸仍然是: $(num_steps, batch_size, embed_size)$，相当于每一步（共num_steps，可认为是 num_steps 个 Toke）都输出了一个新的 embed_size 大小的向量；另一个输出是隐空间变量 states，尺寸是 $(num_layers, batch_size, num_hiddens)$，相当于是当前输入Token与上一个Token对应的隐变量共同作用生成了当前Token对应的隐变量，这个隐变量就包含了前面所有 Token 的信息。 当前 token 的 output 与隐变量 state 之间的关系是：`output = Mlp(state)`。

```python {linenos=table, linenostart=0}
def rnn(inputs, state, params):
    # Shape of `inputs`: (`num_steps`, `batch_size`, `vocab_size`)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # Shape of `X`: (`batch_size`, `vocab_size`)
    for X in inputs:
        H = np.tanh(np.dot(X, W_xh) + np.dot(H, W_hh) + b_h)
        Y = np.dot(H, W_hq) + b_q
        outputs.append(Y)
    return np.concatenate(outputs, axis=0), (H,)
```

上面说完 Encoder 部分，Decoder 部分结构相同，但重要的是 Decoder 部分的输出应该怎么决定。

输入主要包含两个部分，首先是起始Token，这里起始 Token 是一个特殊字符，`<bos>`；另一个部分就是隐变量的确定，这里使用 Encoder 输出的隐变量作为初始隐变量，注意这里 Encoder - Decoder 需要使用相同的层数，这样隐变量的尺寸才匹配，即：$(num_layers, batch size, embed_size)$。下面给出的示例代码中，还会将 Encoder 输出的最后一层的隐变量与输入 X 拼接起来进行计算。Decoder 的输出就是$(batch size, num steps, vocab size)$。

```python {linenos=table, linenostart=0}
class Seq2SeqDecoder(d2l.Decoder):
    """The RNN decoder for sequence to sequence learning."""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = rnn.GRU(num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Dense(vocab_size, flatten=False)

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]

    def forward(self, X, state):
        # The output `X` shape: (`num_steps`, `batch_size`, `embed_size`)
        X = self.embedding(X).swapaxes(0, 1)
        # `context` shape: (`batch_size`, `num_hiddens`)
        context = state[0][-1]      # 最后一层对应的隐变量
        # Broadcast `context` so it has the same `num_steps` as `X`
        context = np.broadcast_to(
            context, (X.shape[0], context.shape[0], context.shape[1]))
        X_and_context = np.concatenate((X, context), 2)
        output, state = self.rnn(X_and_context, state)
        output = self.dense(output).swapaxes(0, 1)
        # `output` shape: (`batch_size`, `num_steps`, `vocab_size`)
        # `state[0]` shape: (`num_layers`, `batch_size`, `num_hiddens`)
        return output, state
```

### Transformer 中的实现

Encoder 部分简单的是 Transformer 层的堆叠。Transformer 层包含两个sublayer，分别是 MultiHead Self Attention 以及 Positionwise FFN，这两个 sublayer 都会通过残差连接并紧跟着计算一个 LayerNorm （这里不讨论pre-norm的实现）。

Decoder 部分相比于 Encoder 的两个 sublayer 构成，多了一个 cross-attention 的层。cross-attention的主要区别在于输入的 K, V 来自于对应的 Encoder 层，Query 来自于 Decoder 中上一层的 MultiHead Self Attention的输出。整体结构如下。

![图-3 Transformer Encoder-Decoder 示意图](/imgs/origin-transformer/transformer2.png)

下面给出了MXNet实现代码，非常详细，但是解答了下面几个疑问。

* Decoder 最开始的输入是`<bos>`，在训练时，这个也是拼接在最前面的
* Decoder 中每一层中 Cross Attention 的 K, V 都是相同的，都来自于 Encoder 的最后输出
* 在最大 num_steps 限制下，最后一个元素是 `<eos>` 时则退出 Decoder 部分

```python {linenos=table, linenostart=0}
class EncoderDecoder(nn.Block):
    """The base class for the encoder-decoder architecture."""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)

class EncoderBlock(nn.Block):
    """Transformer encoder block."""
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout,
                 use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = d2l.MultiHeadAttention(num_hiddens, num_heads,
                                                dropout, use_bias)
        self.addnorm1 = AddNorm(dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))

class TransformerEncoder(d2l.Encoder):
    """Transformer encoder."""
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                 num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for _ in range(num_layers):
            self.blks.add(
                EncoderBlock(num_hiddens, ffn_num_hiddens, num_heads, dropout,
                             use_bias))

    def forward(self, X, valid_lens, *args):
        # Since positional encoding values are between -1 and 1, the embedding
        # values are multiplied by the square root of the embedding dimension
        # to rescale before they are summed up
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[
                i] = blk.attention.attention.attention_weights
        return X

class DecoderBlock(nn.Block):
    # The `i`-th block in the decoder
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout, i,
                 **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        self.attention1 = d2l.MultiHeadAttention(num_hiddens, num_heads,
                                                 dropout)
        self.addnorm1 = AddNorm(dropout)
        self.attention2 = d2l.MultiHeadAttention(num_hiddens, num_heads,
                                                 dropout)
        self.addnorm2 = AddNorm(dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        # During training, all the tokens of any output sequence are processed
        # at the same time, so `state[2][self.i]` is `None` as initialized.
        # When decoding any output sequence token by token during prediction,
        # `state[2][self.i]` contains representations of the decoded output at
        # the `i`-th block up to the current time step
        if state[2][self.i] is None:
            key_values = X
        else:
            # 这里是只使用已预测的Token进行计算
            key_values = np.concatenate((state[2][self.i], X), axis=1)     
        state[2][self.i] = key_values

        if autograd.is_training():
            batch_size, num_steps, _ = X.shape
            # Shape of `dec_valid_lens`: (`batch_size`, `num_steps`), where
            # every row is [1, 2, ..., `num_steps`]
            dec_valid_lens = np.tile(np.arange(1, num_steps + 1, ctx=X.ctx),
                                     (batch_size, 1))
        else:
            dec_valid_lens = None

        # Self-attention
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        # Encoder-decoder attention. Shape of `enc_outputs`:
        # (`batch_size`, `num_steps`, `num_hiddens`)
        ## 这里使用 Encoder 最后的输出的 enc_outputs 当作 K, V 进行计算！！
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state

class TransformerDecoder(d2l.AttentionDecoder):
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                 num_layers, dropout, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add(
                DecoderBlock(num_hiddens, ffn_num_hiddens, num_heads, dropout,
                             i))
        self.dense = nn.Dense(vocab_size, flatten=False)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range(2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            # Decoder self-attention weights
            self._attention_weights[0][
                i] = blk.attention1.attention.attention_weights
            # Encoder-decoder attention weights
            self._attention_weights[1][
                i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights

num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10
lr, num_epochs, device = 0.005, 200, d2l.try_gpu()
ffn_num_hiddens, num_heads = 64, 4

train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)

encoder = TransformerEncoder(len(src_vocab), num_hiddens, ffn_num_hiddens,
                             num_heads, num_layers, dropout)
decoder = TransformerDecoder(len(tgt_vocab), num_hiddens, ffn_num_hiddens,
                             num_heads, num_layers, dropout)
net = d2l.EncoderDecoder(encoder, decoder)

d2l.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
    translation, dec_attention_weight_seq = d2l.predict_seq2seq(
        net, eng, src_vocab, tgt_vocab, num_steps, device, True)
    print(f'{eng} => {translation}, ',
          f'bleu {d2l.bleu(translation, fra, k=2):.3f}')

def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=False):
    """Predict for sequence to sequence."""
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [
        src_vocab['<eos>']]
    enc_valid_len = np.array([len(src_tokens)], ctx=device)
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # Add the batch axis
    enc_X = np.expand_dims(np.array(src_tokens, ctx=device), axis=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # Add the batch axis
    ## 最开始的是 '<bos>'
    dec_X = np.expand_dims(np.array([tgt_vocab['<bos>']], ctx=device), axis=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        # We use the token with the highest prediction likelihood as the input
        # of the decoder at the next time step
        dec_X = Y.argmax(axis=2)
        pred = dec_X.squeeze(axis=0).astype('int32').item()
        # Save attention weights (to be covered later)
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # Once the end-of-sequence token is predicted, the generation of the
        # output sequence is complete
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq

def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    """Train a model for sequence to sequence."""
    net.initialize(init.Xavier(), force_reinit=True, ctx=device)
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': lr})
    loss = MaskedSoftmaxCELoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[10, num_epochs])
    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)  # Sum of training loss, no. of tokens
        for batch in data_iter:
            X, X_valid_len, Y, Y_valid_len = [
                x.as_in_ctx(device) for x in batch]
            bos = np.array([tgt_vocab['<bos>']] * Y.shape[0],
                           ctx=device).reshape(-1, 1)
            dec_input = d2l.concat([bos, Y[:, :-1]], 1)  # Teacher forcing
            with autograd.record():
                Y_hat, _ = net(X, dec_input, X_valid_len)
                l = loss(Y_hat, Y, Y_valid_len)
            l.backward()
            d2l.grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            trainer.step(num_tokens)
            metric.add(l.sum(), num_tokens)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '

```
