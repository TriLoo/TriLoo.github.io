---
title: "常见掩码生成方式 2"
date: 2021-10-27T22:08:15+08:00
draft: false
mathjax: true

excerpt_separator: <!--more-->
---
这是接着上一篇掩码生成方式写的，主要仅包含SpanBERT & MacBERT的原理与实现。<!--more-->


## SpanBERT

SpanBERT的官方实现代码比想象中的复杂，并且还没有比较好的说明文档，所以这里单独作为一篇笔记详细分析一下。SpanBERT这篇论文主要的创新点有三个，（1）引入了新的掩膜方式，即随机掩膜掉连续长度的token，（2）引入了掩膜边界预测损失，即Span Boundary Object (SBO)，（3）去掉了BERT中使用的 NSP 任务

### SpanBERT创新点

对于第一点，需要确定连续掩膜的起始位置 + 掩膜长度。这里起始位置是uniform随机选择的；掩膜长度是按照$l \sim \textrm{Geo}(p=0.2)$ 来确定的，其中Geo对应的是[Geometric Distribution](https://en.wikipedia.org/wiki/Geometric_distribution)。

Gemoetric Distribution分布可以从两个角度理解，试验k次，成功的次数，其中p为每次实验成功的概率，或者实验k次，成功前连续有k-1次失败的概率，前者k的取值是$[1, 2, \ldots ]$，后者的取值是$[0, 1, \ldots]$。对应的数学公式如下。

$$\textrm{Geo}(p) = (1-p)^{k-1}p$$

这里 k 也就是掩膜的长度了，也就是对应的掩膜长度对应1、2、3 ... 等的概率了。并且，p越接近于1，概率下降越快，也就更倾向于更短的掩膜；还有一点是，这里的长度值得是 whole word 的个数，而不是wordpiece元素的个数! 与前面WWM的做法不同的是，SpanBERT将位于Span内的要被掩膜的tokens采用同一种掩膜方式，现在掩膜方式有三种：使用Mask、随机Token、保持不变。

SBO 任务实际上是用Span边界位置未被掩膜掉的两个 token 预测被掩膜掉的 token 的分类任务，与 MLM 相互补充，只不过 MLM 用的是所有 token 的信息，SBO 只用了边界位置 token 的信息。比如输入的 token 序列是$x_1, \ldots, x_n$，然后$x_s, x_e$表示掩膜span的边界，则SBO使用$x_{s-1}, x_{e+1}$两个token配合在相对位置$i$上的信息来预测具体的 token $y_i$，如下式所示。

$$y_i = f(e_{s-1}, e_{e+1}, p_{i-s+1})$$

$p$为相对位置编码，维度是200，对应的函数$f$是一个两层的前向网络，激活函数为 GELU，上述三个输入数据拼接起来后送入到这个前向网络，伪代码如下。

$$h_0 = \[ x_{s-1}; x_{e+1}; p_{i-s+1} \]$$

$$h_1 = \textrm{LayerNorm}(\textrm{GeLU}(W_1h_0))$$

$$y_i = \textrm{LayerNorm}(\textrm{GeLU}(W_2h_1))$$

然后SpanBERT对应的预训练任务用公式表示如下。

$$\mathcal{L}_(x _i) = \mathcal{L} _{MLM}(x _i) + \mathcal{L} _{SBO}(x _i) = -\log P(x _i | \mathbf{x _i}) - \log P(x _i | \mathbf{y _i})$$

其中，$x_i$为预测的 token 的索引，$\mathbf{x_i}$ 为模型输出的对应位置的特征向量，$\mathbf{y_i}$为上述通过 SBO 计算出来的特征向量。

去掉 NSP 任务是指，只用一个句子来计算上述的SBO任务，作者主要发现BERT用到的NSP任务中构造负样本的方式（其它doc的句子作为负样本）噪声太大，会妨碍模型的学习。

总结上文，SpanBERT论文里起始用到的两个掩膜Loss，一个是普通的 MLM，一个是 SBO，他们的关系以及计算方式如下图所示。

![图 - 1 SpanBERT计算MLM & SBO损失函数示意图](/imgs/mlm-related/span0.png)

### SpanBERT代码实现

要分析SpanBERT的实现，代码部分主要就是三个部分：数据加载、模型计算、Loss计算，至于transformers & fairseq 等框架问题都比较清洗，花点时间理解一下即可。

数据加载部分的重点在于掩码的生成，经过`spanbert.py::SpanBertTask.load_dataset()`函数内调用`indexed_dataset.py::IndexedRawTextDataset`以及`BlockDataset`等Dataset类，最终在`NoNSPSpanBertDataset`类内完成掩膜的生成以及对应label的计算等，这里采用的掩膜方式为`PairWithSpanMaskingScheme`类。下面是主要的`mask()`函数的代码。

送入到损失函数计算的数据生成代码是：

``` python
def __getitem__(self, index):
        with data_utils.numpy_seed(self.seed + index):
            block = self.dataset[index]

        # tagmap -> default None
        tagmap = self.dataset.tag_map[block[0]:block[1]] if self.dataset.tag_map is not None else None
        masked_block, masked_tgt, pair_targets = \
            self._mask_block(self.dataset.tokens[block[0]:block[1]], tagmap)        #   进行掩码!

        item = np.concatenate(
            [
                [self.vocab.cls()],
                masked_block,
                [self.vocab.sep()],
            ]
        )
        target = np.concatenate([[self.vocab.pad()], masked_tgt, [self.vocab.pad()]])
        seg = np.zeros(block[1] - block[0] + 2)
        if pair_targets is not None and  len(pair_targets) > 0:
            # dummy = [[0 for i in range(self.args.max_pair_targets + 2)]]
            # add 1 to the first two since they are input indices. Rest are targets.
            pair_targets = [[(x+1) if i < 2 else x for i, x in enumerate(pair_tgt)] for pair_tgt in pair_targets]
            # pair_targets = dummy + pair_targets
            pair_targets = torch.from_numpy(np.array(pair_targets)).long()
        else:
            pair_targets = torch.zeros((1, self.args.max_pair_targets + 2), dtype=torch.long)
        return {
            'id': index,
            'source': torch.from_numpy(item).long(),
            'segment_labels': torch.from_numpy(seg).long(),
            ## 重点是 lm_target & pair_targets 的生成过程以及内容
            'lm_target': torch.from_numpy(target).long(),
            'pair_targets': pair_targets,
        }
```

可见，返回的就是 sample 的index，当前句子对应token的索引，还有就是segment type ids，前两者是BERT用到的数据，后面的就是MLM、SBO任务对应的标签数据了，分别是lm target 以及 pair targets。

``` python
def mask(self, sentence, tagmap=None):
        sent_length = len(sentence)
        mask_num = math.ceil(sent_length * self.mask_ratio)
        mask = set()
        word_piece_map = self.paragraph_info.get_word_piece_map(sentence)
        spans = []
        while len(mask) < mask_num:
            span_len = np.random.choice(self.lens, p=self.len_distrib)
            tagged_indices = None
            if tagmap is not None:
                tagged_indices = [max(0, i - np.random.randint(0, span_len)) for i in range(tagmap.length()) if tagmap[i]]
                tagged_indices += [np.random.choice(sent_length)] * int(len(tagged_indices) == 0)
            anchor  = np.random.choice(sent_length) if np.random.rand() >= self.args.tagged_anchor_prob else np.random.choice(tagged_indices)
            if anchor in mask:
                continue
            # find word start, end
            left1, right1 = self.paragraph_info.get_word_start(sentence, anchor, word_piece_map), self.paragraph_info.get_word_end(sentence, anchor, word_piece_map)
            spans.append([left1, left1])
            for i in range(left1, right1):
                if len(mask) >= mask_num:
                    break
                mask.add(i)
                spans[-1][-1] = i
            num_words = 1
            right2 = right1
            while num_words < span_len and right2 < len(sentence) and len(mask) < mask_num:
                # complete current word
                left2 = right2
                right2 = self.paragraph_info.get_word_end(sentence, right2, word_piece_map)
                num_words += 1
                for i in range(left2, right2):
                    if len(mask) >= mask_num:
                        break
                    mask.add(i)
                    spans[-1][-1] = i
        sentence, target, pair_targets = span_masking(sentence, spans, self.tokens, self.pad, self.mask_id, self.max_pair_targets, mask, replacement=self.args.replacement_method, endpoints=self.args.endpoints)
        if self.args.return_only_spans:
            pair_targets = None
        return sentence, target, pair_targets
```
 
变量`sentence`里面包含的就是所有的token元素，`get_word_piece_map()`函数将输入的对应位置tokens判断是否是词首还是非词首。外面的 `while` 循环主要就是生成 span 范围，这里每次生成 span 范围作为一个`[left, right]`保存到 span 变量里面。生成所有的 span 信息之后，就作为参数传入到 `span_masking()`函数里了，也是一个非常重要的函数。

``` python
def span_masking(sentence, spans, tokens, pad, mask_id, pad_len, mask, replacement='word_piece', endpoints='external'):
    sentence = np.copy(sentence)
    sent_length = len(sentence)
    target = np.full(sent_length, pad)
    pair_targets = []
    spans = merge_intervals(spans)
    assert len(mask) == sum([e - s + 1 for s,e in spans])
    # print(list(enumerate(sentence)))
    for start, end in spans:
        # endpoints = `external`
        lower_limit = 0 if endpoints == 'external' else -1
        upper_limit = sent_length - 1 if endpoints == 'external' else sent_length
        if start > lower_limit and end < upper_limit:
            if endpoints == 'external':
                pair_targets += [[start - 1, end + 1]]
            else:
                pair_targets += [[start, end]]
            # pair_targets[-1]元素的结构是: [s-1, e+1, x_s, x_{s+1} ... x_{e}]，元素个数是 2 + （e - s)
            pair_targets[-1] += [sentence[i] for i in range(start, end + 1)]
        rand = np.random.random()       # 整个 span 只用一种替换方式，比如 mask 或者随机其它 token 或者 全保持不变
        for i in range(start, end + 1):
            assert i in mask
            target[i] = sentence[i]
            if replacement == 'word_piece':
                rand = np.random.random()
            if rand < 0.8:
                sentence[i] = mask_id
            elif rand < 0.9:
                # sample random token according to input distribution
                sentence[i] = np.random.choice(tokens)
    # pair_targets 的维度是：(pair nums, 2 + (e - s))
    # + 2 表示的是 s - 1, e + 1 这两个位置信息
    pair_targets = pad_to_len(pair_targets, pad, pad_len + 2)
    # if pair_targets is None:
    return sentence, target, pair_targets
```

这个函数，首先调用`merge_intervals()`函数合并那些有重叠的span区域，可以参考源代码，比较简单。`mask`参数保存的是那些位于 span 掩膜下面的位置信息。然后`for`循环里面可以分为两个部分，上面一部分生成SBO的标签，下面的`for`循环生成MLM的标签，这里`endpoints`的参数是`external`，对应的上面原理部分的$x_{s-1}, x_{e+1}$两个 token 对应的位置。SBO标签信息保存在`pair_targets`里面，这是一个二维list，里面的每个 list 表示一个 span 范围，并且元素的结构是`[s-1, e+1, x_{s}, x_{s+1} ... x_{e}]`，这里`x`表示真是的token字符。然后第二部分就体现了论文中提到的只用一个掩膜方式进行掩码，也就是有一个全局的`rand`参数存在，`for`循环里面就是正常的 MLM 标签生成过程了。注意送入`pad_to_len`中最大长度 + 2 了，`pad_len`对应模型实现部分的`max_targets`参数。

最后的`pad_to_len()`函数将`pair_targets`参数扩展到`pad_len + 2`的形式，产生的结果会直接用于Loss的计算，所以这里给出具体实现，函数如下。

``` python
def pad_to_len(pair_targets, pad, max_pair_target_len):
    for i in range(len(pair_targets)):
        pair_targets[i] = pair_targets[i][:max_pair_target_len]
        this_len = len(pair_targets[i])
        # 补全
        for j in range(max_pair_target_len - this_len):
            pair_targets[i].append(pad)
    # 返回的数据尺寸是（pair nums, max_pair_target_len）
    return pair_targets
```

这个函数做的事情就是截断 & 补全，返回的数据尺寸见注释。至此，就分析完了数据加载过程，送给模型输入的数据就是上面`__getitem__`函数的返回结果了。那么怎么组成一个 batch 数据呢，毕竟每个sample可能的mask长度以及 span pair 的个数不一定相同。具体的`collector`函数实现如下。

``` python
def _collate(self, samples, pad_idx):
        if len(samples) == 0:
            return {}

        def merge(key):
            return data_utils.collate_tokens(
                [s[key] for s in samples], pad_idx, left_pad=False,
            )
        def merge_2d(key):
            return data_utils.collate_2d(
                [s[key] for s in samples], pad_idx, left_pad=False,
            )
        pair_targets = merge_2d('pair_targets')

        return {
            'id': torch.LongTensor([s['id'] for s in samples]),
            'ntokens': sum(len(s['source']) for s in samples),
            'net_input': {
                'src_tokens': merge('source'),
                'segment_labels': merge('segment_labels'),
                'pairs': pair_targets[:, :, :2]
            },
            'lm_target': merge('lm_target'),
            'nsentences': samples[0]['source'].size(0),
            'pair_targets': pair_targets[:, :, 2:]      # (pair nums, max pair target len)
        }
```
以及对应的`merge_2d()`函数，这个函数的实现如下。    

``` python
def merge_2d(key):
            return data_utils.collate_2d(
                [s[key] for s in samples], pad_idx, left_pad=False,
            )
def collate_2d(values, pad_idx, left_pad, move_eos_to_beginning=False):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size_0 = max(v.size(0) for v in values)
    size_1 = max(v.size(1) for v in values)
    res = values[0].new(len(values), size_0, size_1).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            assert src[-1] == eos_idx
            dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i, size_0 - v.size(0):, size_1 - v.size(1):] if left_pad else res[i, :v.size(0), :v.size(1)])
    return res

```

可以看出，这里将`pair_targets`进行了分离，分别用于模型前向 & 损失计算。

然后再看模型结构。模型构成主要分为两部分，一个是底层的由正常 transformer 层构成的backbone，然后另一个就是在其上的 Head 部分，这里主要就是SBO任务对应的Head 的实现，这部分实现在`BertPairTargetPredictionHead`类中。

``` python
class BertPairTargetPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights, max_targets=20, position_embedding_size=200):
        super(BertPairTargetPredictionHead, self).__init__()
        self.position_embeddings = nn.Embedding(max_targets, position_embedding_size)
        self.mlp_layer_norm = MLPWithLayerNorm(config, config.hidden_size * 2 + position_embedding_size)
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),      # hidden size
                                 bert_model_embedding_weights.size(0),      # vocab size
                                 bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))
        self.max_targets = max_targets

    def forward(self, hidden_states, pairs):
        ## 整体思路是使用 Span 边界的两个 token，预测被 mask 掉的 token 对应的 word
        bs, num_pairs, _ = pairs.size()
        bs, seq_len, dim = hidden_states.size()
        # pair indices: (bs, num_pairs)
        left, right = pairs[:,:, 0], pairs[:, :, 1]
        # (bs, num_pairs, dim)
        left_hidden = torch.gather(hidden_states, 1, left.unsqueeze(2).repeat(1, 1, dim))
        # pair states: bs * num_pairs, max_targets, dim
        left_hidden = left_hidden.contiguous().view(bs * num_pairs, dim).unsqueeze(1).repeat(1, self.max_targets, 1)
        right_hidden = torch.gather(hidden_states, 1, right.unsqueeze(2).repeat(1, 1, dim))
        # bs * num_pairs, max_targets, dim
        right_hidden = right_hidden.contiguous().view(bs * num_pairs, dim).unsqueeze(1).repeat(1, self.max_targets, 1)

        # (max_targets, dim)
        position_embeddings = self.position_embeddings.weight
        hidden_states = self.mlp_layer_norm(torch.cat((left_hidden, right_hidden, position_embeddings.unsqueeze(0).repeat(bs * num_pairs, 1, 1)), -1))
        # target scores : bs * num_pairs, max_targets, vocab_size
        target_scores = self.decoder(hidden_states) + self.bias
        return target_scores
```

可以看出，主要过程就是将左边边界的token以及右边边界的 token 取出来，与相对位置编码`self.position_embedding`拼接起来送入前向计算网络。值得注意的是这里返回的数据尺寸是`(bs * num_pairs, max targets, vocab size)`。

提到的损失函数的实现如下。

``` python
@register_criterion('span_bert_loss')
class NoNSPPairLoss(FairseqCriterion):
    """Implementation for loss of SpanBert
        Combine masked language model loss with the SBO loss. 
    """

    def __init__(self, args, task):
        super().__init__(args, task)
        self.args = args
        self.aux_loss_weight = getattr(args, 'pair_loss_weight', 0)

    def forward(self, model, sample, reduce=True):
        net_output = model(**sample['net_input'])
        lm_targets = sample['lm_target'].view(-1)

        # mlm loss
        lm_logits = net_output[0]
        lm_logits = lm_logits.view(-1, lm_logits.size(-1))
        lm_loss = F.cross_entropy(
            lm_logits,      # (bs * seq_len, )
            lm_targets,     # (bs * seq_len, )
            size_average=False,
            ignore_index=self.padding_idx,
            reduce=reduce
        )

        # SBO loss
        pair_target_logits = net_output[2]
        pair_target_logits = pair_target_logits.view(-1, pair_target_logits.size(-1))
        pair_targets = sample['pair_targets'].view(-1)
        pair_loss = F.cross_entropy(
            pair_target_logits,     # (bs * pair_nums * max_target, vocab size)
            pair_targets,           # (bs * pair_nums * max_target, )
            size_average=False,
            ignore_index=self.padding_idx,
            reduce=reduce
        )


        nsentences = sample['lm_target'].size(0)
        ntokens = utils.strip_pad(lm_targets, self.padding_idx).numel()
        npairs = utils.strip_pad(pair_targets, self.padding_idx).numel() + 1

        sample_size = nsentences if self.args.sentence_avg else ntokens
        loss = lm_loss / ntokens + (self.aux_loss_weight * pair_loss / npairs)
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'lm_loss': utils.item(lm_loss.data) if reduce else lm_loss.data,
            'pair_loss':  utils.item(pair_loss.data) if reduce else pair_loss.data,
            'ntokens': ntokens,
            'npairs': npairs,
            'nsentences': nsentences,
            'sample_size': sample_size,
            'aux_loss_weight': self.aux_loss_weight
        }
        return loss, sample_size, logging_output
```

`forward`函数里面就是对应的MLM / SBO 两个Loss 的计算过程，两个计算过程没啥区别，都是分类任务预测被掩膜的token的具体是啥word。

需要在整理一下SBO任务的模型输出 & 损失计算。首先模型输出的尺寸是(bs * pair num * max pair target len, vocab size)，对应的label的尺寸是(bs * pair num * max pair target len, )，所以尺寸上没有问题，但是相对位置编码的对应关系该如何呢？这里需要在提醒一下，模型实现中的这个position embedding是**相对位置**信息，不是绝对位置信息，所以`BertPairTargetPredictionHead`中使用`.repeat(1, self.max_targets, 1)`没有问题，在与pair target计算损失的时候，预测的数据就是根据相对位置编码预测出来的，然后与对应标签计算分类loss，也就是没有错了！（理解能力真是够了）

至此，就分析完SpanBERT的实现细节了。

## MacBERT

与WWM类似，该作者又提出了 Mac (MLM As Correction) 方式的掩码生成过程。实现细节有以下三点。

* 引入类似 SpanBERT 的 N-Gram 掩膜，N 由 1 - 4 的概率分别为 40%, 30%, 20%, 10%，注意N这个长度是指词组的个数，而不是字的个数，相当于在 WWM 基础上配合 Span 来实现(当N=1时即为WWM)，起始由上面 SpanBERT 的实现可以看出来，SpanBERT 就是英文版的 N-Gram Masking 了
* 使用[Synonyms](https://github.com/chatopera/Synonyms)来得到相似字来作为掩码，但如果没有找到相似词，那么就降级为 Random Masking，即随机token替换，这一步称为 Mac，也就是 MLM As Correction。
* 掩码方式80%的用相似字替换、10%的随机替换、10%保持不变，总的掩码占比是 15%

在分析阶段，作者给出了MacBERT各个部分对效果的影响，包括N-Gram掩膜、相似词替换，并且发现 NSP 不如 SOP 效果好，所以作者实际使用的是 SOP 预训练任务。

![图 - 2 MacBERT中各个trick的影响](/imgs/mlm-related/mac2.png)

作者与XLNet的作者也都提到`[MASK]`字符只在预训练阶段存在，在实际推理阶段并不存在，这种差别会导致效果变差，MacBERT作者实验了以下几个设置用于研究对效果的影响到底有多大。首先需要说明的是，BERT中的预训练任务包括 MLM / NSP，已经很多人发现这个MLM任务比 NSP 任务更重要，但是对于 MLM 任务，需要解答两个问题，首先是怎么选择被掩码的tokens，然后是这些被选出来的token应该用什么被替换，也就是掩码是什么。

下面四种情况都是，句子长度的 15 % 的字符被掩膜，并且15%中的10%部分保持不变，剩下的区别如下。

* MacBERT: 80% 的tokens被替换成相似tokens，10%被随机替换
* Random Replace: 90% 的tokens都使用随机替换
* Partial Mask: 也是 BERT 使用的方式，80% 的tokens被替换成`[MASK]`，10%的被随机替换
* All Mask: 也就是90%的tokens都被替换成`[MASK]`

效果如下，这是在CMRC任务上的结果。

![图 - 2 MacBERT几种MLM的效果](/imgs/mlm-related/mac1.png)

可以发现，即使使用 Random Replace 效果都比BERT的Partial Mask方式更好。

当batch size大于1024时，作者选择使用 LAMB 优化器，而小于1024时，采用AdamW，又一个使用LAMB训练大Batch的论文。

作者也提到，不论是 WWM / SpanBERT / MacBERT 都仅仅设计到训练阶段MLM任务中Mask的生成有关，对其他部分都没有影响。所以这里就给出关键的 N-Gram Masking & Mac 数据的生成过程。

但是现在就只有一个疑问，如果返回的近义词跟原词的长度不一致怎么办？一种办法是直接截断；另一种是如果没有长度相等的词组，那么就随机替换token，类似 WWM 中以 wordpiece 为单位进行随机替换总可以了吧。本来想根据代码里实现找到答案的，但奈何没有看到预训练时数据生成用到的脚本。难办。

## 后记

阅读这个论文真正体现了英语能力限制论文理解层次，一直以为SpanBERT中的这句话

>  However, we perform this replacement at the span level and not for each token individually.

说的是将一个任意长度的文本只用一个 `[MASK]` 替换，然后对应的 SBO 任务是预测出这个`[MASK]`到底代表了几个 token，并且这些 token 的起始位置在哪里！想了半天、看了几天代码才弄明白。

不过按照之前的想法，是否可行呢，是否可以让模型学习到更深入层次的语言结构信息呢？未知、未验证。
