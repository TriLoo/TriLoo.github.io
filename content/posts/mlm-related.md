---
title: "常见掩码生成方式"
date: 2021-10-19T19:51:11+08:00
draft: false
mathjax: true

excerpt_separator: <!--more-->
---
主要是几种常见的MLM的改进以及对应的代码实现，包括WWM, SpanBERT, ERNIE这三种。<!--more-->

这一部分可能会涉及到一些分词算法的概念，但是主要用于对分词结果进行处理。

## WWM

WWM 也算是一种简单有效的引入语言知识的方法。直观来说，不再是类似BERT那样分字，然后每个字考虑被掩膜掉，而是使用了 [LTP](http://ltp.ai/) 工具包进行分词，然后制作掩膜的时候，就可以一次性把一个完整的词组进行掩膜掉了。

下图给出了WWM 与 BERT 使用的两种掩膜方式的对比。

![图 - 1 WWM算法分词掩膜效果示意图](/imgs/mlm-related/wwm0.png)

这篇论文还有一点是，作者使用LAMB进行优化，而不是AdamW，毕竟前者更适合Large Batch的情况（作者实际使用的Batch Size =2560(128) / 384(512)）。此外作者还有以下几点BERT训练心得。

* 初始学习率对 BERT 的效果有重要影响，必须仔细调整，但BERT-WWM / BERT 两个模型最优的学习率比较接近，但是ERNIE的学习率差别就很大
* 如果预训练任务与下游任务之间差别较大，则建议基于下有任务也做一下预训练

实际实现中，BERT-WWM / BERT两者最大的区别在于模型实现，数据输入、训练Loss等都没有变化。在英文版WWM中，如果说一个Word被分成若干个子词，那WWM的做法是将这些这些子词分别都进行处理（mask，保留，替换），注意所有的子词并非需要做相同的处理，即同一个Word的多个子词上，可以这个子词作替换，那个子词用mask，还有一个子词保留，这些都是可以的，具体例子参考：[mask的一个小细节](https://github.com/ymcui/Chinese-BERT-wwm/issues/4)，对应的主要信息截图如下。

![图 - 2 WWM算法分词掩膜效果示意图2](/imgs/mlm-related/wwm1.png)

### WWM实现代码

参考：[tf bert-wwm](https://github.com/interviewBubble/Google-ALBERT/blob/master/create_pretraining_data.py)

或者Transformer库里对应的`run_chinese_ref.py`以及`data_collator.py`文件中的`DataCollatorForWholeWordMask`类的实现。

代码分成两个主要部分。第一部分是根据LTP分词结果来为词语非第一个字的前面加上`##`符号；然后第二部分实现对应的 Mask 过程，注意Mask的最大长度不会超过 15% 的阈值。

第一部分根据 LTP 分词结果 配合最长词长度为3开始贪婪尝试。

``` python
def get_new_segment(segment):
    seq_cws = jieba.lcut("".join(segment)) # 分词
    seq_cws_dict = {x: 1 for x in seq_cws} # 分词后的词加入到词典dict
    new_segment = []
    i = 0
    while i < len(segment): # 从句子的第一个字开始处理，知道处理完整个句子
      if len(re.findall('[\u4E00-\u9FA5]', segment[i])) == 0:  # 如果找不到中文的，原文加进去即不用特殊处理。
        new_segment.append(segment[i])
        i += 1
        continue

      has_add = False
      for length in range(3, 0, -1):
        if i + length > len(segment):
          continue
        if ''.join(segment[i:i + length]) in seq_cws_dict:
          new_segment.append(segment[i])
          for l in range(1, length):
            new_segment.append('##' + segment[i + l])
          i += length
          has_add = True
          break
      if not has_add:
        new_segment.append(segment[i])
        i += 1
    # print("get_new_segment.wwm.get_new_segment:",new_segment)
    return new_segment
```

第二部分，根据第一步的结果进行Mask，具体的函数是第一个参考代码里的`create_masked_lm_predictions()`函数。代码省略，但是思路大体就是将属于同一个词组（以`#`开头）放到同一个列表中，然后所有的词组又放在额外一层列表中，至此构成一个两层的列表。接下来就是将这个列表随机打乱，然后从左到右依次mask，直到总的 mask 的长度大于 15% 了，注意这里是随机打乱，然后从左到右依次mask！至于Mask的过程就是以分字为单位进行的。另外一点是，代码里保证总的 mask 掉的长度小于15%，如果mask 掉下一个词组的话，那么就略过；还有一点是，最后通过一个 sort 函数来恢复被打乱的顺序，这个顺序信息是给每个 token 做了个索引。

另一种实现是 transformer 里面的实现，生成 mask 的主要过程在 `_whole_word_mask()` 函数里，主要思路与 Bert-wwm 的实现思路一致，也是随机打乱，然后从头向后一次生成掩膜，不过这里借助额外的单独文件离线生成词组信息。

## ERNIE

百度 ERNIE 的主要思路是将phrase-level strategy & entity-level strategy两种知识引入到模型训练中，具体对应的是 phrase-level masking & entity-level masking，命名实体包括人物、地点、组织、产品等。示意图如图 - 3所示。

![图 - 3 ERNIE 与 BERT MLM区别示意图](/imgs/mlm-related/ernie0.png)

对于实现，[ERNIE-text-classification-pytorch](https://github.com/lonePatient/ERNIE-text-classification-pytorch)仓库里的代码只适合微调训练，不支持预训练任务。

这里以 paddlepaddle 官方库中 ERNIE v1.0 的实现为例进行说明，模型的前向计算以及 Loss 的计算方面与普通 BERT 相同，所以重点在于`ErnieDataReader` 类的实现。

根据[PaddlePaddle-ERNIE-github](https://github.com/PaddlePaddle/ERNIE/blob/repro/README.zh.md#%E6%95%B0%E6%8D%AE%E9%A2%84%E5%A4%84%E7%90%86)文档里的说法，输入数据的一个示例如下:

``` text
1 1048 492 1333 1361 1051 326 2508 5 1803 1827 98 164 133 2777 2696 983 121 4 19 9 634 551 844 85 14 2476 1895 33 13 983 121 23 7 1093 24 46 660 12043 2 1263 6 328 33 121 126 398 276 315 5 63 44 35 25 12043 2;0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1;0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55;-1 0 0 0 0 1 0 1 0 0 1 0 0 1 0 1 0 0 0 0 0 0 1 0 1 0 0 1 0 1 0 0 0 0 1 0 0 0 0 -1 0 0 0 1 0 0 1 0 1 0 0 1 0 1 0 -1;0
```

每个样本由5个 ';' 分隔的字段组成，数据格式: token_ids; sentence_type_ids; position_ids; seg_labels; next_sentence_label；其中 seg_labels 表示分词边界信息: 0表示词首、1表示非词首、-1为占位符, 占位符对应的词为 CLS 或者 SEP。代码中生成 mask 的主要逻辑在`mask()`函数内，这里仅给出实体词级的掩膜，忽略汉字 word piece级别的掩膜生成。

``` python
def mask(batch_tokens,
         seg_labels,
         mask_word_tags,
         total_token_num,
         vocab_size,
         CLS=1,
         SEP=2,
         MASK=3):
    """
    Add mask for batch_tokens, return out, mask_label, mask_pos;
    Note: mask_pos responding the batch_tokens after padded;
    """
    max_len = max([len(sent) for sent in batch_tokens])
    mask_label = []
    mask_pos = []
    prob_mask = np.random.rand(total_token_num)
    # Note: the first token is [CLS], so [low=1]
    replace_ids = np.random.randint(1, high=vocab_size, size=total_token_num)
    pre_sent_len = 0
    prob_index = 0
# [[sentence 0], [sentence 1], [sentence 2] ... [sentence N-1]], batch size = N
    for sent_index, sent in enumerate(batch_tokens):        # sent: current sentence
        mask_flag = False
        mask_word = mask_word_tags[sent_index]
        prob_index += pre_sent_len
        if mask_word:
            beg = 0
            for token_index, token in enumerate(sent):      # tokens in current sentence
                seg_label = seg_labels[sent_index][token_index]     # 表示分词边界信息
                if seg_label == 1:                                  # 非词首
                    continue
                if beg == 0:                                        # 从 3th token 开始
                    if seg_label != -1:
                        beg = token_index
                    continue

                prob = prob_mask[prob_index + beg]                  # 当前词被掩膜的概率
                if prob > 0.15:
                    pass
                else:
                    for index in xrange(beg, token_index):          # 对每个词里的所有 word 都进行掩膜
                        prob = prob_mask[prob_index + index]
                        base_prob = 1.0
                        if index == beg:            # 词组的首字
                            base_prob = 0.15
                        if base_prob * 0.2 < prob <= base_prob:
                            mask_label.append(sent[index])
                            sent[index] = MASK      # 用 Mask 掩膜
                            mask_flag = True
                            mask_pos.append(sent_index * max_len + index)
                        elif base_prob * 0.1 < prob <= base_prob * 0.2:
                            mask_label.append(sent[index])
                            sent[index] = replace_ids[prob_index + index]       # 随机替换其它 token 
                            mask_flag = True
                            mask_pos.append(sent_index * max_len + index)
                        else:
                            mask_label.append(sent[index])                      # 保持不变
                            mask_pos.append(sent_index * max_len + index)

                if seg_label == -1:
                    beg = 0
                else:
                    beg = token_index
        else:
            # do wordpiece masking

        pre_sent_len = len(sent)

    mask_label = np.array(mask_label).astype("int64").reshape([-1, 1])
    mask_pos = np.array(mask_pos).astype("int64").reshape([-1, 1])
    return batch_tokens, mask_label, mask_pos
```

其中，`mask_pos`是用于提取出那些被mask掉的token计算 Loss，所以跟具体实现有关系，transformer 里的那种做法是不需要这个信息的。`mask_word`表示当前的句子是否以词组为单位进行掩膜，所以这里也保留了一定的概率进行 WordPiece 掩膜。`if beg == 0`对应的代码块里保证`beg`的取值范围是$[0, token\_index)$，所以可以取到完整的一个词组；然后就是掩膜部分的计算了，这里每个字只有80%的概率用 `MASK` 替换，然后10%的概率随机Token替换，剩下的10%的概率保持不变，值得注意的地方在于首字的概率用来决定是否进行对词组进行掩膜，并且对应的调整`base_prob`为首字的最大概率，而非首字的最大概率仍为1.0。

至此，一方面是mask的生成过程，另一方面就是生成输入数据中的`seg_labels`这个信息了，只要掌握了算法思想，实现起来还是有很多种方法的。


## 其它

分析完上述几个简单的掩膜方法，一个很直接的思路就是结合 ERNIE 的词组掩膜 + Span 思路，就是说用一个`[MASK]`来代替一个实体词组，然后借鉴 Span 的做法，也让模型预测这个词组的 SBO 任务！效果待验证。

Oscar 多模态模型用到了目标检测的结果作为 tag，这个tag与图片、文本的组织方式为：``。
