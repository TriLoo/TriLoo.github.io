---
title: "分词算法基础"
date: 2021-10-22T19:51:11+08:00
draft: false
mathjax: true

excerpt_separator: <!--more-->
---
常见的几种分词算法小结，包括BERT用到WordPiece以及Albert用到的Byte-Pair-Encoding。<!--more-->

BERT 用到的分词算法称为 Word Piece Tokenizer，Albert 用到的是 [SentencePiece](https://github.com/google/sentencepiece)。SentencePiece 用到的是 Byte-pair-Encoding 算法以及Unigram Language Model算法，Roberta用的也是这种。

Albert 等算法直接使用的是 SentencePiece，这个库是包含上面提到的 BPE / ULM 等子词算法。除此之外，SentencePiece也支持字符、词级别的分词。同时为了支持多语言，SentencePiece将句子视为Unicode编码序列，从而子词算法不依赖于语言的表示。

以BERT用到的分词算法为例进行说明，BERT 中 用到的 Tokenizer 分为两步，第一步是 `BasicTokenizer()`进行处理，第二步是`WordPieceTokenizer`进行处理。

参考包括BERT代码 以及 [BERT 是如何分词的](https://zhuanlan.zhihu.com/p/132361501)。

## BasicTokenizer

`BasicTokenizer`只对输入的文本基于空格进行分割。具体过程如下：

* `_clean_text()` 去掉文本中的控制字符，然后将`\t, \n, \r`等字符用空格代替
* `_tokenizer_chinese_chars()`对中文输入中的每个字两边加上空格，英文输入天然以空格分离
* `_run_strip_accents()`去掉变音符号，代码如下

  ``` python {linenos=table}
  def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)
  ```

  这里涉及两个函数：`unicodedata.normalize()`以及 `unicodedata.category()`。变音符是指这些符号: $\dot{a}, \ddot{u}$等，而这个函数可以将类似$r\acute{e}sum\acute{e}$变为$resume$符号。首先`unicodedata.normalize()`函数返回字符串的规范分解形式（Unicode字符有多种规范形式，代码里默认是`NFD`形式，即规范分解）；`unicodedata.category()`函数返回输入字符的[Unicode类别](https://www.compart.com/en/unicode/category)。

  实际上，变音符号由两个字符组成，通过`unicodedata.normalize()`可以将两者拆分出来；而`unicode_category()`函数可以得到每个拆分出来字符的类别。变音符号对应的字符表示是：Mn，即Nonspacing Mark，非间距标记，变音符号也属于这类；剩下的普通字符对应的类别是Ll，即Lowercase Letter，小写字母。

  针对变音字符，Albert 的处理更直接：

  ``` python
    if not self.keep_accents:
        outputs = unicodedata.normalize("NFKD", outputs)
        outputs = "".join([c for c in outputs if not unicodedata.combining(c)])
    if self.do_lower_case:
        outputs = outputs.lower()
  ```

* `_run_split_on_punc()`基于符号（逗号、感叹号、$等字符）进行分割

  ``` python
      def _run_split_on_punc(self, text, never_split=None):
        """Splits punctuation on a piece of text."""
        if never_split is not None and text in never_split:
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]
  ```
  
  这一步其实就是将连在一块的句子符号分离出来。初始`start_new_word=True`，如果遇到符号，那么就把这个符号单独压入`output`里面，然后再从下一个正常字符开始处理。

* 最后将上述分割结果用空格拼接起来然后再按照空格进行分割。

## WordPieceTokenizer

这一步主要思路是根据词表里的单词按照从右向左贪婪的最长匹配方法对词进行分割成更小的单元，即Piece。

代码如下：

``` python {linenos=table}
    output_tokens = []
    for token in whitespace_tokenize(text):
        chars = list(token)
        if len(chars) > self.max_input_chars_per_word:
            output_tokens.append(self.unk_token)
            continue

        is_bad = False
        start = 0
        sub_tokens = []
        while start < len(chars):
            end = len(chars)
            cur_substr = None
            while start < end:
                substr = "".join(chars[start:end])
                if start > 0:
                    substr = "##" + substr
                if substr in self.vocab:
                    cur_substr = substr
                    break
                end -= 1
            if cur_substr is None:
                is_bad = True
                break
            sub_tokens.append(cur_substr)
            start = end

        if is_bad:
            output_tokens.append(self.unk_token)
        else:
            output_tokens.extend(sub_tokens)
    return output_tokens
```

可以看出来，两个while循环里面，从输入单词的右边开始，end不停的向前推进，同时测试start : end之间的字符是否在vocab字典里面，如果在的话，这些后面的 piece 会加上`##`前缀，作为分词结果保存起来。外面的while循环里会更新start参数，直至粉刺结束。也就是说，单词被分成子词，并且子词以`##`开头。

分词词表里那些以`##`开头的字符就是备选的 word piece，不是单词的开头，而是一个单词被分成好几片 piece 时后面的几个 piece。

Roberta / XLM 等模型中提到的 `<s>` 其实就是 `[CLS]`，同理`</s>`对应`[SEP]`。

至此，BERT里面的分词过程分析完了。

## Byte-Pair-Encoding

BPE 算法也被称为字节对编码或二元编码，简单来说，算法过程就是将相邻出现频次最高的两个连续字节数据用一个新的字节数据表示，直到满足词典中单词的个数或下一个最高频的字节对出现频率为1时终止。优点是可以平衡字典词表大小以及步长（编码句子所需要的token数量），缺点是合词过程是固定的，即没有考虑其它更有效的分词单元。

算法会现在每个词的结尾加上一个结束符`</w>`，用于区分是否位于词的结尾还是词的中间，如`st`出现在`st ar`或`wide st</w>`两个位置意义是完全不同的。

对应的代码主要函数包括：

``` python
import re, collections
def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            # 这里pairs的键是一个 list
            pairs[symbols[i],symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out
```

其中，`get_stats()`函数就是将任意两个连续的bytes（char）拼接起来，加入到字典`pairs`里面，然后就可以取出出现频次最大的字节对了。`merge_vocab()`则根据传进来的连续字符拼接结果`best`(是一个list)进行处理，首先是去除`re`相关的特殊字符，`v_in`是当前字典，然后这个函数将字典中best出现的连续两个字符用新的字符替换掉。

## Word Piece

前面提到了 WordPiece 算法怎么使用，这里是一些制作 WordPiece 词表的细节。

与BPE算法类似，WordPiece 算法也是先将输入的句子分解成子词（如最细粒度的char级别），然后合并子词，不同的地方在于 WordPiece 在合并连续子词的时候会考虑句子的语言模型概率。简单来说，BPE选择的是频次最高的相邻子词进行合并，WordPiece选择的是能够提升语言模型概率最大的相邻子词进行合并加入到词表。

下面用数学公式进行说明。假设当前句子$S=(t_0, t_1, \ldots, t_n)$共n个子词构成， 并且假设句子中每个子词是独立分布的，则句子的语言模型概率定义为：

$$\log P(S) = \sum_{i=1}^{n} P(t_i)$$

即所有子词概率的乘积，子词概率的初始值基于训练预料统计子词频率得到。

然后就是合并子词，假设把相邻位置的 x 和 y 两个子词进行合并成 z，那么句子 S 的似然值变化表示为：

$$\log P(t_z) - (\logP(t_x) + \logP(t_y) = \log \frac{P(t_z)}{P(t_x)P(t_y)}$$

也就是似然值的变化是两个子词之间的互信息。简而言之，WordPiece每次选择合并子词，他们具有最大的互信息值，也就是两个子词在语言模型上具有较强的关联性，它们在语料中经常相邻着出现。

## Unigram Language Model

与 WordPiece 类似，同样基于语言概率模型进行合并子词，不同的地方在于，BPE / WordPiece 两个算法的词表都是由小到大增加，而ULM的词表则是减量法，即先初始化一个大词表，然后根据评估准则不断丢弃词表，直到满足限定条件。不同的地方在于，ULM算法会考虑句子的不同分词形式（即子词单元不同），因而可以输出带概率的多个子词分段。

至于在所有可能的分词结果中选取概率最高的分词形式，计算量比较大，可以使用维特比算法实现；另一方面，子词的初始概率是通过 EM 算法得到的。具体的就了解不深入了。

具体可以参考[NLP三大Subword模型详解：BPE、WordPiece、ULM](https://zhuanlan.zhihu.com/p/198964217)。

## 中文分词工具

中文的 wordpiece 就是分字。基于准确率、分词速度角度对比的话，LTP准召比较高，但是速度最慢，整体的对比如图 - 1。

![图 - 1 不同中文分词算法准召、速度对比](/imgs/tokenizers/comp0.png)

* 哈工大 LTP: [LTP](https://github.com/HIT-SCIR/ltp/blob/master/docs/quickstart.rst)

  支持类似 HanLP 的几个功能！


* 结巴: [jieba - github](https://github.com/fxsjy/jieba)

  * 基于前缀字典实现高效的词图扫描，生成句子中汉字所有可能成词情况所构成的有向无环图
  * 利用动态规划，查找最大概率路径，找出基于词频的最大切分组合
  * 对于未登陆词，采用了基于汉字成词能力的HMM模型，使用了viterbi算法

  支持四种分词模式，精确模式、全模式、搜索引擎模型、paddle模式等。

  此外，jieba 支持分词、自定义词典、关键词提取、词性标注。关键词提取支持基于 TF-IDF（返回TF-IDF权重最大的词）、TextRank提取关键词（分词，然后再固定窗口内计算词的共现关系，构建图，然后计算图中节点的PageRank）。

* SnowNLP
* PkuSeg
* THULAC
* HanLP: [HanLP](https://github.com/hankcs/HanLP)

  支持多大10中任务：分词、词性标注、命名实体识别、依存句法分析、成分句法分析、语义依存分析、语义角色标注、词干提取、词法语法特征分析、抽象意义分析等等。
