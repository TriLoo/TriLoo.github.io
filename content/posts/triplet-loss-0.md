---
title: "Triplet Loss 与在线难例挖掘（译）"
date: 2022-01-18T19:07:51+08:00
draft: false
mathjax: true
excerpt_separator: <!--more-->
---
虽然 triplet loss 实现非常简单，但是简单的 loss 要想用好也是需要更细致的分析与调试。<!--more-->

本文主要是对博客[Triplet Loss and Online Triplet Mining in TensorFlow](https://omoindrot.github.io/triplet-loss)的简要翻译。文章中以人脸识别为背景，且基于 Tensorflow 的实现。

## Triplet Loss

* 与Softmax比较

  对于分类任务，直接的做法就是使用 Softmax 获取每个类别的概率，然后与 Label 一起计算Loss，Label 可以是soft label 也可以是 one-hot label。但是，这里有个限制是，使用Softmax的情况，类别数必须固定，也就是没法将样本分类成一个没有见过的类别。

  但是在人脸识别中，类别数并不是固定的，总是会需要判断两个不在训练集中的人物是否是同一个人。Triplet Loss 就可以训练模型，得到输入人脸的 Embedding，然后在 Embedding 空间中，同一个人的高维点之间的距离小于与另一个人之间的距离。

* Tripet Loss 定义

  首先，triplet loss由三个数据构成，分别是 anchor, positive, negative样本；然后定义也可以分为距离的形式、相似度的形式两种形式，本质上肯定是一致的。
  
  以距离的形式为例，定义非常直接，假设 $ap, an$ 分别表示anchor与positive, negative样本之间的距离，这个距离可以是 cosine 距离，也可以是欧式距离，这两个距离本质是一样的，可以互相转换。那么 triplet loss 的计算过程是：

  $$triplet loss = \max(ap - an + margin, 0.0)$$

  以相似度的形式，$ap, an$分别表示 anchor 与 positive, negative 之间的相似度，那么triplet loss定义：

  $$triplet loss = \max(an - ap + margin, 0.0)$$

  基于这个Loss训练的目的就是，让类内样本之间的距离小于类间样本距离，并且差值还要大于一个 margin。

## Triplet Loss Mining

根据 triplet loss 的计算过程，可以将负样本分为三个类别(以距离形式为例)：

* easy triplet
  
  损失函数为0，此时有：$ap + margin < an$
  
* hard triplet

  此时负样本比正样本更接近Anchor样本，即$an < ap$

* semi-hard triplet

  此时负样本距离Anchor的距离比正样本到 Anchor 的距离更远，但是差别小于 Margin，即$ap < an < ap + margin$

上面三种负样本的示例如下：

![图 - 0 负样本分类](/imgs/triplet-loss-0/negatives0.png)

首先需要说明的是，上述三种负样本对于模型训练的贡献肯定是不一样的，最常见的做法就是生成随机的 semi-hard 负样本进行训练；但是更有用的做法是，挖掘出那些更有用的三元组进行训练。

怎么挖掘这种三元组呢？一种方法是离线难例挖掘，另一种方法是在线难例挖掘。

### 离线难例挖掘

离线难例挖掘这种方法就是在每个 Epoch 开始的地方，首先计算所有样本的 Embedding，然后从其中挑选组合出 hard triplet & semi-hard triplet 三元组进行训练。这种方法最明显的缺点就是效率太低了。

### 在线难例挖掘

这种方法是从 Batch 内在线的找出有用的hard triplet三元组，也有两种方案。

* 使用 Batch 内所有可用的负样本

  此时就是在 Batch 内其他类别的所有样本作为当前样本（类别）的负样本，进行训练。

* 使用 Batch 内最难的负样本三元组

  这种方法就是对于每个 Anchor，计算出距离最远的同类别的样本（正样本），然后选出距离最近的其他类别的样本（负样本）构成triplet 三元组进行训练。这里又个需要注意的地方在于，选取样本时，正样本需要保证同类别，负样本需要保证不同类别，选取负样本时，对于那些同类别的样本需要 mask 掉（也就是赋值给一个最大值，这个最大值可以是对应类型的最大值，也可以是一个超过取值范围的值，比如 cosine 距离的时候，可以将最大值设置为 2.0，因为 cosine 距离最大不会超过2.0)；选取正样本的过程比较容易，直接对不同类别的样本赋值给一个很小的数就行了。

文章中，作者说后者（最难三元组）的效果超过第一个，但是实际测试中发现第一中效果更好一点，所以这一点也在此说明了，简单的 triplet loss 要想用好也是需要一些心思的。至于是否有其它更好的使用方法，实际使用中一个小的改动或许可以更好的利用 triplet loss函数的方法就留作下一篇博客里探讨一下吧。
