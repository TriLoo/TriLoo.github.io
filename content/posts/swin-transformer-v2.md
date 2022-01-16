---
title: "Swin Transformer V2"
date: 2021-11-23T12:07:18+08:00
draft: false
mathjax: true
excerpt_separator: <!--more-->
---
本文围绕如何有效的增加模型参数量、弥补不同任务输入图像尺寸不同时的Windows大小不同导致的相对位置编码变化问题这两个任务提出了解决方案，总的来说，方案简单有效，值得学习。<!--more-->

## 背景

首先，NLP里面用到的 Transformer 模型的参数量借助 MoE 等技术已经到达万亿规模，虽然可能存在没有充分利用模型容量的问题，但是整体来看，随着模型容量（参数量）的提高，在语言任务上的效果是越好的。

然而在CV领域并非这样，随着模型参数量的增加，模型训练也更加困难，并且其实现阶段图像还是一个比较低效率的信息载体（没法像文本那样传递更抽象的信息），通常存在大量的像素冗余，而这些冗余又是传递信息所必需的，这导致图像数据的收集以及挖掘抽象信息从而高效利用数据都存在问题；另一方面，CV多样的任务对输入图像的尺寸也有不同的要求，如果只是全局分类，那么图像尺寸只需要 224 * 224 就可以做，但是如果涉及到分割、目标检测等任务则需要非常细粒度的信息，也就需要保证输入图像的分辨率足够大，一般来说模型预训练数据量大，为了保证训练效率，预训练一般采用比较低的分辨率，这也就导致与下游这些任务对分辨率需求上存在 Gap，也导致模型的效果在这些任务上下降。

针对模型的 Scaling 问题，本文主要分析了为什么随着模型参数量的增加，会存在模型训练不稳定的问题，作者发现主要是因为残差结构的存在，导致约往后，模型的输出的数值的量级也越大，结果就是后面层的激活值输出相对于浅层的激活值相差达到10e4这个量级。然后针对分辨率的变化，作者提出用 Log-spaced Continuous Position Bias 来Scaling 相对位置编码参数。

关于CV图片数据的利用，可以参考kaiming的 MAE论文，也就是人为的掩盖75%的像素来让模型高效利用数据。但是不论是CV中模型参数量的扩展还是数据的利用都应该会有更大突破的想法出现。

下面是本文的主要三个技术的一些细节。

## 方案

针对模型参数 Scaling 问题，作者发现是因为模型后面层的激活输出值量级太大了，如图一所示，可以看出（以B-Pre）为例，Block 22 的最大值达到了Block 1 最大值的 10e4 倍。

![图 1 不同层的输出值的量级增加非常快](/imgs/swin-transformer-v2/swinv2-0.png)

### Post Normalization & Scaled Cosine Attention

看 ViT 的实现代码可以发现，ViT 采用了 Pre-Normalization 实现方式，所以这里采用 Post-Normalization 的方式，注意，这里只对 Attention / MLP 层的输出计算 Layer Norm；Scaled Cosine Attention 其实就是使用 Scaled Cosine 计算代替原来的 Scaled Dot-Product Attention 的计算。这里虽然都有 Scaled，但是前者的 Scale $\tau$ 是学习得到的参数，并且大于0.01，但是后者是一个固定的数$\sqrt{d_k}$，也就是每个 Head 的维度。

Scaled Cosine Attention 的数学表达式是：

$$Sim(q_i, k_j) = \cos (q_i, k_j) / \tau + B_{ij}$$

其中，$B_{ij}$就是相对位置编码参数，本文中也就是下面提到的 Log-spaced Continuous Position Bias来计算的。

上述两个改动与V1版本的对比示意图如图2所示。

![图 2 Post Norm & Scaled Cosine Attention 示意图](/imgs/swin-transformer-v2/swinv2-1.png)

图3展示了上述两个改动的 ablation 实验，发现两个改动都对效果有帮助，当然重要的还是可以将模型容量进行扩充。

![图 3 Post Norm & Scaled Cosine Attention 效果分析](/imgs/swin-transformer-v2/swinv2-3.png)

### Log-space Continuous Position Bias

这一部分的目的是实现 Scaling Up Window Resolution。首先来看下什么是 Continuous Position Bias，这是相对于参数化的相对位置编码而言的，后者是直接学习相对位置编码的 Embedding；而 Continuous Position Bias 的方案是采用一个小的 Meta 网络来映射相对位置：

$$B(\Delta x, \Delta y) = \mathcal{G}(\Delta x, \Delta y)$$

其中$\mathcal{G}$可以是一个中间使用 ReLU 激活函数的2层 MLP。

为了避免因为 windows 大小变化太大导致需要外推出很多之前没用过的相对位置信息，作者提出用将线性空间的星队距离映射到 log 空间中，然后输入到上述Meta网络中生成相对位置编码。映射到 log 空间的过程如下：

$$\hat{\Delta x} = sign (x) \cdot \log (1 + | \Delta x |)$$

$$\hat{\Delta y} = sign (y) \cdot \log (1 + | \Delta y |)$$

其中，$\Delta x, \Delta y, \hat{\Delta x}, \hat{\Delta y}$分别表示线性空间、log空间的相对位置量，与Swin Transformer使用的参数化的位置编码相比，log域的效果最好。

![图 4 3种相对位置编码效果对比](/imgs/swin-transformer-v2/swinv2-2.png)

表格里每个位置表示不适用 / 使用微调训练的效果，可以看出，使用 log 域的相对位置编码，即使不进行微调训练也可以在一定程度上保持模型效果，而且还可能效果更好（这主要是因为windows变大了）！

### 省显存

作者用到了下面三个措施来降低显存使用。

* ZeRO Stage 1，即将AdamW优化器的一些参数分配到不同的 GPU 上（类似模型并行），这个做法可以显著降低显存开销，重要的是对训练速度影响非常小
* Activation Checkpoint，这样大概会降低训练速度30%的样子
* Sequential Self-Attention Computation，讲Batch内样本的 Self Attention 的计算串行化，对训练速度非常小

此外，增加模型参数量的方法主要还是增加 channel 宽度、增加 Stage 3 的层数，如6 -> 18 -> 42等。

## 实现

有待补充。
