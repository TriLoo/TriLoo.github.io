---
title: "Cnn Transformer 系列之 Volo"
date: 2021-09-06T16:26:11+08:00
draft: false

mathjax: true

excerpt_separator: <!--more-->
---
VOLO论文。<!--more-->

## Outlook Attention

关于论文里主要的 Outlook Attenion 模块的实现有下面几点理解。

* Unfold 计算
  
  实现过程是将输入 Tensor 在划窗内的数据Flatten 成列，Flatten 过程中按照 Row-Major （第一行 -> 第二行 -> 第N行） 的方式完成；窗口滑动过程中得到新的列则拼接在后面。假设 unfold 操作的 kernel size 为 K，输入Tensor的尺寸为 $C \times H \times W$，则 unfold 得到的结果为 $(C \cdot K^2) \times (H \cdot W)$，最后一维表示考虑 padding 后，窗口一共滑动了 $H\cdot W$次。

* Fold 计算

  此计算是 Unfold 的反过程，并且将恢复过程中对应同一个位置(i, j)的$K^2$结果求和作为该位置上新的结果。

* 与 Convolution 的区别

  卷积计算过程是将以 (i, j) 为中心的窗口内的元素进行加权求和，权重为对应的 kernel 数据；Outlook Attention 计算过程是将 (i, j) 位置的元素在不同的滑窗内计算的结果进行求和，在不同滑创内的结果是指，该窗口内的 $K \times K$ 个元素基于 $K^2 \times K^2$ 个权重得到 $K \times K$个位置上的向量，所以滑窗内每个位置上都有一个对应当前滑窗的结果，Outlook Attention 也就是将(i, j)位置上向量参与的所有的滑窗内的结果求和得到新的向量。

  以$3 \times 3$的滑窗为例，则输入 Tensor 每个位置上的元素一共参与了 9 个滑窗，则结果就是这9个滑窗内的结果求和，而滑窗内的结果是根据该滑窗内的9个元素加权平均（权重经过 Softmax 了）得来的。

* 论文中的伪代码

  其中比较奇怪的一个地方在于，`v=v_pj(x).permute(2, 1, 0)` 得到的尺寸是$(C, W, H)$，为什么 W / H 这两个维度还需要转置呢？

  目前的想法是作者写错了，因为代码实现里还是 (C, H, W) 的顺序。伪代码里mul(a, v)计算相当于第 (i, j) 位置对应滑窗内的权重是由 (j, i) 位置上向量根据一个 Linear 层得到的，这肯定是不对的，毕竟图片肯定不是沿着对角线对称的，即使对称，Stem Block 计算的结果也不是。如果理解错了请告诉我。

论文中给出的伪代码如下：

![图-1 Outlook Attention示例代码](/imgs/volo/volo0.png)

## 引入多头注意力

在这里，多头注意力简单来说就是得到 N (head num)个不同的权重分布，同时 Value 矩阵的 hidden status 维度分成 N 组，然后每组对应一个权重分布。多头注意力就是在 Value 的每个分组上求解加权平均，然后拼接起来恢复输入时候的尺寸。

对应 Outlook Attention 的实现，就是将得到权重的全连接层由$W^A \in R^{C \times K^4}$变为$W^A \in R^{C \times N \cdot K^{4}}$，同时将Value结果由$R^{H \times W \times C}$Reshape成$R^{H \times W \times N \times C_{in}}$即可。计算过程中，只需要将 $A$ 分成 $N$份，然后分别用于 $N$ 组 $V$ 最后将结果进行拼接完成计算。

## 构建模型

模型部分主要分为三个部分：Stem, Volo Stages, Transformer Stages。

Stem 部分默认采用的是3层（Conv + BN + ReLU）结构，将输入数据下采样一倍，然后再经过一个 Patch Projection 层（Conv计算），继续下采样 4 倍，所以一共下采样8倍。这种 Stem 避免了 ViT 中的 $16 \times 16$这种大Kernel size / stride 的卷积计算，相关论文指出，这种大 ks / stride 的方式不利于训练稳定性。

中间是由 Multi-head Outlook Attention 构成的Stage。

最后是 Transformer 层构成的 Stages。

论文中对比了 Outlook Attention 层数与 Transformer 层数之间的比例，结论是 1 : 3 的时候效果比较好。

## 模型精度

论文中采用了 Token Labeling，是 LV-ViT 论文中提出来的。基于 VOLO-D1，在不使用 Token Labeling 计算 Loss，同时引入 Random Augmentation 等增广方式后，自己训练精读是 82.2，使用论文里面的配置，确实可以达到 84.19 的准确率，总体来说效果还是不错的。
