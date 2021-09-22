---
title: "Resnet Series"
author: "Triloon"
date: 2021-09-02T14:12:59+08:00
draft: false

excerpt_separator: <!--more-->
---
Residual Connection以及后续发展。<!--more-->

主要是为了自己梳理一下，总不能最基础的残差网络也忘了吧。更多的信息可以参考：[ResNet系列网络演绎过程](https://zhuanlan.zhihu.com/p/353185272)

## 基础

残差网络(ResNet)是2015年何凯明在[Deep Residual Learning for Image Recognition](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)提出的，旁路连接方便了梯度回传，可以帮助模型更好的训练。基础结构如下图1。

![图1 - 残差块](/imgs/resnet-series/residual0.png)

我们知道，VGG / ResNet / Mobilenet 等论文里已经说明现在网络结构设计可以通过简单的 Block 堆叠来构建，并且Blocks可以分组为若干个 Stage，每个 Stage 包含若干层 Block。为了提高计算性能以及提高感受野等，不同 Stage 之间会下采样降低空间分辨率同时提高 channel 个数（神经元）来保证模型容量。对于每个 Stage 的第一层 Block 需要完成下采样、channel翻倍的任务，为了保证输入数据与这两步处理后的输出数据尺寸相同，需要修改旁路，不再是 Indentity，而需要通过卷积完成映射。论文里在每个Block的第一层卷积里使用`stride=2`来完成下采样。 有论文表明，使用 avg pooling 进行下采样会更好，避免丢失很多的信息。

另外，一般配合 BN 时，CNN 的 bias 作用不明显可去掉。对于 bias 的作用可简单参考：[The role of bias in Neural Networks](https://www.pico.net/kb/the-role-of-bias-in-neural-networks/)，猜测是在 BN 之前用于修正 `W * x` 的偏置，**防止方差过大导致训练困难**，即学习一个参数来降低输出值的方差，类似于降噪。

上述过程的示例代码如下：

```python {linenos=table, linestart=0}
from torch import nn

class ResBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_c, out_c, stride, downsample=False, reduce_first=1, ks=3, padding=1, **kwargs):
        first_planes = out_c // reduce_first

        self.conv1 = nn.Conv2d(in_c, first_planes, ks, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm2d(first_planes)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(first_planes, out_c, ks, stride, padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.act2 = nn.ReLU()

        ## for downsample & double channel number
        self.downsample = None
        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_c, out_c, ks, stride, padding, bias=False),        # 通常这里的 ks = 1, senet 里 ks = 3
                nn.BatchNorm2d(out_c)
            )
    
    def forward(self, x):
        feat = self.conv1(x)
        feat = self.bn1(feat)
        feat = self.act1(feat)
        feat = self.conv2(feat)
        feat = self.bn2(feat)
        ## add se block here
        # feat = self.se(feat)

        if self.downsample is not None:
            x = self.downsample(x)
        feat += x
        feat = self.act2(feat)

        return feat
```

注意在Stem模块中，不使用残差模块，并且通过 `stride=2` 以及一个 `MaxPooling(stride=2)` 来将输入图片下采样4倍。

## Bottleneck 结构

[第一小节](#基础)里提到的结构更多的是用于 resnet-18/34等浅层网络，为了构建深层网络（resnet-50/101-152）等，作者提出了 Bottleneck 模块。Bottleneck 模块包含三层卷机，分别是 `conv1x1, conv3x3, conv1x1`，并且第一个 `conv1x1`将输入数据的channel根据一个因子（通常是4）进行缩小，最后一个`conv1x1`在缩放回原来大小，这样既可以完成残差计算，也降低了中间`conv3x3`的计算。实验表明，这里即使不降低 channel 个数也不会影响性能，所以Bottleneck 完全为了实际中提高计算效率，至于 Mobilenetv2 里提到的 Inverted Residual Block，不会展开。网络结构示意图如图2右边部分。

实现的时候需要注意的是，每个 Stage 里Block内的Channel变化过程，最后一个`conv1x1`的是第一个`conv1x1`**输入**的expansion倍。下图展示的其实是Stage内非第一个Block的结构，相较于输入，第一个`conv1x1`将channel数降低了4倍；而第一个Block的输入channel数时上一Stage输出的channel数，配合channel double的过程，第一个`conv1x1`只是将channel下降了2；此外，下采样部分是在 `conv3x3, stride=2` 部分完成的，如果放在第一个`conv1x1`里，会导致3/4的信息丢失。

![图-2 Bottleneck 结构](/imgs/resnet-series/residual1.png)

具体实现代码如下。

```python {linenos=table, linenostart=0}
from torch import nn
class BottleneckBlock(nn.Module):
    def __init__(self, in_c=256, out_c=64, expansion=4, stride=1, downsample=False, **kwargs):
        super().__init__(**kwargs)
        out_planes = out_c * expansion

        self.conv1 = nn.Conv2d(in_c, out_c, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(out_c, out_c, 3, stride, 1, bias=False)      # stride = 2 时进行下采样
        self.bn2 = nn.BatchNorm2d(out_c)
        self.act1 = nn.ReLU()

        self.conv3 = nn.Conv2d(out_c, out_planes, 1, 1, bias=False)         # 注意输出 channel 的个数
        self.bn3 = nn.BatchNorm2d(out_c)
        self.act3 = nn.ReLU()

        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_c, out_planes, 1, 1, bias=False),
                nn.BatchNorm2d(out_planes)
            )
        else:
            self.downsample = False
    
    def forward(self, x):
        feat = self.conv1(x)
        feat = self.bn1(feat)
        feat = self.act1(feat)

        feat = self.conv2(feat)
        feat = self.bn2(feat)
        feat = self.act2(feat)
        ## use avg pooling to downsample here

        feat = self.conv3(feat)
        feat = self.bn3(feat)
        ## add se here
        # feat = self.se(feat)
        ## drop path here, i.e. random drop some samples along batch axis
        ## downsample projection path here
        if self.downsample is not None:
            x = self.downsample(x)
        feat += x
        feat = self.act3(feat)
        return feat
```

总结一下SE的位置，SE Block 均是在每一个 Block 最后一层卷积 BN 之后的特征上进行。

## ResNet v2

论文地址：[Identity Mappings in Deep Residual Networks](https://arxiv.org/pdf/1603.05027.pdf)

论文里其实是对 BN / ReLU 两个函数的位置进行了挪动。作者测试了下面几种排列组合，发现最后一种实现效果最好。

![图-3 ResNet v2改进](/imgs/resnet-series/residual2.png)

分析一下，(b)里 BN 在 Identity (左侧)分支里，会改变Identity分支的分布，影响信息传递，在训练开始的时候会阻碍Loss的下降。这一点可以通过论文里的梯度反向传播推导过程看出来。

(c)里residual(右侧)分支是 ReLU 的输出，导致这个分支对结果只有正向影响，毕竟非负，但我们希望有两个方向的影响，所以非最优。关于(d, e)，实验表明都不如(f)，毕竟 BN 在Residual分支上可以对输入就起到正则化的作用。

## ResNeXt

网络结构如图4。

![图-4 ResNeXt网络结构](/imgs/resnet-series/residual3.png)

(a)为最开始的思想，(c)为等价形式。也就是说，中间的`conv3x3`替换为分组卷积计算。
主要改动就是将普通残差结构中的 Residual 分支用 Inception 思想进行修改，用多路并行卷积代替原来的一支卷积，与Inception论文不同的是，这里每个分支采用相同的参数配置，如kernel size等。

## 其它

* ResNeSt

  与[SKNet](https://arxiv.org/abs/1903.06586)类似。

* Res2Net

  在单个残差块内引入Inception思想，感受野逐步增大，最后concatenate 起来送入 `conv1x1` 计算。

* SKNet
