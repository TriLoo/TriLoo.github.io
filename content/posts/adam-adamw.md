---
title: "从Adam到AdamW"
date: 2021-09-03T20:30:59+08:00
draft: true

excerpt_separator: <!--more-->
---
Adam算法的实现以及一个主要改进AdamW的原理与实现。<!--more-->

突然觉着NCHW尺寸的张量比 NHWC Layout 的张量更容易理解，因为后者来看，就是N个样本，每个样本 H * W 的空间维度，然后每个空间元素点是一个 C 维的特征向量。NCHW Layout 的话就需要从后向前理解了...
