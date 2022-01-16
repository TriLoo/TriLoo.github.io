---
title: "大Batchsize训练用到的优化器"
date: 2021-08-31T14:09:48+08:00
draft: true

tags : [
    "Big Batch Size",
    "Optimization",
    "Train"
]

excerpt_separator: <!--more-->
---

一些为了适应大的Batch Size训练的优化算法。<!--more-->

偶尔就有时间看到有标题：训练 XX 模型只用 ** 秒 / 分钟，当然除了钱多卡多，在算法实现方面也看看有什么不同。

有个概念是 cross-view prediction (Hinto92年的一篇论文)，现在很多自监督模型都是在 representation space 完成这个预测任务，但是这容易导致模型坍塌，所以 BYOL 增加了一个 Prediction Layer，这也是相对于 Mean Teacher 这篇论文最大的改进了。对比学习的做法相当于学习当前图片的一个augmentation与其他图片的augmentation之间的区分。

主要包括：LARS, LAMB 等。

## 基础：Adam

## LARS算法

## LAMB算法
