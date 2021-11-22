---
title: "图像表征算法中的自监督学习方法"
date: 2021-08-31T14:15:48+08:00
draft: false
mathjax: true

excerpt_separator: <!--more-->
---

经典自监督模型，包括MoCo / SimCLR / SwAV / BYOL / SimSiam 等。<!--more-->

主要关注无监督策略的研究，模型结构不是本文重点，所以主要包括 MoCo 系列、SimCLR 系列、SwAV、BYOL 等几篇论文。无监督训练模型的一点在于要避免模型坍塌，通过 Contrastive Loss, Clustering Constraints, Predictor(Stop Gradient), Batch Normalization等。

整体来说，图像自监督学习方法按照自监督实现思想可以分为下面几类。

* 基于contrastive loss
* 基于蒸馏的方式，一般设计 momentum
* 基于聚类的方法

关于预训练任务也存在多种选择。

* 预测图像选装方向。图像经过 0/90/180/270 等几个角度的随机旋转，然后训练模型进行4分类
* 预测图片不定位置相对关系。图像被分割成 3 * 3 的表格，然后选取中心小图与另外8个子图中的随机一个进行位置分类，分类类别为8（两个子图的输出拼接起来送入分类层），一些技巧是图像分割成子图时可以增加缝隙或者抖动等
* 补丁拼图。将图片分割成 3 * 3 的子图，然后随机打乱，将子图的所有输出特征拼接起来送入分类层，正常来说，类别说应该是 9!，但是作者对这些排列类别做了合并，因为很多排列比较相似，合并过程基于汉明距离进行
* 图片上色。灰度图片输入 Encoder，然后Decoder输出彩色图片，可以使用 L2 Loss，或者 LAB 颜色空间等
* 自编码器系列。
* GAN系列。
* 对比学习。需要构造丰富的负样本，比如大的 Batch Size 或者借助 Memory Bank等

## MoCo 系列

## SimCLR 系列

## SwAV

## BYOL

分析了怎么防止模型坍塌（也就是所有的输入的模型输出都是相同的），关键是要让模型的输出部分层学习到新的知识。在这里，一方面是借助 Mean Teacher，一方面是在 Student Network 上面增加了一层 Predictor，这两个因素可以让 Prediction 层不断学习新的知识，从而避免模型坍塌。BYOL包含两个模型，一个称为 Online，一个称为 Target。

按理来说，没有负样本，那么优化损失函数的梯度$\nabla_{\theta}(\mathcal{L}_{\theta, \epsilon}^{\mathrm{BYOL}})$应该很快导致模型坍塌啊，也就是损失降为0，但实际没有发生，作者认为这是因为这个损失的梯度下降方向与 Target 模型参数的变化方向是不一致的，也就是梯度下降方向 与 Target 模型让 Online 模型参数更新的放向不一样，所以避免了模型坍塌。另一方面，这也就意味着不存在一个Loss可以同时优化 Target / Online 模型的权重，类似于 GAN 模型的G / D的参数无法同时优化一样。作者也用消融实验表明，保持 prediction 足够好貌似是防止坍塌的关键。

为啥 SimCLR 依赖于 color jitter 这个变换，因为如果去掉这个变换的话，两次 crop 的图像的颜色直方图分布其实是非常接近的，导致模型非常容易学习。

发现去掉 Weight Decay 后，模型发散，说明 WD 对自监督模型的重要性，但是增加模型初始化时的初始值范围对模型性能影响不大。

## SimSiam

SimSiam 的 Prediction Head 需要固定 Learning Rate，也就是不随 Scheduler 变化。

## 消融实验

## 一些 Tricks

* Rethinking Image Mixture for Unsupervised Visual Representation Learning

  在无监督训练中引入了Image Mixture & Label Smooth。

* Whitening for Self-Supervised Representation Learning
* Barlow Twins: Self-Supervised Learning via Redundancy Reduction
* Contrastive Multiview Coding 
