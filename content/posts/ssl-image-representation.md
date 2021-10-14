---
title: "图像表征算法中的自监督学习方法"
date: 2021-08-31T14:15:48+08:00
draft: true
---

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

## SimSiam

SimSiam 的 Prediction Head 需要固定 Learning Rate，也就是不随 Scheduler 变化。

## 消融实验

## 一些 Tricks

* Rethinking Image Mixture for Unsupervised Visual Representation Learning

  在无监督训练中引入了Image Mixture & Label Smooth。

* Whitening for Self-Supervised Representation Learning
* Barlow Twins: Self-Supervised Learning via Redundancy Reduction
* Contrastive Multiview Coding 
