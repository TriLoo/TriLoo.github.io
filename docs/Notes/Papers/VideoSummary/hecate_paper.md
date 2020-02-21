---
title: To Click or Not To Click
---

# To Click or Not To Click 论文阅读

## 简介

* 定义Thumbnails的两个条件
  * high relevance to video content 
  * superior aesthetic quality

针对上面的两个条件，分别做如下处理

* select *attractive* thumbnails by analyzing various visual quality and aesthetic metrics of video frames
* perform *clustring* analysis to determine the relevance to video content

上面也是本文的主要算法方面的工作内容。

## 相关工作

这里近关注 Video Summary相关的工作，关于 Thumbnail selection & Computational aesthetics两个方面的相关工作可以参考论文

* 直接去除内容中的冗余信息来实现 Summary，典型的两篇文章是：Towards scalable summarization of consumer video & Quasi real-time summarizatioin for consumer video，这两篇文章都是通过学习一个字典来实现去除冗余信息，即通过最小的一个字典可以最大程度的重建原来的视频
* TVSum: Summarizing web various using titles & Video co-summarization: Video summarization by visual co-occurrence & Video2GIF: Automatic Generation of Animated GIFs from Video 三篇论文里面的方法都需要借助额外的信息，如title，mannual animated gif 等数据来训练网络实现Summary，本文直接通过`photographic attractiveness`来实现视频帧的选取

## 提出的方法

首先来看算法的整体流程：

<div align="center"><img src="/note_imgs/sourcecode_imgs/hecate/hecate_diagram_0.png" width="484", height="404" alt="hecate schematic diagram"></div>

整个算法流程分为了三步，即frame filtering, keyframe extraction, thumbnail selection。

### Frame Filtering

好的 thumbnails 应该具有一下几个品质：

* brightness
* sharpness
* colorfulness

所以去除不具备这些特点的帧。

**Low-quality frames**

* brightness小于某个阈值
  $\text{Luminance}(I_{rgb})=0.2126I_r + 0.7152I_g + 0.0722I_b$
* sharpness小于某个阈值
  $\text{Sharpness}(I_{gray})=\sqrt{\Delta_xI_{gray}^2 + \Delta_xI_{gray}^2}$
* colorfulness小于某个阈值
  $\text{Uniformity}I_{gray}=\int_0^{5\%}cdf(sort(hist(I_{gray})))dp$

**Transition frames**

这一类帧数据的定义是位于两个`shot`变换之间，如包含淡入淡出、溶解等转场效果的帧。具体使用的算法是 Edge Change Ratio，论文是 A feature-based algorithm for detecting and classifying production effects.

**具体表现**

经过上述两个步骤的预处理，所话费的时间是视频长度的0.3%，但可以去掉11.36%的帧，即保留88.64%的有效帧。

### Keyframe Extraction

普通的做法就是基于帧特征对所有的帧进行聚类，然后将距离每个聚类中心的帧作为关键帧。

本文中的做法是使用帧图像的`aesthetics`来从每个聚类选取一个帧，尤其是最静止的那个帧。这是因为，大量观察发现，那些因运动补偿而比较模糊的帧都不美观，相反的，motion energy更小的帧视觉体验更好，因此作为关键帧。motion energy的定义见下面的`stillness metric`。

**Feature Extraction**

即用于帧聚类所使用的帧特征，这一步的目的是去除相邻的比较相似重复帧，所以选择最简单的color & edge histogram特征就可以满足需要，具体使用下面两种特征。

* pyramid of hsv histogram with 128 bins per channel
* pyramid of edge orientation and magnitudes with 30 bins for each

选择的尺度层级数为2，共5个区域，所以每一帧的特征维度是2220。

**Subshot Identification**

因为后面都是基于这一步选择的关键帧进行处理，所以这一步最好是采样更多的关键帧。实现方法就是对剩余的帧（是指去掉shot连接处的帧 + 低质帧，重复帧是在这一步结束后才去掉的）进行k-means聚类，聚类中心的个数就是上一步`shots`的个数，然后每个`shots`中属于不同聚类中心的帧作为一个subshots。

**The Stillness metric**

对subshot中的每一帧计算 `stillness`，即与前后两帧的sum-squared pixel-wise frame difference。

**Keyframe Extraction**

将每个subshot中`stillness metric`最大的那个帧作为关键帧。试验结果表明，这种方法就可以获得比较好看的帧图像。

### Thumbnail Selection

这一步就是从上一步得到的关键帧里面选取一个封面图，使用的指标有两个：

* relevance
* attractiveness

**Frame relevance**

即考察当前的帧与视频内容的一致性有多高。

具体做法如下。对所有的关键帧进行聚类，然后根据类别内样本的个数来衡量内容的相关性。聚类算法使用Estimating the number of clusters in a data set via the gap statics来求解最优的聚类中心个数。

> The gap statistic method does so by comparing the change in within-cluster dispersion with that expected under an appropriate reference null distribution.

这一步的结果是，每个聚类中心选择一幅图像，然后根据`aesthetic score`与`cluster size`来计算得分并排序。

**Frame Attrativeness**

一种方法是使用`stillness`来作为指标，选择得分最高的那一帧；另一种有监督的方法是通过学习一个模型来实现美学打分。这里仅以第一种方法进行整理。

本文设计了一种包含52种图像性质的评价指标，主要包括：

1. Color
  - Contrast
  - average hsv, centra average hsv, hsv color histogram, pleasure, arousal, dominance
  - hsv contrast
2. Texture
  - entropy
  - energy
  - homogeneity
  - contrast of the GLCM
3. Quality
  - contrast balance
  - exposure balance
  - jpeg quality
  - sharpness
4. Composition
  - rule of thirds
  - symmetry
  - uniquness

所有的用于美学评估的特征如下图所以。
<div align="center"><img src="/note_imgs/sourcecode_imgs/hecate/hecate_aesthetic_feats_0.png" width="486", height="477" alt="hecate schematic diagram"></div>

## 试验过程与分析

这部分可参考原文。

## 总结

* 好的封面图与帧图像的客观质量指标有很大的关联性，包括Color, Texture, Composition, Basic Quality等；
* 后续改进可以将一些辅助数据引入，如titles, search queries, mouse tracking signals等；
* 还有一点就是这个领域还需要客观的评价指标！
