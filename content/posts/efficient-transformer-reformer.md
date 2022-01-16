---
title: "Efficient Transformer 系列之 Reformer"
date: 2021-09-06T10:50:19+08:00
draft: true

mathjax: true

excerpt_separator: <!--more-->
---
Reformer论文。<!--more-->

## 概览

Reformer论文的信息量还是有点大，主要创新点在于下面几个：

* 引入了新的 LSH Attention

## ANN

ANN (Aproximate Nearest Neighbor, 近似最近邻搜索)。常见的ANN算法可以分为三类：

* LSH(Locality Sensitive Hashing)
* 树方法（如HNSW）
* Product Quantization

## LSH Attention

对比普通的 Self-Attention (如下式)，新的LSH Attention改动如下。

$$Attention(Q, K, V) = Softmax(\frac{QK^T}{\sqrt{d_k}}) \times V$$
