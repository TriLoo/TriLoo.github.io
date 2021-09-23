---
title: "Model Visualization"
date: 2021-09-11T15:16:07+08:00
draft: false
mathjax: true

excerpt_separator: <!--more-->
---
深度学习中的一些可视化技术。<!--more-->

主要包括CAM、t-SNE两个方面，CAM又包括Gradient Free 以及 Gradient Based 两种实现思路；t-SNE更多的是用于高维空间在低维空间的可视化。

## CAM

再记录GradCam之前，可以先看下Cam算法的实现。参考博客是：[万字长文：特征可视化技术(CAM)](https://zhuanlan.zhihu.com/p/269702192)。

### CAM基础

全称 Class Activation Mapping。也就是获取每个类别在Feature Map上关注点的分布，比如利用最后一层CNN的Feature Map，将所有的Channel加权融合为一个二维图片，然后这个二维图片就被认作激活图。以ResNet18为例，最后一层CNN的Feature Map包含512个Channel，如果单独可视化每个通道，则比较难理解，所以CAM会根据每个通道不同的贡献大小对所有的通道进行加权融合获取一张CAM。

效果如图-1。

![图 - 1 CAM结果示意图](/imgs/model-visualization/cam0.png)

CAM实现的步骤如下：

1. 提取需要可视化的特征层，例如尺寸为7 * 7 * 512的张量
2. 获取该张量的每个channel的权重，即长度为512的向量
3. 对 Step1 中的张量按照 Step2 中的权重进行加权，获取尺寸为 7 * 7 的Map
4. 对该Map进行归一化，并通过插值的方式

上面提到，CAM可以分为Gradient Free / Gradient Based两种方式，两者的主要区别在于Step 2中计算Channel的权重方式不同，后者会利用梯度信息，前者不需要。

Gradient Based常见的算法包括

* Grad CAM (2016.10)
* Grad CAM ++ (2017.10)
* Smooth Grad-CAM++ (2019.08)

Gradient Free常见的算法包括

* CAM (2015.12)
* score-CAM (2019.10)
* ss-CAM (2020.06)
* Ablation-CAM (2020)

### 利用GAP获取CAM

属于Gradient Free类算法。

CAM的实现依赖于CNN卷积之后使用Global Average Pooling (GAP) 来实现；也就是说网络结构具有如下特征，经过若干层CNN得到Feature Map，然后利用GAP来压缩空间维度，然后压缩后的向量利用一个线性变换得到对应的类别预测。然后CAM就利用最后一个线性变换的权重作为每个类别的Channel的权重进行加权。示意图如图2所示，对于Australian terrier类的全连接权重为 $w_1, \ldots, w_n$。

![图 - 2 CAM实现示意图](/imgs/model-visualization/cam1.png)

> Global average pooling outputs the spatial average of the feature map of each unit at the last convolutional layer. A weighted sum of these values is used to generate the final output. Similarly, we compute a weighted sum of the feature maps of the last convolutional layer to obtain our class activation maps.

利用数学公式说明就是，设$f_k(x, y)$表示最后一层CNN的Feature Map中第 k 个unit (channel)在空间位置(x, y)处的数值，则GAP的计算就是$\sum_{x, y}f_k(x, y)$。然后对于类别 c，线性变换得到输入 Softmax 的数值，也就是 $S_c = \sum_k w_k^c F_k$，其中$w_k^c$也就是第 k channel 对类别 c 分类的重要度。

然后，论文定义class activation map ($M_c$) 为：

$$M_c(x, y) = \sum_k w_k^c f_k(x, y)$$

CAM这种Gradient Free算法的定义就是上面公示了。

作者分析了为啥不使用Global Max Pooling (GMP)，而是使用GAP。简单来说，GAP会让模型学习Object边界内的所有点的信息，而GMP则可以让模型只关注Object内最有区分性的空间位置即可，有很大可能 Object 内的其它位置对结果没有影响，毕竟对Max Pooling的结果没有影响。

CAM的缺点就是要求模型输出层之前需要使用 GAP，如果模型默认不是这一计算，则还需要替换成 GAP 重新进行训练。实现代码示例如下。

```python {linenos=table linenostart=0}
# 获取全连接层的权重
self._fc_weights = self.model._modules.get(fc_layer).weight.data
# 获取目标类别的权重作为特征权重
weights=self._fc_weights[class_idx, :]
# 这里self.hook_a为最后一层特征图的输出
batch_cams = (weights.unsqueeze(-1).unsqueeze(-1) * 
                self.hook_a.squeeze(0)).sum(dim=0)
# relu操作,去除负值
batch_cams = F.relu(batch_cams, inplace=True)
# 归一化操作
batch_cams = self._normalize(batch_cams)
```

### Grad CAM

此算法可以克服上面CAM算法中要求模型必须包含GAP的存在才行。基本思路是用梯度来计算Channel加权的权重，而且用ReLU去除权重为负数的那些channel。Grad CAM支持对任意一层CNN的 Feature Map 进行CAM可视化。

首先计算类别 c 的得分(score, before the softmax)对CNN层$A^k$的梯度，也就是$\frac{\partial y^c}{\partial A^k}$，然后这些题都在 $A^k$ 的空间维度上进行求平均，也就是 GAP（作者发现 GAP 好于 GMP），然后作为权重对$A^k$的channel进行加权，也就是

$$\alpha_k^c = \overbrace{\frac{1}{Z}\sum_i \sum_j}^\text{global average pooling} \underbrace{\frac{\partial y^c}{\partial A_{ij}^k}}_\text{gradients via backprop}$$

实验表明，越是浅层的CNN，Grad CAM的效果越差，因为这些层的感受野也越小。后就是ReLU的使用保留那些只起正向作用的空间点：

$$L_{\text{Grad-CAM}}^c = ReLU(\sum_k \alpha_k^c A^k)$$

注意，$y^c$在这里不一定必须是分类模型的class score，可以是任何可微分的激活函数的输出。

![图 - 3 Grad-CAM实现示意图](/imgs/model-visualization/cam2.png)

示意图可以看出，Grad CAM支持多种模型的输出层结构以及对应的任务。为了实现更高分辨率的可视化，作者提出首先将$L_{\text{Grad-CAM}}$使用双线性插值进行上采样，然后与 Guided Backpropagation 的结果进行按元素乘的结果作为可视化结果。

另外一个方面，论文中提到使用Softmax之前的输入作为 $y^c$，但是某些代码中也使用了 Softmax 的输出作为 $y^c$，知乎文章里有给出为啥可能采用Softmax之前（论文中的做法）会效果更好。实现代码如下。

```python {linenos=table linenostart=0}
# 利用onehot的形式锁定目标类别
one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
one_hot[0][index] = 1
one_hot = torch.from_numpy(one_hot).requires_grad_(True) 
# 获取目标类别的输出,该值带有梯度链接关系,可进行求导操作
one_hot = torch.sum(one_hot * output)
self.model.zero_grad()
one_hot.backward(retain_graph=True) # backward 求导
# 获取对应特征层的梯度map
grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
target = features[-1].cpu().data.numpy()[0, :] # 获取目标特征输出
weights = np.mean(grads_val, axis=(2, 3))[0, :] # 利用GAP操作, 获取特征权重
cam = weights.dot(target.reshape((nc, h * w)))
# relu操作,去除负值, 并缩放到原图尺寸
cam = np.maximum(cam, 0)
cam = cv2.resize(cam, input.shape[2:])
# 归一化操作
batch_cams = self._normalize(batch_cams)
```

### Grad CAM++

Grad CAM++ 是对 Grad CAM 的进一步改进，优势在于定位更加精准，也更适用于图像中包含不止一个目标类别物体的情况。

具体来说，Grad CAM认为 Feature Map上每一个点的重要度是一样的（使用GAP得到），而Grad CAM++认为每个位置上点的贡献度不同，因此额外增加了一个权重用来表示Feature Map上每个元素的重要度。

![图 - 4 Grad-CAM++实现示意图](/imgs/model-visualization/cam3.png)

这里的重点也是权重$\alpha_{ij}^{kc}$的计算。

$$\alpha_{ij}^{kc} = \frac{(\frac{\partial S^c}{\partial A_{ij}^k})^2}{2(\frac{\partial S^c}{\partial A^k_{ij}})^2 + \sum_a \sum_b A_{ab}^k (\frac{\partial S^c}{\partial A^k_{ij}})^3}$$

知乎参考博客里还有其他一些比较新的Feature Map的可视化方法，可以去参考一下，这里略过。

## t-SNE

主要的参考是 [t-distributed stochastic neighbor embedding](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding)。

这个算法其实适用于展示高维数据的，也就是将高维数据降维到低维数据，然后方便查看两个数据点之间的距离。与其他降维算法（如PCA）相比，t-SNE创建一个缩小的特征空间，相似的样本由附近的点建模，不相似的样本由高概率的远点建模。最后一句话可以通过下面的例子进行说明。

整体来说，t-SNE是将高维数据映射到低维空间，所以就涉及到几个问题：这个映射过程怎么确定，等价于低维空间的点应该怎么选择；然后就是维度灾难，也就是原始高维空间的数据计算距离的时候会受维度灾难的影响。

t-SNE其实更是为了更好的可视化相似性，而不是为了降维。整体思路也是保证映射前后，也就是高维空间、低维空间中，相似的点继续保持相似，也就是距离比较近。采取的方法是对两个空间中的点的距离进行概率建模，也就是一个点与其它所有点的距离映射成一个概率分布。首先是高维空间，将欧氏距离映射成高斯分布。

N个高维空间点，$x_1, \ldots, x_N$，计算概率$p_{ij}$为与点$x_i, x_j$之间的欧式距离成正比。

$$p_{j|i} = \frac{\exp{(-\parallel x_i - x_j\parallel^2} / 2\sigma_i^2)}{\sum_{k\neq i}\exp(-\parallel x_i - x_k \parallel^2 / 2\sigma_i^2)}$$

其中，$j \neq i$，并且有$p_{i|i}=0$以及$\sum_j p_{j|i}=1, \forall i$。这个式子是指以$x_i$为中心点的概率分布中，点$x_j$的概率。由于距离是对称的，所以定义：

$$p_{ij} = \frac{p_{j|i} + p_{i|j}}{2N}$$

此时也就有$p_{ij}=p_{ji}, p_{ii}=0, \sum_{i,j}p_{ij}=1$了。

其中参数$\sigma_i$是由高斯分布的perplexity(困惑度)与二分法（bisection method）计算得到的困惑度相等来得到的。就体现在，越密集的数据空间中，$\sigma_i$越小，也就是低困惑度更关注局部数据点，高困惑度更关注全局结构。

这里需要注意的地方在于，$x_i$在高维空间中会面临维度灾难，此时欧氏距离没有区分性，也就是任何两个点之间看上去都不是那么相似了，毕竟高维空间里这些点还是非常稀疏的，也就导致计算的概率分布$p_{ij}$也区分性不大，针对这个问题，有一些方式是基于 [intrinsic dimension](https://en.wikipedia.org/wiki/Intrinsic_dimension) 的 [power transform](https://en.wikipedia.org/wiki/Power_transform) 来缓解这个问题。

回到上文，t-SNE的目标也是获得对应的到d维空间的映射，$y_1, \ldots, y_N, y \in \mathbb{R}^d$，同时可以反映出高维空间的概率分布$p_{ij}$。具体做法是，计算低维空间里点的概率分布，用$q_{ij}$表示。

$$q_{ij} = \frac{(1 + \parallel y_i - y_j \parallel^2)a^{-1}}{\sum_k \sum_{l\neq k}(1 + \parallel y_k - y_l \parallel^2)^{-1}}$$

这里也有$q_{ii}=0$。而且不再使用高斯分布计算概率了，而是选择student t-distribution来计算概率分布，好处是这个分布相比高斯分布，是胖尾的，也就可以将不相似的两个点用更远的低维映射点进行拟合。

优化映射过程是基于KL散度这个损失函数来实现的，高维、低维空间的分布分别用$P, Q$表示。

$$\mathrm{KL}(P\parallel Q) = \sum_{i\neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}$$

通过优化上述损失函数，利用梯度下降法来更新$y_i$的取值。优化的结果也就可以反映高维空间中点之间的相似度了。

下面是一个用在多模态预训练中文本 embedding 矩阵使用 t-SNE 可视化的结果，直观来看分布是比较均匀的。

![图 - 5 embedding 矩阵的t-SNE可视化](/imgs/model-visualization/txt_embedding_0.png)

对应PCA降维之后可视化的结果如下，就不是这么均匀了。

![图 - 6 embedding 矩阵的PCA可视化](/imgs/model-visualization/txt_embedding_1.png)
