---
title: "基于Transformer结构的图像自监督模型及训练"
date: 2021-09-28T14:07:50+08:00

draft: false
mathjax: true

excerpt_separator: <!--more-->
---
一些基于Transformer结构的图像自监督模型以及训练过程中遇到的问题。<!--more-->

这篇文章里除了给出 DINO / MoCoV3 的具体结构以及对应的自监督训练思想，还给出了两种position embedding的具体实现。

## 概览

目前出现了一系列的图像表征模型，包括 MoCo 系列、SimCLR系列以及BYOL/SwAV/SimSiam等，本文主要关注基于 Transformer 结构的一些无监督训练模型的细节，主要包括 MoCo v3, DINO, MoBY等。

性能对比主要分为：Linear Acc 以及 End-2-End Fine Tuning Acc 两种方式。前者是无监督预训练之后，仅对最有一层分类输出层进行微调，后者是微调整个网络。

几个模型的主要指标对比如下，ViT-S 与 ResNet50 参数量（21M vs 23M(resnet50)）以及吞吐率（1007 vs 1237 img/sec）还有有监督训练精度（79.8(v) VS 79.3(r)）都类似。

Model | Pretrain Epochs | Linear Acc | E2E Acc | Notes
------ | ------ | ------ | ------ | ------
SimSiam, ResNet-50      | 100 | 68.1 | - 
SimSiam, ResNet-50      | 200 | 70.0 | - 
SimSiam, ResNet-50      | 400 | 70.8 | - 
SimSiam, ResNet-50      | 800 | 71.3 | - | 与 SimCLR/MoCoV2/SwAV 几乎差不多
MoCo V3, ResNet-50      | 100 | 68.9 | - 
MoCo V3, ResNet-50      | 300 | 72.8 | - 
MoCo V3, ViT Small      | 300 | 73.2 | 81.4
MoCo V3, ViT Base       | 300 | 76.7 | 83.2
MoCo V3, ViT Large      | 300 | 77.6 | 84.1
DINO, ResNet-50         | 300 | 74.5 | -
DINO, ResNet-50         | - | 75.3 | -
DINO, ViT Small         | 300 | 76.1 | -
DINO, ViT Small         | - | 77.0 | -
DINO, ViT Base/16       | - | 78.2 | -
DINO, ViT Base/8        | - | 80.1 | -
MoBY, DEIT-S            | 300 | 72.8 | -
MoBY, DEIT-S(multi crop)| 300 | 75.9 | -
MoBY, Swin-T            | 100 | 70.9 | -
MoBY, Swin-T            | 300 | 75.0 | -

所以，在 epoch = 300 这个门槛上，最好的模型以及对应无监督训练策略应该是 DINO + ViT Base/8。在 ResNet50 / 300 epochs 上，最好的无监督训练模型也是DINO的74.5；在 ViT-S / 300 epochs 上，最好的当属 DINO 的76.1。而且 ViT B/8 比 ViT B/16 效果更好。

MoCo V3 以及 DINO 以及 MoBY 都借助了 Momentum Update 的技巧来防治模型坍塌。

## MoCo v3

主要思想是基于 Contrastive Loss 进行训练。

数据增广方式由两种Augmentation List构成(采取BYOL论文的做法)，第一种与 SimSiam 类似，第二种Aug List 里新增了BYOL中的 Solarization 增广方式。效果体现在对输入图片的两个Crop分别进行增广，因为`RandomResizedCrop()`函数的实现会先 Crop，然后在Resize 到指定大小。具体的增广参考[moco-v3](https://github.com/facebookresearch/moco-v3)里的代码。

MoCo V3 丢弃了 MoCo V1/V2 里采用的 Memory Queue 来构造负样本，与 SimCLR 的观察类似，也是通过大的 Batch Size 来保证负样本的数量；另一方面，模型采用了EMA思路，也就是有两个 Encoder 网络$f_q, f_k$，其中 $f_k = m * f_k + (1 - m) * f_q$。采用的 Loss 是 InfoNCE，即：

$$\mathcal{L}_q = - \log \frac{\exp(q \cdot k^+ / \tau)}{\exp(q \cdot k^+ / \tau) + \sum_k \exp(q \cdot k^- / \tau) }$$

完整的算法实现见图-1。

![图 - 1 MoCo v3 Algorithm](/imgs/img-transform-ssl/mocov3-0.png)

值得注意的地方在于，Predictior 仅用与计算 Q，K 是直接 $f_k$ 模型的输出结果；Loss的最终计算会乘上一个因子 $2 * \tau$，默认的$\tau=0.2$；在训练过程中，算法中的$m$也随着 epoch 的增加而 cosine 下降；类似 SimCLR 新增了 3 层的 Mlp 层，具体维度的变化可以参考代码；其它具体实现可以参考上面链接中的代码。两个细节，首先是计算 K 的时候使用的更新后的 $f_k$，其次是没有像 SimSiam 中那样使用 fix lr 技巧。

对于参数 $m$ 的选取，作者对比了 0, 0.9, 0.99, 0.999 等值，发现 m = 0.99 时效果最好，但是实际实现是按照下面一行代码进行更新，并且初始值是0.99。

```python {linenos=table linenostart=0}
def adjust_moco_momentum(epoch, args):
    """Adjust moco momentum based on current epoch"""
    m = 1. - 0.5 * (1. + math.cos(math.pi * epoch / args.epochs)) 
                        * (1. - args.moco_m)
    return m
```

对应的曲线如图-2，其中横轴为epoch的取值，纵轴是对应 epoch 的 momentum 参数。

![图 - 2 MoCo V3 中 Momentum 参数的变化趋势](/imgs/img-transform-ssl/mocov3-3.png)

与参数量差不多的 ResNet 模型相比，ViT 占有优势！

![图 - 3 不同 Backbone 在 MoCo V3 上的性能表现](/imgs/img-transform-ssl/mocov3-2.png)

### Sin-Cos位置编码

Sin-Cos位置编码基于下面的公式计算每个位置上的 position embedding。

$$PE(pos, 2i) = \sin \left( \frac{pos}{10000^{2i / d_{model}}} \right)$$
$$PE(pos, 2i + 1) = \cos \left( \frac{pos}{10000^{2i / d_{model}}} \right)$$

### 使用 ViT 时的训练稳定性

一般来说，可以直接将 ResNet50 替换成 ViT 模型进行训练，但是作者发现 Transformer 网络不太稳定。论文里作者给出了几个导致训练不稳定的原因以及可能的改正方法。

首先是不稳定因素。

* Batch Size

  实验发现，Batch Size 由 1K -> 2K 的时候，Linear Acc 是有提高的（71.5 -> 72.6）。但是当 Batch Size 继续增加到 4K 的时候，按理说可用负样本更多了，效果应该更好，但是实际是 Acc 下降到了 72.2，而且Trainging Curve 上也出现了很多的 Dips。当 Batch Size = 6K 的时候，现象更严重。可能的原因在于大Batch Size 训练时模型会跳出当前局部最优解，然后重新进行优化。

  ![图 - 4 训练稳定性与Batch Size的关系](/imgs/img-transform-ssl/mocov3-1.png)

* Learning Rate

  学习率也是一个因素，当学习率太小的时候，模型没法学到最优导致ACC降低，当学习率太大的时候，也会出现 Dip，从而训练不稳定导致ACC下降。文章里使用的 learning rate scale 规则是：$lr * \mathrm{BatchSize} / 256$，其中基础的 lr = 1.5e-4。

* Optimizer

  一般来说，大 Batch Size 训练需要使用专门的优化器，比如 LARS，以及 LAMB等。作者对比了 LAMB，发现效果与 AdamW 效果类似，但是对 lr 会更敏感，导致不好调参，所以还是使用 AdamW。

然后作者给出了一种解决办法：

* Random Patch Projection
  也就是将 Patch Embedding 层固定为随机初始化权重；并且发现使用 BN / Weight Norm 等方法也不如使用 Random Patch Projection训练更好。
* Long Warm-Up
  使用 40 epochs 的Warm Up，也有助于增加训练稳定性。

## DINO

主要思路是基于 Self Distillation 来实现。既然是类似于 Knowledge Distillation 的实现，那就必须要有一个 Teacher 模型、一个 Student 模型，如图-5所示。

![图 - 5 DINO网络结构示意图](/imgs/img-transform-ssl/dino-0.png)

在SSL中，Teacher模型的实现包含两个方面，首先是输入数据，然后是模型参数怎么更新。输入数据涉及到了同一幅图片的 Multi-Crops(Views)的实现。在 DINO 中，给定一幅图片 x，产生一个 view 集合，集合中包含两个 global view 以及多个（如6个）Local Views，Local View 具有更小的分辨率。所有的 Views 都会进行 student 网络的前向计算，但是只有两个 Global Views 才会进行 Teacher 网络的前向计算，所以在基于上述 Student / Teancher 模型进行知识蒸馏的时候可以获得 "Local-to-Global" 的对应。然后用下式对 Student 网络参数进行更新。

$$\min_{\theta_s} \sum_{x \in \{ x_1^g, x_2^g \}} \sum_{x'\in V, x' \neq x} H(P_t(x), P_s(x'))$$

其中，$H(a, b) = -a \log b$。Teacher / Student 两个网络具有相同的结构但是不同的参数。为了处理不同的输入分辨率（这里为224，96两种），代码中借助 XCiT 的思路进行实现。

与 Knowledge Distillation 不同的是，SSL 中没有Label 来训练 Teacher 模型，那么Teacher 模型的权重怎么更新呢？直观的方案有以下两种方案。

* 直接拷贝Student网络的参数
  * 直接将最新的Student网络的权重拷贝过来 - 不收敛
  * 拷贝上一次 Iteration 的权重 - 不收敛
  * 拷贝上一次 Epoch 的权重 - 66.6
* 利用 EMA 机制来更新 Teacher 模型的权重，也就是 Momentum Encoder - 72.8

  $$\theta_t \leftarrow \lambda \theta_t + (1 - \lambda)\theta_s$$

  其中，权重$\lambda$ 会按照 Cosine 的方式从0.996 上升到 1.0。这种方式的 Teacher 实现原理类似于model ensembling中的Polyak-Ruppert Averaging算法。

实际实验发现，直接拷贝最新的Student权重或者上一次 Iteration 的权重，都会导致模型坍塌；相比之下，拷贝上一Epoch 的 Student 的权重 或者基于 EMA 进行更新的方式得到的 Teacher 模型不会导致坍塌。

接下来就是具体的网络结构了。

DINO 框架下的模型包含两部分，一个是backbone $f$，一个是 projection head $h$，所以得到的特征提取函数是$g = h \circ f$，没有使用BYOL中的 Prediction Head，因为加上之后效果会下降(76.1 vs 75.6)。Projection Head 部分包含3 层的MLP，hidden size为2048 + l2 normalization，最后 3-layer MLP 的输出在接一个**weight normalized fully connected layer**，如图-6所示，最终输出维度是 K，如果backbone 是 ViT，那么MLP中也会去掉 BN，毕竟ViT中没有使用 BN。实验表明，如果不实用 l2 normalization 的话，MLP 中层数大于 2 层后就会出现模型坍塌，但是 层数为1、2层时不会出现这个问题，但是1 - 4 层来看，层数越多效果越好；对于输出维度 K 来说，小于等于65536时，越大模型效果越好；MLP中使用的激活函数GELU效果比 ReLU更好。

对应的 Projection Head 的实现示意图如图-6。

![图 - 6 DINO Projection Head 实现示意图](/imgs/img-transform-ssl/dino-2.png)

DINO模型使用Centering & Sharping 两种操作相互配合来防止模型坍塌。Sharping 的实现就是类似 Knowledge Distillation 中计算 Softmax 时，激活数值除以$\tau_s$（student），或者 Teacher 模型的 $\tau_t$，一般来说，这个参数越小，Softmax 的输出就越 sharp，但是这里只针对Teacher模型而言。即：

$$P_t(x) = \frac{\exp (g_{\theta_t} (x)^{(i)}/ \tau_t)}{\sum_{k=1}^K \exp (g_{\theta_t} (x)^{(k)} / \tau_t)}$$

其中$K$就是Student / Teacher Projection Head 部分的输出维度。对应的 Centering 的实现是引入了新的偏置项用于Teacher 模型的输出：

$$g_t (x) \leftarrow g_t(x) + c$$

然后这个偏置项会根据Teacher模型的输出进行移动平均更新。

$$c \leftarrow m * c + (1 - m) \frac{1}{B} \sum_{i=1}^B g_{\theta_t} (x_i)$$

对应的伪代码实现如下。

![图 - 7 DINO伪代码实现](/imgs/img-transform-ssl/dino-1.png)

### Position Embedding 的生成

与 MoCoV3 中使用 sin-cos 的位置编码不同，DINO 针对不同的分辨率采用的是对 position embedding 利用 bicubic 的方式进行上/下采样。

具体实现代码如下，可以看出主要是将 Patch 恢复成 $(B, C, H, W)$ 格式，然后借助`torch.nn.functional.interpolate()`函数进行下采样。

```python {linenos=table linenostart=0}
def interpolate_pos_encoding(self, x, w, h):
    npatch = x.shape[1] - 1
    N = self.pos_embed.shape[1] - 1
    if npatch == N and w == h:
        return self.pos_embed
    class_pos_embed = self.pos_embed[:, 0]
    patch_pos_embed = self.pos_embed[:, 1:]
    dim = x.shape[-1]
    w0 = w // self.patch_embed.patch_size
    h0 = h // self.patch_embed.patch_size
    # we add a small number to avoid floating point error in the interpolation
    # see discussion at https://github.com/facebookresearch/dino/issues/8
    w0, h0 = w0 + 0.1, h0 + 0.1
    patch_pos_embed = nn.functional.interpolate(
        patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
        scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
        mode='bicubic',
    )
    assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
```

其中，$-1$是为了去掉增加的 `[CLS]` 这个 Token，然后将 `patch_pos_embed` 恢复空间维度，利用`interpolate()`函数进行下采样，最后恢复成$(B, SeqLen, Dim)$的格式与CLS的embedding拼接起来做为返回。

### 最后一层的处理

DINO模型对最后一层全连接的处理，包括两个方面，首先是使用了 weighted normalized 的全连阶层，其次是在第一个 epoch 的时候，会固定 last layer 的参数，不进行更新。

Weighted Normalized的全连接的实现。对应的函数是`torch.nn.utils.weight_norm()`，对应的论文是[Weight Normalization](https://arxiv.org/abs/1602.07868)，中文博客可以参考：[模型优化之Weight Normalization - 知乎](https://zhuanlan.zhihu.com/p/55102378)。

主要是思想是将全连阶层的权重分离为大小、方向两个部分，然后使用 SGD 分别优化这两个部分。

$$w = g \frac{\mathbf{v}}{\parallel \mathbf{v} \parallel}$$

其中$\mathbf{v}$为表示方向的向量。然后SGD优化的时候，分别优化$g$与$\mathbf{v}$。

另一个处理是第一个 epoch 内取消最后一层的梯度更新，实现代码其实就是设置`grad=None`，即。

```python {linenos=table linenostart=0}
def cancel_gradients_last_layer(epoch, model, freeze_last_layer=1):
    if epoch >= freeze_last_layer:
        return
    for n, p in model.named_parameters():
        if "last_layer" in n:
            p.grad = None
```

### DINOLoss的实现

论文中提到的 Centering & Sharping 两个做法都体现在这个类里；而且包括对应 Sharping 的参数的调整（前30个epoch由0.04 -> 0.07，即`warmup_teacher_temp_epochs=30`）提高训练稳定性。

对应的实现代码如下。

```python {linenos=table linenostart=0}
class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops        # 2 (global crops) + local crop nums
        # 注意 Center 尺寸，用于Centering
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        # Teacher 模型的 Sharping 参数的更新
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        # 分成不同的chunk，一共 ncrops 份，每份对应一个 crops
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        ## 论文里提到的思路在这里
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        ## 收集teacher 模型的输出并计算平均值
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
```

### DINO实现中的其它一些细节

* 使用Pre-norm的实现。由于输入的 Crops 之间尺寸不一致（224 与 96），所以需要注意`MultiCropWrapper`类的实现以及`DINOLoss`损失函数的实现
* learning rate scaling rule: $lr = 0.0005 * batchsize / 256$；然后前10 epoch 进行线性warmup，之后按照 cosine annealing进行下降
* wd参数会按照 cosine 过程从0.04 增加到0.4
* Centering 中的$m=0.9$时效果最好，太大如0.999的话就会坍塌
* $\tau_s=0.1$，但是Sharping中的$\tau_t$在前30个epoch 会由0.04线性增加到0.07；这个设置时对比了几种$\tau_t$取值的结果得到的
* 数据增广方式与 BYOL 相同，也就是 Asymmetric Data Augmentation的方式
* 随着 patch 大小的下降，模型效果会上升，如5 * 5 > 8 * 8 > 16 * 16，但是模型计算耗时会增加
* 对Batch Size不是非常敏感，1024时效果最好(59.9)，512时为59.6，256时为59.1，128时为57.9
* 作者对 Teacher 模型的输出对比了三种输出处理方式
  * Centering（本文默认做法）
  * Sinkhorn-Knopp
  * Softmax
  * 发现，当使用 Centering 配合 Momentum Teacher Update from Student 时效果最好；Momentum 的效果都大幅优于不使用 Momentum 的训练效果
* Multi-Crop的数据增广方式对DINO模型的帮助最大
* 作者尝试了集中 `[CLS]` 的使用方式
  * 取后面$l$层的`[CLS]`拼接起来，ViT-S模型取最后4层的`[CLS]`时效果最好(77.0 VS 76.1)；但是对 ViT-B模型而言，这种拼接方式没有帮助，而取`[CLS]`的输出与其它Tokens的平均拼接起来效果最好（78.2 vs 78.0）
* 在第一个 Epoch 训练过程中，固定 Last Layer 的参数，具体参考`cancel_gradients_last_layer()`函数

* BYOL中基于Predictor & BN来对抗模型坍塌，但是 DINO 使用的是Teacher Output Centering来实现，而且与 Sharping 结合起来效果才是最好。这里采用Momentum的方式来更新 Teacher 模型，如果不采用这种方式，另一种常见的方式是拷贝 Student 的权重 + Stop Gradient来避免模型坍塌。
* 与 MoCo V3中的发现类似，ViT-S 相比于 ResNet50 具有更大的潜力，尤其是 k-NN 指标下，两者差距达到 14%。
* 类似于 Mean Teacher，不论是 ViT / ResNet50，在训练过程中，Teacher 模型的效果都优于 Student 模型的效果。
* 论文里贴出来了一些 Attention Map，具体实现是取出最后一层的 Attention 矩阵，尺寸是 $(B, HeadNum, 1 + SeqLen, 1 + SeqLen)$，其中 1 表示新增加的`[CLS]` Token，然后就是

### 与 MoCo V3 的异同

相同点。

* 都包含两个Encoder，MoCo V3中称为Base/Target Encoder，DINO中称为 Student / Teacher 模型
* 两个 Encoder 之间都是通过 Momentum 的方式由一个 Encoder 的权重来更新另一个Encoder 的权重，并且这个 m 参数都按照 cosine 的方式增加到 1.0

不同点。

* MoCo V3中，两个 Encoder 结构不一样，主要在于 Base Encoder 多了一个 Prediction Head；而DINO中两个模型的结构是完全相同的，都不包含 Prediction Head
* DINO 计算 Softmax 的$\tau$在两个 Encoder 中是不同的，使用的Loss也不同
* DINO 使用了 Multi Crop 的数据增广方式实现 local-to-global 内容的学习，也就是 DINO 不仅要学习对 Transformers 的不变性，还要学习到 local-to-global 的一致性!
* 其它实现细节上的不同，如 $\tau_t$ 的调整等

## MoBY

本文没啥新的技巧，主要就是将 MoCo V2 与 BYOL 进行了组合创新，然后 Backbone 替换为 Swin Transformer 模型，以及验证预训练对下游如 Object Detection & Semantic Segmentation等任务的收益。

BYOL 的Asymmetric Encoder 结构是指模型包含两个 Encoder，分别是online encoder 以及 target encoding，两个 Encoder 都包含一个Backbone 以及一个 Projection Head，但是 Online Encoder 会多包含一个 Prediction Head，这也就是 Asymmetric Encoder 名字的来源。Online Encoder的参数基于梯度进行更新；Target Encoder 的参数基于 Online Encoder 的参数基于momentum的方式进行更新，并且momentum的参数会从0.99逐步更新到1.0。

Loss的实现与MoCo V3 的实现类似，不赘述。

作者新引入了Aynmmetric Drop Path 来提高性能。Drop Path 在基于 Transformer 的有监督训练过程中已经被证明是一个非常有用的正则化手段。这里非对称 Drop Path 的意思是，Drop Path 仅应用在 Online Encoder 上，如果也用在 Target Encoder 上会导致性能下降（70.9 vs 69.0）。这一技巧其实在 DINO 的代码里也有被使用的体现。

其它的结果对比以及数据细节可参考论文。

## 消融实验

### Position Embedding

MoCo V3 里基于 Linear Acc 对Position Embedding进行了对比，发现 Sin-Cos 方式略好于 Learned 的方式，Sin-Cos 的实现可以参考下 MoCo V3 的源码；另一方面，与不使用 Pos Embedding 相比的结果来说，目前的 Position Embedding 的使用还有待进一步挖掘。

ViT-B, 300 ep | sin-cos | learned | None
------ | ------ | ------ | ------
linear acc. | 76.5 | 76.1 | 74.9

### Class Token

MoCo v3 的作者测试了使用 Global Avg Pooling 代替 `[CLS]` Token的特征，发现Acc没有明显变化(76.5 VS 76.3)；其中 LN + Pool 表示在 Pooling 层之前先计算一次 LN。

ViT-B, 300 ep - | - w / CLS - | - w/o CLS; LN + Pool - | - w/o CLS; Pool
------ | ------ | ------ | ------
lienar acc. | 76.5 | 69.7 | 76.3

### Traning Length

MoCo V3 作者发现，模型越小(ViT/S, ResNet50)，训练越长的增益越大。

 | | 300 ep | 600 ep
 | ------ | ------ | ------
 | ViT-S/16 | 72.5 | 73.4 (+0.9)
 | ViT-B/16 | 76.5 | 76.7 (+0.2)

对于 DINO也有同样的发现。

DINO ViT-S | 100 ep | 300 ep | 800 ep
------ | ------ | ------ | ------
k-NN top1 | 70.9 | 72.8 | 74.5

## 其它

实际实验中，按照 SimSiam 的方式， Backbone 换成 VOLO，优化器使用 AdamW，然后模型就很容易塌缩了，表现在 Loss 数值快速到达 -1.0（SimSiam 使用的 Loss），表明已经达到最优了，无法继续优化。更换成 SGD 优化器之后，情况有所缓解，但是收敛速度还是非常快，Epoch 1 之后也开始坍塌。模型替换为 ResNet50 之后，模型就可以训练。所以暂定原因为 Transformer 模型不适用于 SimSiam 无监督训练框架。

将自监督训练框架切换为 MoCo V3 的方式，可以正常训练。针对参数$T$，当取值为1.0时收敛变慢，取值为默认的0.2之后收敛变快。然后使用预训练后的参数初始化Transformer模型，可以让模型收敛的更好，达到与使用RandomAug等数据增广训练的模型非常接近的精度水平，而且前30个epoch的结果就基本稳定了；但是如果也加上这些数据增广，那么精度怎么变化呢？基于有限的实验，前60epoch的结果来看，使用自监督训练模型进行初始化时，精度比从头训练的情况增加了不到1个百分点（0.8左右）最终的精度对比时：80.63 vs 80.58，这里有一个因素是模型结构发生了变化，也就是 CNN Stem 的结构由(conv7x2 -> conv3x1 -> conv3x1 -> conv4x4)变为(conv7x2 -> conv3x2 -> conv3x2 -> conv3x1)。主要问题在于前者训练过程非常不稳定，精度增加非常缓慢，后者的精度则稳步上升。经过进一步验证，同一种架构同一个模型，不同初始化方式，精度对比是：80.63 vs 80.52，只有0.1个百分点的提高而已。至于温度参数的作用，可以参考[Understanding the Behaviour of Contrastive Loss](https://arxiv.org/abs/2012.09740)这篇论文。

将自监督训练框架切换为 DINO 的方式，也可以正常训练，但是训练速度会变慢比较多，毕竟 crop 的个数增加了。warmup epochs 占总的 epoch 的比例由 40% -> 10% 之后，收敛速度变快。
