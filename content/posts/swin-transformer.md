---
title: "ICCV2021 Best Paper - Swin Transformer"
date: 2021-10-15T22:07:45+08:00
draft: false
mathjax: true

excerpt_separator: <!--more-->
---
重读论文之后，发现还真是一个非常精巧的模型。<!--more-->

## 模型结构

Swin-Transformer的创新点主要提出了下面几个新的结构：

* 基于Transformer的层级结构提取多尺度的Feature Map
* Shifted Window based Self-Attention

对于多尺度Feature Map与ViT等模型的但尺度对比如下，左图红色框表示 Windows，灰色框表示Patch。SwinT 每层 Feature Map 的Windows 不同（每个 Windows 内Patch数量固定），包含的总的 Patch 也就不同（即分辨率不同）；而右边 ViT 模型自始至终都只包含一个 Windows，且 Windows 内都包含固定的 Patch 个数。

![图 - 1 多尺度Feature Map示意图](/imgs/swin-transformer/swin0.png)

Swin-Transformer 中涉及了几个层次概念：Pixels, Patch, Windows。Pixels 是在输入的原始图像中定义的，比如 224 * 224 的空间维度；Patch 是基于Pixels定义的，每个Patch论文中指定为 4 * 4 个像素，扁平化之后，每个 Patch 对应的尺寸是：`1 * 48`，其中48 = 4 * 4 * 3，即每个像素包含3个 RGB 通道；Windows 基于 Patch，论文中每个 Windows 内包含7 * 7 个Patch，计算Self-Attention时是在每个 Windows 内进行的，所以只要固定 Windows 大小，则 Windows 的个数与图像的 H * W 成正比，而不是与 $(H*W)^2$成正比了。注意，在不同的 Stage 中的 Windows 对应的 Pixels 个数是不同的，比如在 Stage1，对应的像素是 28 * 28(4*7)，在 Stage 2 就对应了 56 * 56(4 * 7 * 2)。

由于不同的 Windows 之间不会重叠，所以作者机智的引入了 Shifted Window Self Attention并引出了对应提高计算效率的做法。

总而言之，模型结构还是非常巧妙的。

### 整体结构

整个模型结构可以分为3个部分：Stem / Backbone / Head 部分。

对于 Stem 部分，Swin-Transformer 定义了 Patch，每个 Patch 是 raw pixel RGB 数值拼接起来的，论文中每个 Patch 对应 4 * 4 个raw pixels，所以每个 Patch 的特征维度是 4 * 4 * 3 = 48。然后经过一个全连阶层映射到 embedding size 维度，也就是论文以及代码中的 C，实际实现中，这一步是通过一个 `conv2d(ks=4, stride=4)` 的卷积层实现的。

Backbone 部分是本文的主要内容，包含四个 Stage，每个 Stage 都是由若干层 Swin Transformer Block构成，不同 Stage 之间通过 Patch Merge 来下采样。Patch Merge 的实现过程就是将`2 * 2`个相邻的Patch拼接起来，由`2 * 2 * C`的数据得到一个`1 * 4C`的数据，然后再经过一个全连阶层降维到`1 * 2C`，经过这一步，Tokens 数量下降4倍，在空间维度相当于将Feature Map的维度下采样一倍，这既提高了后续层的计算效率，与提高了模型的感受野，生成具有不同感受野的Feature Map！每个 Stage 下采样1倍，Stage 2/3/4 对应的 Feature Map 的空间维度分别为：$\frac{H}{8} \times \frac{W}{8}, \frac{H}{16} \times \frac{W}{16}, \frac{H}{32} \times \frac{W}{32}$，输入是 224 * 224的图像，最后输出的是 7 * 7的Feature Map。

对于 Swin Transformer Block的结构细节在下面。

Head 部分就是一个普通的Global Pooling + Linear(feat dim, cls num) 的结构。

说回 Swin Transformer Block，与正常 Transformer 的唯一区别是计算 Self Attention的时候，FFN 与正常 Transformer 结构一致。Swin Transformer Block 使用了Shifted Window based Self-Attention进行计算，将普通 Self Attention 中计算 Attention Score 的复杂度由与像素个数的平方成正比下降到与像素个数成正比！

主要涉及到的想法有两点：

* Windows Based Self Attiontion: 只在 Windows 内计算Attention Score，假设Windows 的大小是 M（即每个 Windows 内包含 M * M 个Patch，对应M * 4 * M * 4个像素），则计算复杂度与 $M^2$成正比，不同 Windows 之间不重叠
* Shifted Windows Based Self Attention: 上一步导致Windows 内的像素与 Windows 外的像素关联性较低，作者提出了 Shifted Window Partition in Successive Blocks，也就是下一层 Transformer 计算时，Feature Map 会有一个平移，所以原来不交互的相邻的两个 Windows现在属于同一个 Windows了

对于第二点，在Shifted的时候会导致 Windows 的分区变多（因为要保证原来属于边界两边的像素不能互相计算相关性），作者提出了 Efficient Batch Computation For Shifted Configuration 的方法 + 配合 Attention Mask 来提高计算效率。

对于Patch / Shifted Windows 的示意图如下图。

![图 - 2 Shifted Window示意图](/imgs/swin-transformer/swin1.png)

包含 Patch Merging 的整体模型结构如下图，可以看到核心的 Patch Parttition (Stem Block) 以及 Patch Merging 以及对应的 Token 的个数与特征维度。

![图 - 3 多尺度Feature Map示意图](/imgs/swin-transformer/swin2.png)

这里 Patch Merging 是在每个 Stage 开始的时候完成的，但实际在代码实现中，是在每个 Stage 的最后才进行的，这样 Feature Dim 导致模型的计算量增加速度会下降。

在Swin Transformer Block中，正常 Windows Based Self Attention 与 Shifted Windows Based Self Attention 的分布是相互交替的，并且这里 Shift 的大小是 `window_size // 2`。

``` python {linenos=table}
    # ...
        shift_size=0 if (i % 2 == 0) else window_size // 2,
    # ...
```

针对不同的参数量，模型结构如下。

![图 - 4 不同参数量SwinT的结构](/imgs/swin-transformer/swin9.png)

下面详细说明一下两种 Windows Based Self Attention 的实现。

### Windows based Self Attention

与正常 Transformer 中 Self Attention 的区别在于这里计算 Attention Score 时仅限于 Windows 内的 Patch 之间计算，与 Windows 外的 Patch 不会计算，这一步通过将 Seq Len 由 $H * W$ 变为 $\frac{H}{window size} \times \frac{W}{windows size}$，其余多出来的数据加入到 Batch 维度实现的。这一步是基于`window_partition / window_reverse`函数实现的。

其中，`window_partion()`就是切分然后合并到 Batch 维度上。

``` python {linenos=table}
def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x
```

这一步的计算量变化。正常 Transformer 层的计算复杂度：

$$\Omega (\textrm{MSA}) = 4hwC^2 + 2 (hw)^2 C$$

变为 Windows Based Self Attention 的计算复杂度：

$$\Omega (\textrm{MSA}) = 4hwC^2 + 2 M^2 (hw) C$$

两个式子等号右边的第一项都是表示4个Linear层的计算复杂度（Q, K, V + 输出），第二项是 Self Attention 的计算复杂度，可以看出，当每个 Windows 内的 Patch 数量 M * M 固定时，这一项与像素个数（图像大小）成正比，而第一个式子是与像素个数的平方成正比！这里每个 Windows 之间互相不重叠。

另一点是，虽然这里是在 Windows 范围内计算 Self Attention，但是对于计算Multi Head的方式计算并不影响，所以还是可以实现MultiHead Windows based Self Attention。`WindowAttention`类的实现如下。

``` python
class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        # B_ = num_windows * B
        # (B_, num_heads, N, C // num_heads), N = Wh * Ww
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))    # (B_, num_heads, N, N), N = Wh * Ww

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        # nH = num_heads
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
```

### Shifted Windows based Self-Attention

主要是提高不同 window 之间的信息交互程度。

正常的 attention mask的输入尺寸是:`[bs, seq_len]`，然后被扩展到 `[bs, 1, 1, seq_len]`，其中第二维对应的是 head 维，第三维对应的是batch内当前样本的Token输入。

使用`[PAD]`表示仅用于拼接的无效Token，然后一个正常的输入 Token 系列是：`[我][是][谁][PAD][PAD][PAD]`等，也就是`seq_len=6`，对应的 `attention_mask=[1, 1, 1, 0, 0, 0]`，并假设每个 Token 对应的维度是128，则计算 Attention Score 的结果（相似度得分）如下：

``` text
我-我       我-是       我-谁       我-PAD      我-PAD      我-PAD
是-我       是-是       是-谁       是-PAD      是-PAD      是-PAD
谁-我       谁-是       谁-谁       谁-PAD      谁-PAD      谁-PAD
PAD-PAD     PAD-PAD       PAD-PAD       PAD-PAD      PAD-PAD      PAD-PAD
PAD-PAD     PAD-PAD       PAD-PAD       PAD-PAD      PAD-PAD      PAD-PAD
PAD-PAD     PAD-PAD       PAD-PAD       PAD-PAD      PAD-PAD      PAD-PAD
```

然后对应的 Attention Mask 就是：

``` text
1   1   1   0   0   0
1   1   1   0   0   0
1   1   1   0   0   0
1   1   1   0   0   0
1   1   1   0   0   0
1   1   1   0   0   0
```

所以与Attention Score矩阵求和时，只有前三个字计算 Softmax 权重，后面三个 PAD 的权重都非常小；但是这里也有一个问题，就是 Attention Score矩阵的下面三行以及左边三列对应的Softmax权重也不是一个非常小的数，所以下一层的时候，对应上一层 PAD 位置的输出就包含了上一层中的`我是谁`三个字的信息了。这会不会造成什么污染呢？有待验证。

图-2展示了 Shifted Window 前后 Windows 范围的比较，论文中选在向Top-Left方向进行平移，借助`torch.roll(+-)`来实现以及复原。需要指出的，平移之后，原本在左上角的Patch会移动到右下角，与正常 Feature Map的右下角的 Patch 变成相邻的，但是此时虽然相邻，但是不能将它们按照正常 Windows based Self Attention 进行计算，而应该分开计算，即原来相邻的 Patch 计算 Attention Score，不相邻的Patch之间不能计算 Attention Score。对应下图，其中A, C, D, E, F, G几个区域都应该单独计算 Attention Score。

![图 - 5 Shifted Windows分区示意图](/imgs/swin-transformer/swin3.png)

这导致Windows的个数由正常的$\lceil \frac{H}{M} \rceil \times \lceil \frac{W}{M} \rceil$ 变成 $(\lceil \frac{H}{M} \rceil + 1) \times (\lceil \frac{W}{M} \rceil + 1)$，这一点虽然看上增加不多，但是当$\lceil \frac{H}{M} \rceil = 2$时，由2 变为3，则相当于计算量增加了2.25倍，所以有必要对一点进行优化。

作者提出使用Batch computation for shifted configuration计算 Self Attention，但当前的代码前提是所有输入图片的尺寸是不一致的。主要思想就是配合 Attention Mask 来保证只关注自己区域内的 Patch。代码如下。

``` python {linenos=table}
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),    # start = 0, stop = -window_size, step=None
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
```

这里主要思路就是将属于同一个区域内的 Patch 赋值一个相同的标记`cnt`，然后同一个Windows的Patch的标记与其它patch的标记相减以后，相同 Windows 内对应的数值就为0，而不为0就意味着需要Masked掉对应位置的Attention Score的计算，也就是赋值为-100.0。对于`window_partion()`函数的实现见下面具体的代码实现。

### Relative position bias

SwinT使用Windows内的相对位置编码来学习位置信息。

$$\textrm{Attention}(Q, K, V) = \textrm{SoftMax}(QK^T/\sqrt{d} + B) V$$

其中，$Q, K, V \in \mathbb{R}^{M^2 \times d}$，由于所有的 Stage 中相对位置的取值范围是：$[-M + 1, M - 1]$，所以作者让 B 的取值来自于$\hat{B} \in \mathbb{R}^{(2M-1) \times (2M-1)}$，这样对于Windows内任意两个Patch之间都会有一个相对位置编码向量进行表示(而且还考虑了分axis)！注意，每个 Swin Transformer Block层的 B 取值是不同的。

这里最主要的代码逻辑是要将所有的相对位置$[(-M + 1, -M+1) \times (M-1, M-1)]$映射到$[0, (2M - 1) * (2M - 1)]$范围内唯一的索引。对应代码实现，就是将每个元素的用二维坐标进行编码，然后将二维坐标映射到一维数字。映射成二维坐标是通过三行代码实现的：

``` python
    coords_h = torch.arange(self.window_size[0])
    coords_w = torch.arange(self.window_size[1])
    # [0, ...] 表示行索引，[1, ...]表示列索引
    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
```

然后是基于二维坐标计算相对位置，这里相对位置的计算也是分成行、列分别进行计算。

``` python
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
```

现在`relative_coords`的最后两维分别表示行、列的相对位置，取值范围都是：$[-M + 1, M - 1]$。

然后就是将二维相对位置信息映射到一维索引，首先是将相对位置的取值范围变为：$[0, 2M - 2]$，这一步通过加上$M - 1$实现，然后就是将行相对位置 * 宽度 + 列相对位置，这里宽度是$2M - 1$，映射后的一维索引取值范围是：$[0, (2M - 1) \times (2M - 1)$。

``` python
    relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
    relative_coords[:, :, 1] += self.window_size[1] - 1
    relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
    relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
    self.register_buffer("relative_position_index", relative_position_index)
```

上述得到相对位置的一维索引后，需要根据一个 Table 来获取对应的相对位置编码，也就是从 $\hat{B}$ 中获取，定义如下：

``` python
self.relative_position_bias_table = nn.Parameter(
    torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
```

至此就是所有的Swin Transformer的算法结构了。

## 代码实现与一些细节

* drop rate: 0.0
* atten drop rate: 0.0
* drop path rate: 0.1

然后 Multi-Head Self Attention中Output Mlp用到的 drop rate 与 FFN 中用到的 drop rate 是同一个，atten drop rate 只用于Self Attention中对 Attention Mask进行处理。FFN中的Dropout的位置即结构如下: Linear + Act + Dropout + Linear + Dropout。

Patch Merge 层的位置，论文中这个层被放在每个 Stage 开始的位置，但是实际代码里是被放在每个Stage最后一层的。

``` python {linenos=table}
class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

```

### PatchMerging

可以看出，后续每个Patch的Merge过程就是取出 x0, x1, x2, x3 四个矩阵，然后拼接起来，送入一个`LayerNorm + Linear`层，后面的 Linear 层将channel维度从 4C 映射到 2C，也就是每个 Stage 下采样一倍，然后特征维度也只增加一倍。

``` python {linenos=table}
class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x
```

### SwinTransformerBlock

基于上面提到的 Shifted Windows based Self Attention 中 Mask 的分析以及 window partion以及WindowAttention等函数/类的实现，现在可以给出 Swin Transormer Block 的完整实现了。

``` python
class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),    # start = 0, stop = -window_size, step=None
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x
```

## 实验结果

Swin Transformer 在 分类、识别、分割几个任务上都获得了提高，尤其是识别、分割提升显著。在训练分类时，数据增广与 ViT 同，且训练300 epochs，warmup epochs为20。

ImageNet上的性能表现。

![图 - 6 SwinT在ImageNet上的性能对比](/imgs/swin-transformer/swin4.png)

COCO上的性能表现。

![图 - 7 SwinT在COCO上识别性能对比](/imgs/swin-transformer/swin5.png)

分割性能表现。

![图 - 8 SwinT在分割任务上的性能对比](/imgs/swin-transformer/swin6.png)

不同位置编码的影响。

![图 - 9 SwinT中不同位置编码对精度的影响](/imgs/swin-transformer/swin7.png)

速度对比，GPU为V100。

![图 - 10 SwinT速度对比](/imgs/swin-transformer/swin8.png)
