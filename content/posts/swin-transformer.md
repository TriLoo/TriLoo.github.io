---
title: "ICCV2021 Best Paper - Swin Transformer"
date: 2021-10-15T22:07:45+08:00
draft: true

excerpt_separator: <!--more-->
---
<!--more-->

## 模型结构

Swin-Transformer的创新点主要提出了下面几个新的结构：

* 基于Transformer的层级结构提取多尺度的Feature Map
* Shifted Window based Self-Attention

### 整体结构

### Shifted Window based Self-Attention

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

