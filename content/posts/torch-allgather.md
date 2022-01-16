---
title: "Torch all_gather 的梯度问题"
date: 2022-01-16T18:43:13+08:00
draft: false
mathjax: true

excerpt_separator: <!--more-->
---
pytorch all_gather 计算结果是叶子节点，也就是不会继续向后传递梯度了。<!--more-->

## 背景

* 背景一：使用 all_gather 来获取其它 GPU 上的参数

  最早接触使用Pytorch的`all_gather`来获取其它GPU上的数据在当前进程中使用的代码应该是 MoCo 论文中的实现：
  
  ``` python {linenos=table linenostart=0}
  @torch.no_grad()
  def concat_all_gather(tensor):
      """
      Performs all_gather operation on the provided tensors.
      *** Warning ***: torch.distributed.all_gather has no gradient.
      """
      tensors_gather = [torch.ones_like(tensor)
          for _ in range(torch.distributed.get_world_size())]
      torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
  
      output = torch.cat(tensors_gather, dim=0)
      return output
  ```
  
  一方面是，`concat_all_gather`函数使用了`no_grad()`修饰器；另一方面，即使不用 `no_grad` 修饰，这里的结果（也就是`output`）的梯度也不会传递给输入参数`tensor`。

* 背景二：对 all_gather 的结果进行梯度后向传播

  代码中使用了普通的 triplet loss 计算 Loss，然后进行梯度更新，triplet loss 函数的中的 anchor 来自于离线计算好的数据，因此不会进行梯度后向传播（requires_grad = False），而 pos, neg 则来自于上述`concat_all_gather()`函数的输出。最开始的时候，`autograd.backward(loss)` 的计算会报错，提示计算 loss 的几个参数都不需要计算梯度，去掉`torch.no_grad()`之后，错误仍然存在。
  
  另一个现象是，当 anchor 也来自于模型计算（可以梯度后向传播时），使用`concat_all_gather()`的结果计算 triplet loss 会比只使用当前 GPU 上输出作为 pos / neg 时速度快上一倍以上，这就非常违反直觉了，因为 `all_gather` 的通信开销应该导致速度更慢才对。
  
  因此， 考虑`concat_all_gather()`函数体中导致梯度传播中断的计算。主要是两个地方，一个是`ones_like()`这里，默认创建的 tensor 具有`requires_grad=False`参数，因此将代码替换为：
  
  ``` python {linenos=table linenostart=0}
      tensors_gather = [torch.ones_like(tensor, requires_grad=True)
          for _ in range(torch.distributed.get_world_size())]
  ```
  
  然而错误、或者训练速度异常仍然存在，因此，错误也就只可能出在 `all_gather()` 计算上了。

搜索引擎了一下，发现下面相关帖子:

[Will “dist.all_gather” break the auto gradient graph?](https://discuss.pytorch.org/t/will-dist-all-gather-break-the-auto-gradient-graph/47350)

## 让all_gather支持梯度传播

上面的问题总结出来就是，torch.dist 中自带的 `all_gather` 函数会阻断梯度的后向传播。针对这个问题，帖子中也给出了一个新的实现代码，并且配合新的`concat_all_gather`的实现代码如下：

``` python {linenos=table linenostart=0}
import torch
import torch.distributed as dist

class AllGather(torch.autograd.Function):
    """ 
    all_gather with gradient back-propagation
    """
    @staticmethod
    def forward(ctx, tensor_list, tensor):
        dist.all_gather(tensor_list, tensor)
        return tuple(tensor_list)

    @staticmethod
    def backward(ctx, *grad_list):
        grad_list = list(grad_list)
        rank = dist.get_rank()

        dist_ops = [
            dist.reduce(grad_list[i], i, async_op=True) for i in range(dist.get_world_size())
        ]

        for op in dist_ops:
            op.wait()

        return None, grad_list[rank] 


all_gather = AllGather.apply

def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    all_gather(tensors_gather, tensor)

    output = torch.cat(tensors_gather, dim=0)
    return output
```

后记，看MoCo代码中`concat_all_gather`的注释，原来答案就在纸面上，擦。
