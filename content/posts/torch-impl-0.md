---
title: "Torch实现原理分析积累"
date: 2021-09-18T11:08:11+08:00
draft: false
mathjax: true

excerpt_separator: <!--more-->
---
Pytorch 实现学习积累。<!--more-->

## 基础

* All objects in pytorch are passed by reference in python. But doing `a=` does not try to change `a` in-place, it only give the name `a` to the object returned by the right hand side.
* 矩阵乘：@， Matmul，mm（后两者的区别在于 mm 仅适用于二维Tensor，matmul适合高维Tensor）；*，mul 实现的是element-wise乘
* `_` suffix ops 是in-place操作
* Tensor 与 Numpy 之间可以共享底层存储空间，所以修改一个也会导致另一个变量发生变化。如`.numpy()`操作，`from_numpy()`等
* 自定义Dataset，需要自己实现`__init__`、`__len__`、`__getitem__`等函数；`ToTensor`会将PIL Image、NumPy ndarry转换成`FloatTensor`，并且将像素上的数值范围缩放到(0.0, 1.0)之间。
* 继承`nn.Module`创建模型的时候，会自动收集定义在models内的fields，并且让所有的 parameters 都可以被`parameters()`以及`named_parameters()`等方法获取到

## Module

Module 在调用的时候实际会调用`Module._call_impl()`函数，这个函数里调用顺序如下。

1. 调用`_global_forward_pre_hooks`或者`self._forward_pre_hooks`里面所有的hook，对当前的Module以及输入数据进行处理，hook 函数的格式是：`hook(module, input) -> None or modified input`，如果 hook 函数会返回数据，那么这个返回的数据才是真正的输入 forward() 函数进行计算的数据
2. 调用`forward_call()`函数完成前向计算
3. 调用`_global_forward_hooks`或者`self._forward_hooks`里面的所有hook，hook函数签名是`hook(module, input, output) -> None or modified output`，函数的输出是最终的输出
4. `full_backward_hooks`里的 hooks

## Autograd

通过设置Tensor的`requires_grad`来决定是否需要计算 Loss 对该 Tensor 的梯度。

* torch.autograd.Function

  记录对Tensor的操作，是一个类，包含`forward()`、`backward()`两个静态成员函数。每个Function完成对 Tensor 的一个操作，并记录发生的事情。所有的 Function 被组织成有向无环图（DAG），边表示数据依赖(input <-- output)。当反向传播时，按照拓扑顺序依次调用Function的`backward()`函数。

  实际使用的时候就是继承Function类并实现这两个静态成员函数。一个具体例子如下，所以都是静态成员函数进行操作，无需创建具体实例。

  ```python {linenos=table linenostart=0}
  class Exp(Function):
      @staticmethod
      def forward(ctx, i):
          result = i.exp()
          ctx.save_for_backward(result)
          return result
      @staticmethod
      def backward(ctx, grad_output):
          result, = ctx.saved_tensors
          return grad_output * result

    output = Exp.apply(input)
  ```

  注意，Function知道Tensor的前向计算，也支持后向传播，后向传播函数保存在`tensor.grad_fn`属性中。也就是说Function 是计算图中的节点，边才是 Tensor。

* is_leaf
  
  这个函数用来判断Tensor是否保存了grad。

  * 如果Tensor的`requires_grad=False`，则通常是 Leaf
  * 如果 Tensor 是用户创建的，那么即使`requires_grad=True`也是Leaf，意味着这些Tensor不是一个Op的结果，并且`grad_fn=None`
  * 只有Leaf Tensor 才会在`backward()`过程中保存梯度结果；如果需要获取那些non-leaf节点的grad，可以使用`Tensor.retain_grad()`来修改
  * 第三条与第一条貌似冲突，其实不冲突，因为 `requires_grad=False`的含义是指这个 Tensor 的梯度不需要向后传播了，而不是不会计算该 Tensor 的梯度，也就是实际是指`grad_fn=None`。
  * 从CPU拷贝到 GPU 上也算是一个 Op 操作，具体例子可以查看：[torch.tensor.is_leaf](https://pytorch.org/docs/stable/generated/torch.Tensor.is_leaf.html?highlight=is_leaf#torch.Tensor.is_leaf)

* Disabling Gradient Tracking

  有时候需要停止一些 Tensor 的梯度后向传播，那些`requires_grad=True`的 Tensor 都会跟踪该Tensor 的计算历史，并支持梯度计算。所以要想阻止后向传播，有两种方式：

  * 使用 `torch.no_grad()` block 进行封装
  * 使用 `detach()`，相当于新建了一个Tensor返回的，所以计算梯度更新这个新的 Tensor，之前旧的 Tensor 数值也会保持不变。

  下面的方式适合单个 Parameter 的梯度更新。

  * 设置`parameter.requires_grad=False`
  * 设置`parameter.grad=None`，优化器在根据梯度更新这个参数时，如果发现 `grad=None`，则略过当前参数，从而实现防止梯度反向传播的目的

  经过上述两种方式处理后的 Tensor 直接影响是，不会向后传播 Gradient，也不会发生数值变化。

* Tensor Gradients and Jacobian Products

  大部分情况下，Loss函数计算得到的是一个Scalar数值，计算梯度容易理解。但是当 Loss 是一个多维的Tensor时，反向传播计算的就是`Jacobian product`，而不是真正的梯度了。

  一般来说，输入、输出都是 Tensor 时，反向传播得到的是一个`Jacobian matrix`，但是 pytorch 支持`Jacobian product`的计算，此时需要一个与输出Loss同等尺寸的Tensor作为`backward()`函数的输入。

  下式中，`x, y`为输入输出，计算`y`对`x`的梯度时，引入的 `v` 就是上面提到的需要跟 `y` 尺寸相同的新引入的 Tensor，具体例子可参考[Automatic Diff](https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)下方。

  $$y=f(x), J = \frac{\partial y}{\partial x}, v^T \cdot J$$

* optimize steps

  1. call `optimizer.zero_grad()`
  2. call `loss.backward()`
  3. call `optimizer.step()`

* 其它

  * 每次`backward()`之后，创建的计算图都会被重置，从而支持每次 iter 之间修改数据的尺寸、条件判断修改计算图等，也就是对动态计算图的支持；如果想保留当前的计算图，可以在 `backward()`函数中设置`retain_graph=True`
  * 但是连续两次`backward()`时，同一个 Tensor 的梯度会被累加。

## Extending Pytorch

主要参考：[Extending Pytorch](https://pytorch.org/docs/stable/notes/extending.html#extending-torch-autograd)

### Extending Autograd

TODO

## Optimizer

实现自己的 Optimizer 的时候，需要继承`torch.optim.Optimizer`类。需要实现`__init__、__setstate__、step`等函数；然后将新 Optimizer 的参数，比如lr, eps, betas等参数保存到`defaults`字典中，并跟parameters一起传给Base类的`__init__`函数。`__setstate__`函数主要是为了比如在pickle等序列化中使用，并做必要的更新，比如`self.param_groups`里的成员。在 `step()`函数里，会更新`self.state`成员变量，然后后面更新的时候就可以直接从 `state` 里面取出来进行更新就可以了。

此外，defaults 字典里面的信息在`add_param_group()`函数里面被放入`self.param_groups`里面了，如lr, eps, betas等；特定Optimizer的相关数据放在`self.states`里面了，如Adam里面的 m / v 等。

具体例子可以参考 TIMM 库里的AdamW算法实现。
