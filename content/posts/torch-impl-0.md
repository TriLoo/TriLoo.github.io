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

这里是新增一个支持前向、后向计算的方法，也就是说，当前 Torch 内所有支持训练的计算（支持后向传播梯度）本质上都是来自`torch.autograd`命名空间下的`Function`。所以新增一个计算方法，需要作为派生自`torch.autograd.Function`类的子类来完成。

存在这种extending方法的主要原因是，希望新增一个自定义操作，可以用在模型训练中，而这个新增的操作要么不可求导、要么是一个非Torch的变量（比如Numpy Array），但是还是希望模型中新增了这个计算之后，梯度仍然可以沿着模型传递，从而支持 autograd engine 的模型参数更新。换句话说，新增的 Function 子类，可以隐藏不支持求导的计算，将断开的梯度传播链路 chain 起来。另一种情况是，新增自定义 Function 可以Wrap C++实现的操作，或者进行一些类似Op融合的操作来提高运算效率。

新增 Autograd Function 的步骤主要分为四步，具体写代码是实现两个Function子类的静态函数。下面是四个实现步骤:

1. 派生`torch.autograd.Function`子类并且实现两个静态函数

  * forward 函数

    用于前向计算的函数，可以接收任意数目的参数，如果有默认值，则对应的参数是可选的。输出参数的类型可以是单个 Tensor 输出，或者 Tuple 形式的多个输出。

  * backward 函数

    定义梯度计算函数。输入的参数是对应 `forward()` 函数输出参数的梯度，也就是前向过程中有几个输出，这里就有几个输入，然后就可以根据这些输入的梯度参数计算输出梯度了，而返回变量个的个数与`forward()`函数的输入参数的个数一致。当`foward()`有可选参数的时候，这些参数对应的返回的梯度应该是None。

2. 使用`ctx`参数提供的一些操作来保证新增的Function可以适应autograd engine中的计算

  ctx 提供了一些有用的参数可以帮助新 Function 的实现，并且支持 autograd engine 的计算。

  * `save_for_backward()`函数

    前面提到，`backward()`函数的输入参数都是梯度值，有些计算过程还需要模型对应计算的状态参数，比如 CNN 中的权重/偏置项等。这个函数的作用就是为了在前向计算函数中保存这些参数的，然后在后向过程中取出来用于计算梯度。

  * `make_dirty()`函数

    前向计算中，如果参数使用了in-place操作，那么就需要用这个函数来指示。

  * `mark_non_differentiable()`函数

    告诉 autograd engine，对应的输出不可求导。

  * `set_materialize_grad()`函数

    我的理解是，如果有些参数的梯度是None，但是如果设置了`set_materialize_grad(True)`，那么这些梯度会用合适大小的全零的 Tensor 代替；如如果设置为 False，则这些参数传入 `backward()` 函数中对应的梯度就会保持 None。

3. 必要的时候使新增的`Function`支持高阶求导

  为了支持高阶求导，需要在 `backward()` 的修饰器中使用 `once_differentiable()` 来设置该后向传播函数只能求导一次。

4. 建议使用`torch.autograd.gradcheck()`函数对结果进行验证

  使用`torch.autograd.gradcheck()`函数来验证实现的后向传播函数是否正确。

一个具体的例子如下。

``` python {linenos=table linenostart=0}
# Inherit from Function
class LinearFunction(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias
```

在实际使用时，为了方便，一般会有下面的一条赋值：

``` python
linear = LinearFunction.apply
```

### Extending nn

一般来说，扩展 nn 有两种方式，一种是上面提到的 Function 方式，一般适用于那些没有自身计算状态参数（如卷积权重）的操作，另一种是定义 Module 子类的方式，后者需要自定义`__init__()`以及`forward()`两个成员函数，`forward()`成员函数内一般就会调用上面提到的 Function 来实现操作。

## Optimizer

实现自己的 Optimizer 的时候，需要继承`torch.optim.Optimizer`类。需要实现`__init__、__setstate__、step`等函数；然后将新 Optimizer 的参数，比如lr, eps, betas等参数保存到`defaults`字典中，并跟parameters一起传给Base类的`__init__`函数。`__setstate__`函数主要是为了比如在pickle等序列化中使用，并做必要的更新，比如`self.param_groups`里的成员。在 `step()`函数里，会更新`self.state`成员变量，然后后面更新的时候就可以直接从 `state` 里面取出来进行更新就可以了。

此外，defaults 字典里面的信息在`add_param_group()`函数里面被放入`self.param_groups`里面了，如lr, eps, betas等；特定Optimizer的相关数据放在`self.states`里面了，如Adam里面的 m / v 等。

具体例子可以参考 TIMM 库里的AdamW算法实现。

## PyTorch 部署全流程

一般来说，咱们使用 PyTorch Code (python) 完成模型的开发与训练，然后转换成 TorchScript IR 表示并序列化保存到文件中，接下来可以在不依赖 Python 的情况下进行部署与推理；另一方面，为了优化模型的计算效率，TorchScript IR 会进一步被转换成 ONNX IR 表示，再之后，可以选择直接利用 ONNX Runtime 进行优化部署推理，也可以进一步将 ONNX IR 转换成 TVM / TensorRT 等工具进行优化部署。

实际使用中发现，针对 CPU 这一部署环境以及 Transformer 相关的模型而言，ONNX Runtime 的优化效果出人意料的好，实际耗时出人意料的低。所以这里就列一下整个部署过程中涉及到的步骤以及如果需要深入理解、开发所需要看的内容。

* TorchScript 

  首先是将 PyTorch Code 转换成 TorchScript Code，这是因为TorchScript Language 支持的语法以及操作，只对应 PyTorch Code中的一个子集，所以需要修改原始的代码以可以通过 TorchScript 的编译。获取TorchScript的两种方式：
  * Tracing: 主要缺点是不支持动态 shape 的输入，而且也没法处理 if-else 等逻辑；只会记录当前输入所走过的计算路径
  * Scripting (Annotation): 这种方法会分析PyTorch Code的构成，类似于一个编译过程，所以记录的是实现逻辑。缺点是，既然属于编译过程，那么就很容易出现一些语法错误，需要修改

  其次就是 TorchScript 涉及到的具体实现原理了。这个以后单独在 blog 里贴出来。

* ONNX 

  属于 Protobuf 文件定义的一套 IR 规范，了解  Protobuf  的基本使用后在看源码会更舒服。

* ONNX Runtime

