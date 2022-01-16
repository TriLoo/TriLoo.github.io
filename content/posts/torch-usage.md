---
title: "Torch的一些使用方法记录"
date: 2021-10-01T16:06:21+08:00
draft: false

excerpt_separator: <!--more-->
---

记录一些Torch使用过程中会用到的小知识点。<!--more-->

## 求解中间变量的梯度

前面提到，`backward()`函数只会保存Leaf Node的梯度，如果要想保留中间计算结果的梯度，可以使用`Tensor.retain_grad()`来实现。是不是Leaf Node可以使用 `Tensor.is_leaf`来判断，简单来说Leaf Node有两类：

* `Tensor.requires_grad=False` 的Tensor属于Leaf Node
* `Tesnor.requires_grad=True`并且是由用户创建的Tensor也属于Leaf Node；用户创建意味着不是其它Op产生的Tensor.

实际要获取中间变量的梯度，有以下方法：

1. 使用`retain_grad()`

  ```python {linenos=table}
  def get_inter_grad():
    x = torch.ones((2, 2), requires_grad=True) 
    print(x.is_leaf)

    y = x * 2
    y.retain_grad()
    z = y ** 2

    z.backward(torch.ones_like(z))
    print(x.grad)
    print(y.grad)       # not None
  ```

2. 使用`torch.autograd.grad(outputs, inputs)`

  ```python {linenos=table}
  def get_inter_grad():
    x = torch.ones((2, 2), requires_grad=True) 
    print(x.is_leaf)

    z = y ** 2
    t = z.mean()
    x_res = torch.autograd.grad(t, x, create_graph=True)[0]
    print(x_res)
    y_res = torch.autograd.grad(t, y, create_graph=True)[0]
    print(y_res)
  ```

  注意，`torch.autograd.grad()`只能对Scalar output计算梯度，所以才用了`t = z.mean()`进行反向传播。

3. 使用`torch.Tensor.register_hook()`

  `register_hook()`函数会注册一个backward hook，每次计算该Tensor的梯度时，都会调用这个`hook`函数。函数签名是`hook(grad) -> Tensor or None`，一般来说，这里的 hook 函数不应该对输入的 grad 进行修改，而是返回一个新的梯度来代替 grad。`register_hook()`函数会返回一个 handle，可以调用`handle.remove()`来从当前的Tensor中去掉这个 hook 函数。

  ```python {linenos=table}
    global_grad = 0.0
    def extract_grad(grad):
        print('current grad: ', grad)
        global global_feat
        global_grad = grad
        return grad

    def get_inter_grad():
        global global_grad
        x = torch.ones((2, 2), requires_grad=True) 
        print(x.is_leaf)

        y = x * 2
        z = y ** 2
        t = z.mean()

        y_hook = y.register_hook(extract_grad)
        t.backward()
        print('y grad: ', global_grad)
        print('x grad: ', x.grad)

        y_hook.remove()
  ```

  但是实际使用下来，第三种方法获取到的还是 `global_grad` 原始的数值，有待进一步查原因。

## 获取模型权重的梯度

获取权重的梯度代码非常简单：

``` python {linenos=table}
    # ...
    for name, param in model.named_parameters():
        print(name)
        print(param.grad)           # 真实梯度，param.grad 是一个 Tensor
        print(param.data.grad)      # None
    # ...
```

## 使用checkpoint功能

gradient checkpointing的意思是说，在反向传播时，重新计算对应代码段的前向计算，这样就可以不用在前向计算时保存临时中间激活输出值以及对应的梯度等。

但是有一点需要注意就是需要保证那些具有随机属性的计算的两次前向输出应该是一致的，比如 Dropout，因此需要将`preserve_rng_state=True`传入到`torch.utils.checkpoint.checkpint()`函数中，但是这样做会导致性能下降较大，所以如果没有涉及到RNG 类的操作，那么需要将`preserve_rng_state=False`。另一点是，即使设置了`preserve_rng_state=True`，但是在`run_fn`函数里面将变量移动到一个新的device上的话，那么 RNG 状态的一致性也还是无法保证，所谓的新的device，就是当前device + 传入到 `run_fn` 的参数的device 的合集。

对应实现 `checkpinting` 的函数是：`torch.utils.checkpoint.checkpoint(function, *args, **kwargs)`函数。

checkpointing的工作原理是：`trading compute for memory`，也就是不会保存计算过程中的中间激活值，而是在反向传播时重新计算这些数值。可以应用到任意部分的模型计算。

具体来说，`function`表示的计算前向计算时是在`torch.no_grad()`里面执行的，但是`checkpoint()`函数会保存输入的tuple以及function parameters等。`function`计算可以输出非Tensor的参数，但是gradient recording 只会作用于那些Tensor的输出。注意，如果输出包含在`list, dict, custom objects`等结构体里，即使是Tensor，也不会被计算gradients。

一个具体的使用例子是 Albef 仓库里 `xbert` 的实现:

``` python {linenos=table}
    def create_custom_forward(module):
        def custom_forward(*inputs):
            return module(*inputs, past_key_value, output_attentions)
        return custom_forward

    layer_outputs = torch.utils.checkpoint.checkpoint(
        create_custom_forward(layer_module),
        hidden_states,
        attention_mask,
        layer_head_mask,
        encoder_hidden_states,
        encoder_attention_mask,
    )
```

这里使用了python的闭包方式进行实现function。

另一个API`torch.utils.checkpoint.checkpoint_sequential(functions, segments, input, **kwargs)`可以实现对sequential models进行checkpoints。

> Sequential models execute a list of modules/functions in order (sequentially). Therefore, we can divide such a model in various segments and checkpoint each segment. All segments except the last will run in torch.no_grad() manner, i.e., not storing the intermediate activations. The inputs of each checkpointed segment will be saved for re-running the segment in the backward pass.

``` python
    model = nn.Sequential(...)
    input_var = checkpoint_sequential(model, chunks, input_var)
```
