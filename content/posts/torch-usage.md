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
