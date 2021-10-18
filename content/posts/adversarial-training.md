---
title: "Adversarial Training"
date: 2021-10-05T11:21:33+08:00
draft: false
mathjax: true

excerpt_separator: <!--more-->
---

几个基础的常见的对抗样本的生成算法，包括FGM/FGSM, PGD等。<!--more-->

通常来说，对抗训练样本一般比哪些噪声样本破坏性更大，并且一般来说导致模型A失效的样本也会导致模型B失效。

## FGSM-FGM

### FGSM

论文：[FGSM](https://arxiv.org/pdf/1412.6572.pdf)。

在引言中，作者提到，对抗样本可以对模型提供类似于 Dropout / model average 等方法的正则效果，提高模型的泛化性能；而且作者认为，我们需要平衡下面两个因素：更线性化（浅层）的模型更容易训练，但是非线性更高的模型对对抗样本会有更大帮助。关于后者，已经有论文证明，复杂度更高的模型确实对对抗样本效果更好，如[Adversarial Machine Learning at Scale](https://arxiv.org/abs/1611.01236)以及[Towards Deep Learning Models Resistant to Adversarial
Attacks](https://arxiv.org/pdf/1706.06083.pdf)等。

对抗样本是指哪些对正确分类的样本做一点人眼看不出来的微小改动就导致模型预测错误的样本，这也说明模型没有学习到每个类真正的内容含义。

正常的样本为 $x$，对抗样本，也就是有一个扰动的样本变为 $\tilde{x} = x + \eta$，则在深度学习模型中乘以权重$w$后得到的输出为：$\omega^T \tilde{x} = \omega^Tx + \omega^T \eta$，也就是后一项$\omega^T\eta$的存在导致模型的输出与正常数据$\omega^T x$不一致了，严重的导致样本预测类别发生变化。所以，我们希望对抗样本可以实现的是，当 $\eta$ 很小的时候，即满足$\parallel \eta \parallel_{\infty} < \epsilon$ 时，模型的预测结果没有变化。

然而什么时候会导致模型的输出变化最大呢？也就是$\omega^T \eta$最大，当把$w,\eta$都看成是向量时，两者的方向一致的时候这个数值是最大的。直接体现在$\eta = \textrm{sign}(\omega)$，考虑到存在$\| \eta \|_{\infty} < \epsilon$ 的存在，实际当$\eta = \epsilon \textrm{sign}(\omega)$ 时，这项在输入上的扰动对输出影响最大，当$\omega$的维度为n，每个元素的平均大小是 m 时，输出的变化数值是 $\epsilon mn$，当n足够大的时候，这个数值就会变得非常大，所以当模型的channel 数上来以后，就会更容易受到对抗样本的攻击。

上面对线性模型的分析表明，输入信号中那些与权重方向更一致的数据对输出的影响会更大，这也是从一个简单的角度说明为什么模型会存在对抗样本。

对于存在非线性激活函数的模型而言，上述分析也有一定的适用性。因为实际模型中，通常使用 relu, maxout, sigmoid 等激活函数，为了方便模型的训练，这些激活函数也通常具有较高的线形特征，比如 sigmoid，需要让输入在 0 范围内，否则就会饱和，而在0附近正好对应 sigmoid 最接近线性函数的区域。对于这一类（存在非线性层）模型，我们选择让损失函数变大来代替让输出变化项$\omega^T \eta$变大，对应的扰动是让损失函数变大的方向，也就是梯度的方向：

$$\eta = \epsilon \textrm{sign} (\nabla_x J(\theta, x, y))$$

这一过程也就是将模型在$\theta$处线性展开，即：$f(x_0 + \eta) = f(x_0) +f(x_0)' \eta$，所以当$\eta$的方向与$f(x_0)'$方向一致的时候，才会使得$f(x_0 + \eta) - f(x_0)$的差别最大。这个算法就对应`Fast gradient sign method (FGSM)`。实验发现，当$\epsilon = 0.25$时，99% 的扰动输入都可以让一个浅层的 Softmax 模型预测错误。

虽然按照万能逼近定理来说，深度学习模型可以逼近任何函数，所以可以对抗adversarial样本的扰动，然而，前提是我们需要明确提供这一类的样本让模型进行学习，所以作者提出了下面的损失函数训练模型。

$$\tilde{J}(\theta, x, y) = \alpha J(\theta, x, y) + (1 - \alpha) J(\theta, x + \epsilon \textrm{sign}(\nabla_x J(\theta, x, y)))$$

这里作者设置$\alpha=0.5$。有一点，这里采用$\textrm{sign}$函数，这个函数是不可求导的，所以模型不会知道对抗样本对自身权重变化的反应，而下面提到的 FGM 采用的 `l2 norm` 则会知道，后者会让学习过程变得容易。

> However, we did not find nearly as powerful of a regularizing result from this process, perhaps because these kinds of adversarial examples are not as difficult to solve.

总结一下，对抗样本中涉及的扰动的方向与权重的方向一致才是重要原因，而不是扰动的大小；对抗样本存在是因为模型太线性化了，而不是太非线性化了。Rubbish Class Samples 是指那些没有意义的数据，也就是在人类看来，这些样本不属于任何一个类别，而不是真实数据 + 扰动。

### 为啥 Adversarial 样本具有 transferability

作者简要分析了一下为啥同一个对抗样本在多个模型上都会生效，而且还会预测成同一个类别。简单来说，是因为在同一个数据集上训练的模型权重一般也都比较相似，毕竟学习到的模型具有一定程度的泛化性，而权重的相似性也就导致对抗样本的一致性，也就是与权重的方向比较一致。

> The generalization of adversarial examples across different models can be explained as a result of adversarial perturbations being highly aligned with the weight vectors of a model, and different models learning similar functions when trained to perform the same task.

### FGSM 伪代码实现

参考的是: [ADVERSARIAL EXAMPLE GENERATION](https://pytorch.org/tutorials/beginner/fgsm_tutorial.html)

用于训练中需要验证。

``` python {linenos=table linenostart=0}
def fgsm_attack(data, model, label, loss_fn, optimizer, epsilon):
    data.requires_grad = True
    output = model(data)
    loss = loss_fn(output, label)
    loss.backward()

    data_grad = data.grad.data
    perturbed_data = data + epsilon * data_grad.sign()
    output_adv = model(perturbed_data)
    loss_adv = loss_fn(output_adv, label)

    loss_total = loss + loss_adv
    optimizer.zero_grad()
    return loss_total
```

另外一个参考代码是：[Adversarial Attack and Defense on Neural Networks in PyTorch](https://towardsdatascience.com/adversarial-attack-and-defense-on-neural-networks-in-pytorch-82b5bcd9171)。

### FGM

FGM对应的论文是：[FGM](https://arxiv.org/pdf/1605.07725.pdf)

FGM算法是FGSM算法在文本领域的扩展，文本领域输入的是 Token 在字典中的索引，所以没法对这个索引进行扰动，而FGM选择对token的 Embedding 进行扰动，如下图所示。

![图 - 1 FGM中对 Embedding 的扰动](/imgs/adversarial-training/fgm0.png)

此时，模型可以学到一个技巧，就是让Embedding $v_k$ 的Norm足够大，此时小的扰动相对于模型的输入而言就非常微小了，但是这样的Embedding并没有那么大的意义，因此作者选择对Embedding进行归一化。

$$\bar{v}_k = \frac{v_k - E(v)}{\sqrt{\textrm{Var}(v)}}$$

其中，

$$E(v) = \sum_{j=1}^K f_j v_j, \textrm{Var}(v) = \sum_{j=1}^K f_j(v_j - E(v))^2$$

是字典中单词的个数，$f_i$为第i个单词在所有训练数据中出现的频率。

FGM中对应的小扰动范围内导致Loss增加最大的数值为：

$$r_{adv} = -\epsilon \mathbf{g} / \parallel \mathbf{g} \parallel_2$$

其中，

$$\mathbf{g} = \nabla_x \log p(y | \mathbf{x}; \hat{\theta})$$

注意是对输入 x 的梯度，计算Loss的梯度时需要Label信息$y$的存在，$\hat{\theta}$为模型的权重，在计算题的过程中，这个权重不发生变化。

结合上文提到对于文本任务，这里对 Token Embedding 进行加性扰动。此时用$\mathbf{s}$表示word embedding vectors，Label为y，则定义扰动数值为：

$$r_{adv} = -\epsilon \mathbf{g} / \parallel \mathbf{g} \parallel_2, \mathrm{where} \mathbf{g} = \nabla_s \log p(y | \mathbf{s}; \hat{\theta})$$

实际实现中，这里应该是$+\epsilon \mathbf{g} / \parallel \mathbf{g} \parallel_2$而不是-。然后在训练过程中对应的 Loss 项为：

$$L_{\mathrm{adv}}(\theta) = - \frac{1}{N} \sum_{n=1}^N \log p(y_n | s_n + r_{\mathrm{adv}, n}; \theta)$$

其中，N是样本的个数，可以是 Mini-Batch 内样本的个数。

### Virtual adversarial training

Adversarial training，是针对有label的训练而言的，训练模型让该模型可以对unmodified examples以及adversarial examples都可以正确分类，不仅提高对 adversarial samples 的鲁棒性，也可以提高模型的泛化性。

> Adversarial training requires the use of labels when training models that use a supervised cost, because the label appears in the cost function that the adversarial perturbation is designed to maximize.

Virtual adversarial traingin，可以适用于哪些半监督学习，输入可以是 unlabeled examples。

> This is done by regularizing the model so that given an example, the model will produce the same output distribution as it produces on an adversarial perturbation of that example. Virtual adversarial training achieves good generalization performance for both supervised and semi-supervised learning tasks.

VAT对应的Loss函数是：

$$\mathrm{KL} [p (\cdot | x; \hat{\theta}) \parallel p(\cdot | x + r_{v-adv}; \theta)]$$

其中，

$$r_{v-adv} = \mathrm{ArgMax}_{r, \parallel r \parallel \le \epsilon} \mathrm{KL} [p(\cdot | x; \hat{\theta}) \parallel p(\cdot | x + r; \hat{\theta})$$

对于文本的 Embedding Vector 添加扰动，实现公式如下。

$$\mathrm{KL} [p (\cdot | s; \hat{\theta}) \parallel p(\cdot | s + r_{v-adv}; \theta)]$$

其中，

$$r_{v-adv} = \nabla_{s+d} \mathrm{KL} [p(\cdot | s; \hat{\theta}) \parallel p(\cdot | s + r; \hat{\theta})$$

这里，$d$为初始扰动，是随机初始化的，对应的Adv Loss项为：

$$L_{v-adv}(\theta) = \frac{1}{N'} \sum_{n'=1}^N' \mathrm{KL}[p(\cdot | s_{n'}; \hat{\theta}) \parallel p(\cdot | s_{n'} + r_{v-adv, n'}; \theta)]$$

$N'$为所有的labeled / unlabeled样本。

## PGD

本文的一个贡献是将Adversarial Training不再看成是一个Adversarial Samples训练问题了，而是统一到一个通用的优化Loss函数里。

$$\min \rho_\theta, \mathrm{Where}, \rho_\theta = \mathbb{E}_{(x, y) \sim \mathcal{D}} \left[ \max_{\delta \in \mathcal{S}} L(\theta, x + \delta, y) \right]$$

其中，$\delta$为添加的扰动。

可以看出模型优化过程是一个min-max的过程，里面的 max 过程可以被认为是在$\parallel \cdot \parallel_{\infty} < \epsilon$ 范围内寻找Loss最大的扰动，前面提到的 FGM / FGSM 可以认为只是迭代一次寻找最大值（one-step scheme for maximizing），本文的另一个贡献是提出了多步优化寻找最大值（multi-step），被称作Projected Gradient Descent (PGD)。$x^0$为原始干净输入，$x^1$为干净输入基础上使用下式进行扰动。更新扰动的过程有两种定义。

$$x^{t+1} = \prod_{x+\mathcal{S}} (x^t + \alpha \mathrm{sgn} (\nabla_{x^t} L(\theta, x^t, y))$$

另一种是：

$$\delta^{t+1} = \prod_{\parallel \delta \parallel_\infty \le \epsilon} (\delta^t + \alpha \mathrm{sgn} (\nabla_{\delta^t} L(\theta, x + \delta^t, y))$$

其中，$\prod_{x+\mathcal{S}}$为Project计算，也就是将x的数值范围约束在$x + \mathcal{S}$范围内，实际可以通过`clip()`函数来完成。注意第二种方式里的$\delta = x^t - x^0$是相对于最开始的干净样本总的扰动大小而言的。前者的代码对应：[PGD-pytorch-github](https://github.com/Harry24k/PGD-pytorch/blob/master/PGD.ipynb)，以及[功守道：NLP中的对抗训练 + PyTorch实现](https://fyubang.com/2019/10/15/adversarial-train/)，而FreeAT / FreeLB等官方代码里用到的是后者，而且 PGD 论文里也提到一句：the adversary of choice will be projected gradient descent (PGD) starting from a random perturbation around the natural example，所以这里倾向于后者，毕竟如果说需要随机初始化$\delta_0$，则应该对应的是后者。当然了，不管是哪种方式，都需要保证最终的扰动项$\delta_t < \epsilon$，而$\delta_t = x^t - x^0$。

数学分析中，两种定义方式的区别在于前者的第$t+1$次对抗样本是:$x^{t+1} = x + \delta^t + \nabla_{x^t} + \nabla_{\delta^t}$，而后者对应的是：$x^{t+1} = x + \delta^{t+1} = x + \delta^t + \nabla_{\delta^t}$，显然后者会更好一些。

这个函数中，一种重要的参数是求解 $x^t$ 时所需要的迭代次数，常见的数值是20 / 10等，说明对训练耗时的增加还是挺严重的。

## TRADES

[Theoretically Principled Trade-off between Robustness and Accuracy](https://arxiv.org/pdf/1901.08573.pdf)

定义了一个新的min-max损失函数，用于平衡泛化性与鲁棒性，也就是提高模型在non-adversarial examples 以及 adversarial examples 上的性能。

$$\min_\theta \mathbb{E}_{(x, y) \sim \mathcal{D}} \max_{\parallel \eta \parallel \le \epsilon} \left( \ell (f_\theta (x), y) + \mathcal{L} (f_\theta(x), f_\theta (x + \eta) / \lambda)  \right)$$

对应的 SMART 算法里使用了这种形式。

## 代码实现

主要包含FGM 的代码示例，参考博客是：[功守道：NLP中的对抗训练 + PyTorch实现](https://fyubang.com/2019/10/15/adversarial-train/)。

```python {linenos=start, linenostart=0}
class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}
    
    def attack(self, img, epsilon, emb_name):
        for name, param in self.model.named_parameters():
            if param.requires_grad and embed_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm > 1e-6:
                    r_adv = epsilon * param.grad / norm
                    param.data.add_(r_adv)
    
    def restore(self, embed_name='.embed'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and embed_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
```

使用代码如下：

```python {linenos=start}
fgm = FGM(model)
for batch, label in data:
    loss = model(batch, label)
    loss.backward()
    fgm.attack()
    loss_adv = model(batch, label)
    loss_adv.backward()
    fgm.restore()
    optimizer.step()
    model.zero_grad()
```

PGD的实现实际有两种，一种是对$x^t$求导用于更新adversarial，另一种是对$\delta_t$求导并更新adversarial样本为$x + \delta_{t+1}$。

第一种示例代码如下，即对输入数据求导。这里的参考是上面的博客链接以及这个仓库：[PGD-pytorch - github](https://github.com/Harry24k/PGD-pytorch/blob/master/PGD.ipynb)

```python {linenos=start}
class PGD():
    def __init__(self, model):
        self.model = model
        self.embed_backup = {}
        self.grad_backup = {}
    
    def attack(self, epsilon=1., alpha=0.3, emb_name='.embed', is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and embed_name in name:
                if is_first_attack:
                    self.embed_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm > 1e-6:
                    r_adv = alpha * param.grad / norm
                    param.data.add_(r_adv)
                    param.data = self.project(name, param.data, epsilon)
    
    def project(self, name, data, epsilon):
        r_adv = data - self.emb_backup[name]        # 是相对于原始数据的扰动大小
        norm = torch.norm(r_adv)
        if r_adv > epsilon:
            r_adv = epsilon * r_adv / norm
        return self.embed_backup[name] + r_adv
    
    def restore(self, embed_name='.embed'):
        for name, param in self.model.named_parameters():
            if embed_name in name:
                assert name in self.embed_backup
                param.data = self.embed_backup[name]
        self.embed_backup = {}
    
    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.grad_backup[name] = param.grad.clone()
    
    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]
```

对应的使用过程也稍微麻烦一点。

```python {linenos=table}
pgd = PGD(model)
K = 3
for batch, label in data:
    loss = model(batch, label)
    loss.backward()
    pgd.backup_grad()
    for idx in range(K):
        pgd.attack(is_first_attack=(idx == 0))
        if idx != K - 1:
            model.zero_grad()
        else:
            pgd.restore_grad()
        loss_adv = model(batch, label)
        loss_adv.backward()     # 最后一轮的对抗样本的梯度会被保留
    pgd.restore()               # 恢复 Embedding 参数
    optimizer.step()
    model.zero_grad()
```

第二种 PGD 对应的代码稍微在保证$\parallel \delta \parallel_\infty < \epsilon$时会方便一些，参考代码[Chapter 3 - Adversarial examples, solving the inner maximization](https://adversarial-ml-tutorial.org/adversarial_examples/)。

```python {linenos=table}

def pgd(model, x, y, epsilon, alpha, K):
    delta = torch.zeros_like(x, requires_grad=True)
    for idx in range(K):
        loss = loss_fn(model(x + delta), y)
        loss.backward()
        delta.data = (delta + x.shape[0] * alpha * delta.grad.data).clamp(-epsilon, epsilon)
        delta.zero_grad_()
    return delta.detach()
```
