---
title: "Adversarial Training 2"
date: 2021-10-13T14:30:10+08:00
draft: false
mathjax: true

excerpt_separator: <!--more-->
---

对PGD算法的改进，包括FreeAT, FreeLB, SMART等。<!--more-->

博客[Adversarial Trainig](/content/posts/adversarial-training.md)里面的PGD模型在对抗样本训练中可以起到很好的效果，但是缺点在于求解最优扰动时需要进行迭代，比如迭代K次，则意味着训练耗时增加K + 1倍。本文提到的几个算法主要就是针对这个问题进行优化，包括 FreeAT, FreeLB, SMART 等算法。

## FreeAT

首先在PGD算法中，为了得到$\max_{\delta \in \mathcal{S}} L(\theta, x + \delta, y)$，需要对 adversarial samples 进行迭代生成，$x^{t+1} = \prod_{x + \mathcal{S}} (x^t + \alpha \mathrm{sign} \nabla_x L(\theta, x, y))$。FreeAT主要是为了降低后面迭代求解最优Adversarial Samples的过程。

对于一个 PGD-K 算法而言，每一个batch的训练中，需要计算K次前向-后向，每次前向-后向会更新一下最新的扰动数值，这个扰动数值也会累加，然后使用第K次的adversarial examples计算$L_{adv}$以及 第 K + 1 次前向计算 $L$，最终反向传播一次$L + L_{adv}$。

而对于FreeAT算法，在K次前向-后向计算计算最优扰动项的时候，也会使用$L$进行一次前向-后向传播，并且总的 Epoch 数降低为$ N / K$，N 为正常训练时需要的 epoch 个数。与PGD-K的区别是，每次计算扰动项的时候，模型的参数也会更新，同时降低总的训练epoch数，保证总的iteration数不变。在训练下一个 Batch 的时候，上一个Batch最新的扰动数值作为当前Batch的初始扰动数值。

为了避免参数遗忘，这里的 $FreeAT-K$ 中的K不能太大。

最终，FreeAT 的算法如下。

![图 - 1 FreeAT算法伪代码](/imgs/adversarial-training/freeat0.png)

模型的超参数$m$（也就是上文的$K$）是重要的超参数，图-2展示了在 CIFAR-100数据集上的影响。可见随着 $m$ 的增加，精度会下降，但当$m<10$的时候对精度影响都比较小，而且别PGD-7的效果都好，可以作为备选范围。

![图 - 2 FreeAT算法中m的影响](/imgs/adversarial-training/freeat1.png)

在ResNet-50 + ImageNet 的配置下，当$m=4$的时候效果最好，这一切都是不像PGD那样增加训练计算量的前提下实现的。

### Gradient Masking

Gradient Masking 的意思是指模型的输出对输入的梯度趋近于零，所以当输入发生微小变化时，不会影响模型的输出，从而实现鲁棒性。但是这种方法并没有真正的提高模型的鲁棒性，因为考虑到对抗样本的 transferability，换成另一个模型，这个模型对输入的梯度不接近于零，导致用这个模型生成的对抗样本导致目前正在训练的模型还是会预测错误。如下图所示，

![图 - 3 Gradient Masking导致的后果](/imgs/adversarial-training/gradientmasking0.png)

其中，(a) 中的模型对与输入 $x$ 附近的梯度已经为0了，所以此时对 $x$ 的扰动有一定的鲁棒性，但是当使用另一个模型 (b) 对 $x$ 的梯度来生成对抗样本 $x^*$ 时，基于transferability，这个对抗样本对模型 (a) 仍然是有效的。所以当发生 Gradient Masking 时，模型并非是真正的鲁棒。

Label smoothing 一定程度上也可以提高对抗样本效果，因为知识蒸馏之类的Loss可以让学习到的模型更平滑，也就是对输入更不敏感。

更多的可以参考：[SoK: Towards the Science of Security and Privacy in Machine Learning](https://arxiv.org/pdf/1611.03814.pdf)

## FreeLB

首先，FreeAT那种在每次更新扰动项$\delta$的时候都会更新模型的权重（梯度下降），这会导致`stale gradient`的发生，也就是对于第 $t$ 步，扰动的更新不是最大化模型在 $t$ 时刻的参数$\theta_t$，而是基于下式$\nabla_{\delta} L(f_{\theta_{t-1}}(x + \delta_{t-1}), y)$计算得来的(这个梯度计算公式还是以PGD那里的为准)。

FreeLB算法其实对计算量并没有减少，主要是提出了另一种梯度更新过程。在PGD-K的K次前向-后向计算用于构造adversarial examples时，FreeLB会累加每次后向传播中模型参数的，最后使用这个累加的梯度更新模型的参数。总的来说，前向-后向次数由K + 1次变为 K 次。另一个好处是，这个累加的梯度可以包含更多扰动的信息，可以认为每次模型的梯度更新都使用了更大的Batch的样本计算得到，即$x+\delta_0, \ldots, x + \delta_{K-1}$，而PGD算法只能最小化$x+\delta_{k-1}$位置的扰动损失，理论认为这会比PGD得到更好的泛化性能。

上述改动等价于在两个高维球里求解最优的扰动：

$$\mathcal{I}_ t=\mathcal{B}_{x+\delta_0}(\alpha t) \cap \mathcal{B}_{x}(\epsilon)$$

其中$\mathcal{B}_x(\epsilon)$表示半径为$\epsilon$的球。而通过梯度累加移动平均，则等价于优化下面的损失函数。

$$\min_\theta \mathbb{E}_{(z, y) \sim \mathcal{D}} \left[  \frac{1}{K} \sum_{t=0}^{K-1} \max_{\delta_t \in \mathcal{I}_t} L(f_\theta(x + \delta_t), y) \right]$$

使用FreeLB算法需要特别注意的地方在于，在包含Dropout的模型中，需要保证 K 次前向-后向计算时 Dropout 的Mask保持一致，否则的话，得到的扰动就不是针对某一模型的最优扰动了。所以，使用时需要保证在一个Step内，Dropout用到的 Mask 保持不变。

最终，对应的FreeLB算法伪代码如下。

![图 - 4 FreeLB算法伪代码](/imgs/adversarial-training/freelb0.png)

需要说明的是，论文里提到的 PGD 与原文中的公式定义不太一致。

$$\delta_{t+1} = \prod_{\parallel \delta \parallel_F \le \epsilon} (\delta_t + \alpha g(\delta_t) / \parallel g(\delta_t) \parallel_F)$$

其中，$g(\alpha_t) = \nabla_{\delta} L(f_\theta(x + \delta_t), y)$，这里定义的是对扰动的梯度，而不是对输入$x$的梯度。这一点FreeLB对应的代码里是对应论文里的公式的，需要找PGD的官方实现进行验证。

## SMART

[SMoothness-inducing Adversarial Regularization](https://arxiv.org/pdf/1911.03437.pdf)

SMART论文主要提出了两个创新点。

* Smoothness-inducing adversarial regularization

  这个正则项主要是为了提高模型的鲁棒性。

* Bregman proximal point optimization

  这里是为了提高模型的效果，类似`Mean Teacher`。

### Smoothness-inducing adversarial regularization

提出优化下面的损失函数，

$$\min_\theta \mathcal{F}(\theta) = \mathcal{L}(\theta) + \lambda_s \mathcal{R}_s (\theta)$$

其中，$\mathcal{L}(\theta)$为正常损失函数，是基于数据对$(x_i, y_i)$的有监督损失函数。

$$\mathcal{L}(\theta) = \frac{1}{n}\sum_{i=1}^n \ell(f(x_i; \theta), y_i)$$

$\mathcal{R}_s(\theta)$为smoothness-inducing adversarial regularizer项：

$$\mathcal{R}_ s(\theta) = \frac{1}{n} \sum_{i=1}^n \max_{\parallel \tilde{x}_i - x_i \parallel_p \le \epsilon} \ell_s(f(\tilde{x}_i; \theta), f(x_i; \theta))$$

最新的SMART论文里，基于TRADES论文中的损失函数定义$\mathcal{R}_s$，适用于模型输出端是概率分布的情况。

$$\elll_s(P, Q) = \mathcal{D}_{KL} (P \parallel Q) + \mathcal{D}_{KL} (Q \parallel P)$$

当模型的输出是一个Scalar，也就是做类似回归任务时，有：

$$\ell_s(p, q) = \parallel p - q \parallel^2$$

### Bregman Proximal Point Optimization

这一步主要是为了防止模型的参数在每个Step中更新过大。Transformer模型中的小的 lr 本身也是一种正则化，也就是让模型的参数不会变化非常大，提高模型的泛化性能、鲁棒性等。

Vanilla Bregman Proximal Point (VBPP) 算法定义了模型参数更新过程：

$$\theta_{t+1} = \mathrm{ArgMin}_{\theta} \mathcal{F}(\theta) + \mu \mathcal{D}_{Breg}(\theta, \theta_t)$$

其中，$\mu > 0$，$\mathcal{D}_{Breg}$用于阻止模型的参数更新过大：

$$\mathcal{D} _{Breg}(\theta, \theta_t) = \frac{1}{n}\sum _{i=1}^n \ell_s (f(x_i; \theta), f(x_i; \theta_t))$$

实际使用中，可以借助Momentum Update的方式来更新参考的模型权重：

$$\tilde{\theta}_t = (1 - \beta)\theta_t + \beta \tilde{\theta}_{t-1}$$

然后计算$\mathcal{D}_{Breg}(\theta, \tilde{\theta}_t)$。这个过程与EMA非常相似，在自监督学习中经常被使用。

### 伪代码

SMART算法实现的伪代码如图-5。

![图 - 5 SMART算法伪代码](/imgs/adversarial-training/smart0.png)

## 其它

* YOPO: [You Only Propagate Once: Accelerating Adversarial Training via Maximal Principle](https://arxiv.org/pdf/1905.00877.pdf)

  通过分析Pontryagin’s Maximum Principle，观察到每次对抗样本的更新只与模型的前几层的梯度有关，基于这个观察，作者提出了 YOPO 算法。

  关于 PMP & Hamiltonian 看不懂，略过。

  > Momentum should be accumulated between mini-batches other than different adversarial examples from one mini-batch, otherwise overfitting will become a serious problem.

* ALUM: [Adversarial training for large neural LangUage Models](https://arxiv.org/pdf/2004.08994.pdf)

  这篇文章就提到了，对抗训练一方面是为了提高鲁棒性，另一方面是为了提高泛化性。本文提出的模型叫做 MT-DNN，上面的 SMART 算法与这个算法也可以结合起来。本文发现对抗学习也可以提高预训练阶段的效果，。

  发现，virtual adversarial training 比 conventional adversarial trainging 效果更好，尤其是存在 noisy label 的时候。 BERT 预训练的 MLM 就是属于 noisy label 的情况，因为被 mask 的 word 实际上可以有很多的选择。所以 SMART 中的 $\lambda$ 在预训练阶段会比较大，比如 = 10，微调阶段为 = 1。

## 代码实现

先来看下 FreeLB 的代码实现，参考：[FreeLB - github](https://github.com/zhuchen03/FreeLB)。在这个[run_glue.sh](https://github.com/zhuchen03/FreeLB/blob/master/huggingface-transformers/launch/run_glue.sh) 脚本中，不同的下游任务对 $\delta$ 的初始化也不同，比如全零，或者`uniform()`等。

代码里的亮点在于：(1) 首先获取输入文本的 Embedding 表示 (2) 初始化 $\delta$ (3) 迭代求解最优的对抗样本。

```python {linenos=table}
for step, batch in enumerate(dataloader):
    if isinstance(model, torch.nn.DataParallel):
        embeds_init = model.module.encoder.embeddings.word_embeddings(batch[0])
    else:
        embeds_init = model.encoder.embeddings.word_embeddings(batch[0])
    if args.adv_init_mag > 0:

        input_mask = inputs['attention_mask'].to(embeds_init)
        input_lengths = torch.sum(input_mask, 1)
        # check the shape of the mask here..

        if args.norm_type == "l2":
            delta = torch.zeros_like(embeds_init).uniform_(-1,1) * input_mask.unsqueeze(2)
            dims = input_lengths * embeds_init.size(-1)
            mag = args.adv_init_mag / torch.sqrt(dims)
            delta = (delta * mag.view(-1, 1, 1)).detach()
        elif args.norm_type == "linf":
            delta = torch.zeros_like(embeds_init).uniform_(-args.adv_init_mag,
                                                            args.adv_init_mag) * input_mask.unsqueeze(2)

    else:
        delta = torch.zeros_like(embeds_init)

    # the main loop
    dp_masks = None
    for astep in range(args.adv_steps):
        # (0) forward
        delta.requires_grad_()
        inputs['inputs_embeds'] = delta + embeds_init
        inputs['dp_masks'] = dp_masks

        outputs, dp_masks = model(**inputs)
        loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
        # (1) backward
        if args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        loss = loss / args.adv_steps        # 求解梯度的平均

        tr_loss += loss.item()

        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if astep == args.adv_steps - 1:
            # further updates on delta
            break

        # (2) get gradient on delta
        delta_grad = delta.grad.clone().detach()

        # (3) update and clip
        if args.norm_type == "l2":
            denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
            denorm = torch.clamp(denorm, min=1e-8)
            delta = (delta + args.adv_lr * delta_grad / denorm).detach()
            if args.adv_max_norm > 0:   # 通常为0 或者 1e-7
                delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1).detach()
                exceed_mask = (delta_norm > args.adv_max_norm).to(embeds_init)
                reweights = (args.adv_max_norm / delta_norm * exceed_mask \
                                + (1-exceed_mask)).view(-1, 1, 1)
                delta = (delta * reweights).detach()            # 进入下一次循环
        elif args.norm_type == "linf":
            denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1, p=float("inf")).view(-1, 1, 1)
            denorm = torch.clamp(denorm, min=1e-8)
            delta = (delta + args.adv_lr * delta_grad / denorm).detach()
            if args.adv_max_norm > 0:
                delta = torch.clamp(delta, -args.adv_max_norm, args.adv_max_norm).detach()
        else:
            print("Norm type {} not specified.".format(args.norm_type))
            exit()

        if isinstance(model, torch.nn.DataParallel):
            embeds_init = model.module.encoder.embeddings.word_embeddings(batch[0])
        else:
            embeds_init = model.encoder.embeddings.word_embeddings(batch[0])

    # ============================ End (2) ==================

    if (step + 1) % args.gradient_accumulation_steps == 0:
        if args.fp16:
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        model.zero_grad()
        global_step += 1
```

对应的 SMART 代码在[mt-dnn](https://github.com/namisan/mt-dnn)。
