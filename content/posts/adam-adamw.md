---
title: "从Adam到AdamW"
date: 2021-09-03T20:30:59+08:00
draft: false

mathjax: true

excerpt_separator: <!--more-->
---
Adam算法的实现以及一个主要改进AdamW的原理与实现。<!--more-->

突然觉着NCHW尺寸的张量比 NHWC Layout 的张量更容易理解，因为后者来看，就是N个样本，每个样本 H * W 的空间维度，然后每个空间元素点是一个 C 维的特征向量。NCHW Layout 的话就需要从后向前理解了...

## Adam

对于 Adam 的实现，参考下图即可。

![图-1 Adam的实现](/imgs/adam-adamw/adam0.png)

算法里最后面三行可以通过用下面两个式子代替用来提高计算性能。

$$   \alpha_t = \alpha \cdot \frac{\sqrt{1 - \beta_2^t}}{ (1 - \beta_1^t)} $$
$$\theta_t \leftarrow \theta_{t-1} - \frac{\alpha_t \cdot m_t}{\sqrt{v_t} + \hat{\epsilon}} $$

其中，$\hat{\epsilon} = \frac{\epsilon}{\sqrt{1 - \beta_2^t}}$。此外，$\alpha$学习率设置了每次更新步长$\Delta_t$的置信区间，具体解释可以参考论文。二阶矩的计算中与普通方差的计算差别在于没有减去均值期望，所以被称为uncentered variacen。而最后上面第一个式子说明了两个超参对更新步长的控制，最后一个式子也说明一阶矩、二阶矩对更新步长的控制，如平地中，除式接近于1，陡峭部分除式小于1（因为方差大），更新会更保守一些。

算法的实现里，另一个重要的步骤是对一阶、二阶矩偏置的矫正。即为什么要除以$\sqrt{1 - \beta_{1,2}^2}$这个式子，证明如下。

以一阶矩$m_t$为例进行推导，首先假设$m_0 = 0$，即初始为一个零矩阵。则$t$时刻一阶矩的计算展开为：

$$m_t = (1 - \beta_t) \sum_{i}^{t}\beta_1^{t-i}g_i$$

我们的目标就是矫正$\mathbb{E}[m_t]$与$\mathbb{E}[g_t]$之间的差距。

$$\begin{aligned}
    \mathbb{E}[m_t] = (1 - \beta_1) \mathbb{E}\left[ \sum_i^t \beta_1^{t-i} \cdot g_i \right] \\
    = & \mathbb{E}[g_t] \cdot (1 - \beta_1) \sum_{i}^t\beta_1^{t-i} + \zeta \\
    = & \mathbb{E}[g_t]\cdot (1 - \beta_1) + \zeta
\end{aligned}$$

其中，等比数列求和公式为：$\sum_i^t \beta_1^{t-i} = \frac{1 \cdot (1 - \beta_1^t)}{1 - \beta_1}$；另外一个地方是第一个等式，即$\mathbb{E}[g_i] = \mathbb{E}[g_t] + \zeta_i$，这里主要考虑使用等式右边近似表示等式左边的数值然后加上一个误差项，如果$g_i, g_t$属于独立同分布则这个误差项接近于0；论文里有提到，如果$\mathbb{E}[g_i]$比较稳定的时候，这里的误差项接近于0，或者当$\beta_1$比较小的时候，那么对于很久以前的梯度$g_i$赋予很小的权重。

当不使用这些矫正项的时候，Adam退化为RMSProp + Momentum的优化算法，实验表明，随着$\beta_2 \rightarrow 1$ 时，训练越来越不稳定。下图左侧开始的地方给出了一个示意图，其中绿色为 ground truth，紫色为预测的曲线，可以看到紫色部分在开始的地方偏小。

![图-2 偏置项矫正的作用](/imgs/adam-adamw/adam1.png)

缺点是，像 Adam 这些 adaptive gradient optimization methods 在图像分类等任务上泛化性能不够高。常见的 adaptive gradient methods 包括：AdaGrad，RMSProp，Adam，AMSGrad等。可能的原因包括陡峭的局部最优解的出现或其他自身存在的缺点。

## AdamW

[Decoupled Weight Decay Regularization](https://arxiv.org/pdf/1711.05101.pdf)作者发现，通过解偶weight decay以及基于loss的反向传播两个过程，可以让学习率、weight decay 两个超参的选取解偶，并且极大提高 adam 优化器在分类任务上的泛化性能，与 SGD + Momentum 取得类似的效果。

既然泛化性能不够，那说明正则化强度不够，所以作者就想到了对 L2 / weight decay 在 Adam 中的使用进行了研究。论文研究了 L2 正则项与 weight decay 在 SGD / Adam 中作用的异同，包括下面几点。

* L2 正则化项与 weight decay 的实现是不同的
* Adam中L2正则项作用不明显
* SGD中 L2 与 weight decay 效果类似
* 关于Adam中 Weight decay的选取，一般来说如果训练需要的迭代次数(iteration)越多，这个数值应该越小
* Adam配合全局的学习率调整可以进一步提高性能，比如 cosine annealing等

考虑[L2与WD区别](#l2正则化与weight-decay的区别)，论文提出了 SGDW 优化算法，用于解耦weight decay实现中依赖于学习率来计算l2参数。主要改动在于：惩罚项从计算 Momentum 计算之前移动到之后了；在不考虑momentum的实现时，图-3中的算法那才时 weight decay 的真正实现，至于所说的解耦的好处，这也是 weight decay 本身自带的优势。(当红色起作用时，绿色不起作用；或者相反)

![图-3 SGDW的实现](/imgs/adam-adamw/sgdw0.png)

但是对于Adam的实现来说，带有weight decay的实现与 L2 Loss 计算梯度无法等价，这一点可以通过对 L2 loss 求导替换$\nabla f_t(\theta)$看出来，若需要两者等价，则L2的系数需要满足:

$$\lambda' = \frac{\lambda}{\alpha \mathbf{M}_t}$$

才行。在 SGD 中，L2 与 Weight Decay 的作用都是让权重接近于0，但是在 Adaptive gradient algorithms中，L2 却不会这样，下个式子给出了带有 L2 正则项的Loss函数求导然后更新权重的过程。

$$\theta_{t+1} \leftarrow \theta_t - \alpha \mathbf{M}_t \nabla f_t(\theta_t) - \alpha \mathbf{M}_t\lambda' \theta_t$$

也就是说，L2 惩罚项也会被用于计算梯度，然后这个计算的梯度被用于 Adam 更新算法中，惩罚项并不会直接作用于权重上，而是参与到自适应一阶、二阶矩$\mathbf{M}_t$的调整中。而在 weight decay 的实现中，没有这个问题，因为$\mathbf{M}_t$只依赖于梯度，而weight decay起作用的方式在于$(1 - \lambda)g_t$，具体公式见论文的Proposition2部分以及对应的附录部分。

为什么 L2 惩罚项这种使用方式不明显呢？首先明确一点是，weight decay / L2 都是为了获得更小的权重值，这样模型泛化会更好一些，为了实现这一点，就需要让权重较大的元素下降的更快一些，权重已经比较小的元素更新小一些。来看 L2 在 Adam 中的具体作用。

[Why AdamW matters](https://towardsdatascience.com/why-adamw-matters-736223f31b5d) 博客里最后一个公式，相当于对权重的exponential moving更新过程是：

$$（1 - \frac{\alpha ( 1 - \beta_1)w}{\sqrt{v_t} + \epsilon})x_{t-1}$$

所以当梯度较大的时候，$\sqrt{v_t}$也比较大，导致该权重的更新（exponential moving）比较小，且还不如那些梯度很小的权重变化大，所以L2作用被打折扣，这也是论文里提到的，拥有大梯度的权重x被惩罚的力度还不如其它权重。最后论文的prosition3 给出了weight decay 与 L2 作用等价时候的关系，但是实际不会在 adaptive gradient algorithms 中实现，而且按照公式关系就可以实现对大的权重进行惩罚，尤其是那些历史梯度都比较大的权重。

那么可不可以理论证明 Weight Decay 的效果确实优于 L2 呢？见[bayesian filtering](#从-bayesian-filtering-的角度看待weight-decay)

图-4给出了准确的 adam with weight decay 的实现。

![图-4 AdamW的实现](/imgs/adam-adamw/adamw1.png)

很明显的看出来，现在 weight decay 已经跟一阶、二阶矩没有关系了。论文中，作者实验证明，即使 Adam 会自适应的调整学习率，但是加上一个全局的乘积因子，如 Cosine Annealing 等，效果会更好。

### 从 Bayesian Filtering 的角度看待Weight Decay

论文引用[Aitchison 2018](https://openreview.net/forum?id=BygREjC9YQ)论文的观点，将模型参数的优化过程看做是求解一系列权重参数最优分布的过程。并将训练过程看做是一系列根据训练数据求解模型权重最大似然的过程$P(\theta_t | y_{1..t})$，状态转移函数是一个训练数据无关的过程，即从$P(\theta_{t+1} | \theta_t)$；$\theta$被认为是state，$y_{1..t}$被认为是观测结果。这里还需要更详细的整理。

对于Bayesian Filtering的一个发展是大名鼎鼎的卡尔曼滤波，此外还有其它更复杂的非线性情况下的扩展等实现，这方面一个比较好的教材是：[State Estimation for Robotics](http://asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser17.pdf)，上面提到的状态的预测、更新在这本书里都会有比较详细的支撑内容。

补充书上式3.3推导过程中第一个等式（下图）的来源，参考[Bayes rule with multiple conditions](https://math.stackexchange.com/questions/408774/bayes-rule-with-multiple-conditions)。

![图-5 Bayes推导公式](/imgs/adam-adamw/adamw0.png)

这可以基于chain rule 来思考：

$$P(a, z, b) = P(a, z | b)P(b) = P(a | z, b) P(z, b) = P(a | z, b)P(z|b)P(b)$$

所以就有了：

$$P(x | v, y) = \frac{P(y, x, v)}{P(v, y)} = \frac{P(y|x, v)P(x|v)P(v)}{P(y|v)P(v)}$$

多说一句，[Multiple View Gemometry in Computer Vision](http://www.r-5.org/files/books/computers/algo-list/image-processing/vision/Richard_Hartley_Andrew_Zisserman-Multiple_View_Geometry_in_Computer_Vision-EN.pdf)与上面这本书是视觉SLAM领域两个非常重要的参考读物，一本讲解了从视觉图像上得到三维空间信息，另一本讲解了三维空间信息的使用、变换、误差优化等信息，完美相互配合。

### 其它的一些改进

论文里，作者用实验表明训练不同的迭代次数时，最优的 weight decay 也是不同的。作者提出了“Normalized Weight Decay”，计算如下：

$$\lambda = \lambda_{norm}\frac{b}{BT}$$

其中，$B$是 training point，T是total epochs数，b是batch size。training point 是单个 epoch 内迭代更新的次数（我的理解）。

其它的改进包括 AdamW with Warm Restart and Cosine Annealing等，一个例子如下。

$$\eta_t = 0.5 + 0.5\cos (\pi T_{cur} / T_i)$$

可以配合上面的 normalized weight decay 对$\lambda$进行调整。Warm Restart 的好处是可以保留前几次的信息，形成一个 momentum，具体可以参考论文。此外，附录B, C部分给出了一个例子，用来说明每次 Restart 时 T 增大两倍，以及 $\eta$ 的变化过程。

## L2正则化与weight decay的区别

注意，L2 正则化项与 Weight Decay 在 SGD 中的使用是两个不同的公式，虽然两者之间存在一个基于学习率的系数差别。结论就是，在 Adaptive Gradient methods 中 L2 的正则化能力弱于weight decay。

关于 weight decay，一般来说 CNN 模型会选取一个比较小的数值，如 10-4 / 10-5 这种量级，但是对于 transformer 模型来说，这个数值一般都会在10-2量级，直观的反映了模型容量对正则化强度的需求变化。

下面来看具体的区别。首先是定义，L2正则化项的定义是修改 Loss 函数，即：

$$f_t^{reg}(\mathbf{\theta}) = f_t(\theta) + \frac{\lambda'}{2}\parallel \theta \parallel^2$$

其中$f_t(\theta)$为 Loss 函数，$\theta$为模型的权重。而SGD中 weight decay 对应的实现是在参数更新的时候起作用，即：

$$\theta_{t+1} = (1 - \lambda) \theta_t + \alpha \nabla f_t(\theta_t) $$

可以看出来是在原来的权重基础上做了一个 expontional decay 的类似计算，$\alpha$为学习率。对 L2 正则项的Loss函数求导并更新权重的公式如下：

$$\theta_{t+1} = \theta - \alpha \nabla f_t(\theta_t) - \alpha \lambda' \theta$$

结合上面两个参数更新过程的式子，可以得出，当L2正则项的参数满足$\lambda' = \frac{\lambda}{\alpha}$条件时，这两个正则方式是等价的。另一方面，在 SGD 中，L2 与 weight decay 的参数（$\lambda'$与$\lambda$）是紧密关联的，前者是后者除以学习率得到。


## 其它

1. 为什么weight decay正则化项不会包含 bias 项？

    参考下面的[答案](https://stats.stackexchange.com/questions/153605/no-regularisation-term-for-bias-unit-in-neural-network)。

    > Overfitting usually requires the output of the model to be sensitive to small changes in the input data (i.e. to exactly interpolate the target values, you tend to need a lot of curvature in the fitted function). The bias parameters don't contribute to the curvature of the model, so there is usually little point in regularising them as well. 

    事实上， bias项一般会用 mean = 1 的随机数进行初始化，而不是 mean = 0。

2. 另有Adam的一个改进时：[AMSGrad](https://openreview.net/pdf?id=ryQu7f-RZ)，用于改进Adam很多时候仅能收敛到局部最优点的问题。
