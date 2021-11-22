---
title: "DALI简单实用案例"
date: 2021-11-22T17:20:46+08:00
draft: false

majax: true
excerpt_separator: <!--more-->
---
DALI(NVIDIA Data Loading Library)库是NVIDIA提供的用于加速数据加载过程的代码库，支持在GPU上完成一些数据处理，从而提高加载速度；另一方面是方便多种源数据文件格式的加载，包括MXNet RecordIO / TFRrecord / LMDB 或者 文件目录等形式的数据集加载；第三就是支持多种数据格式，包括图片、视频、音频等。<!--more-->

## 总览

使用DALI，大体分为三个步骤：

* 定义预处理的 Graph
* 编译Graph，用于Engine执行
* （可选）使用迭代器进行封装，进行数据遍历

前两个步骤都是基于`Pipeline`类来完成，定义Graph对应的是定义新的 Pipeline 对象，下面会提到三种方式；编译的过程通过调用`Pipeline.build()`函数来实现。第三个步骤是可选的，DALI提供一些Iterator类，接受一个Pipeline作为参数来，然后进行迭代，其实通过`Pipeline.run()`也可以得到新的元素。

## 一些概念

### Pipeline

Pipeline就是定义数据预处理的对象，包含预处理的计算图定义、执行引擎。用户通过定义新的 Pipeline 来实现新的数据预处理过程，定义新的 Pipeline 的方式有三种：

* 使用修饰器`pipeline_def()`进行定义
* 定义`Pipeline`类，并借助`Pipeline.set_outputs()`函数来指定Pipeline的输出变量列表
* 通过继承`Pipeline`类，并重定义派生类的`define_graph()`函数来定义新的计算图(这是传统的实现方法)

上面三种方式都可以实现将预初期定义为**当前** Pipeline 的计算图，也就是说，`define_graph()`函数定的计算天然的属于当前的 Pipeline；使用第一种中的修饰器也是；对于第二种，则可以使用`with`语句实现，如下例：

``` python {linenos=table}
pipe = dali.Pipeline(batch_size=N, num_threads=3, device_id=0)
with pipe:
    src = dali.ops.ExternalSource(my_source, num_outputs=2)
    a, b = src()
    pipe.set_outputs(a, b)
```

其中用到的`Pipeline`类的重要参数包括：`batch_size`、`num_threads`、`device_id`等，`device_id=None`表示不实用GPU，其他的包括一些性能方面考虑的参数：`set_affinity、max_streams、bytes_per_sample`等。对应上面代码，使用第一种的等价实现代码如下：

``` python
@dali.pipeline_def(batch_size=N, num_threads=3, device_id=0)
def my_pipe(my_source):
    return dali.fn.external_source(my_source, num_outputs=2)

pipe = my_pipe(my_source)
```

Pipeline 中的计算图（Graph）包含两类节点:

* Operators
* Data Nodes: 支持像Python一样的Indexing，并且可以数学计算

Pipeline 按照 Stage 来划分计算流程，包括三种 Stage:

* cpu: CPU输入、CPU输出，通过调用`.gpu()`来将数据拷贝到 GPU 上
* gpu: GPU输入、GPU输出
* mixed： CPU输入、GPU输出

注意，位于GPU上的数据在Graph中是不能被传回CPU的，只能后面使用``。大多数的DALI操作都会接收命名参数，支持的命名参数的种类有两个：

* Python constants
* Argument Input，这是特指的名字，必须是 CPU 操作的输出的变量，变量位于 CPU 上

对应于由Graph编译得到 Engine，通过`build()`成员函数来实现。通过调用`run()`函数可以获得新的迭代元素进行处理，这一步得到的数据类型是下面提到的`TensorList`格式。

### TensorList

`Pipeline.run(), Pipeline.outputs(), Pipeline.share_ouptuts()`等函数返回的数据的类型是`TensorList`，表示一个batch的Tensor；`Pipeline.releast_outputs()`让当前的 Tensor 不在可用，也就是说 DALI 将可以使用对应的存储资源来存放其他的变量，而且当前 iteration 的 Tensor 在下一个 iteration 也会变得不可用，只能保存到其他数据里（如 Torch 的 Tensor / MXNet 的 NDArray）才可以另作他用。

TensorList 包含两个具体的子类，`TensorListCPU / TensorListGPU`，顾名思义不赘述。重要的成员函数包括`at(), as_array()`等，前者类似获取列表对应位置索引处的数据，后者用于将 Tensor 转换为numpy array。`TensorListGPU`的`as_cpu()`来将数据拷贝到  `cpu`端。`layout()`函数获取当前TensorList的数据存储格式，如HWC / CHW等。

与DataNode的区别。DataNode是TensorList的一个表示符号，并不真正的包含数据，只用于Graph定义阶段，也就是链接两个 Operator。既然 TensorList 在Graph定义阶段使用DataNode表征的，所以用于Graph定义的数学计算操作也是对DataNode进行处理的，包括三角函数、指数/对数、开方、N次方、floor、ceil、clamp、加减乘除等操作。注意这些数学表达式中至少有一个参数是DALI Operator的输出才行，其他的输入可以是`nvidia.dali.types.Constant`或者常规的 Python 常量数据；此外，所有的计算操作都是 element-wise 的方式进行的，支持广播。

多说一些 DataNode。DataNode支持indexing / slicing，并且支持负数的索引或者表示范围等。

### Reader

DALI的一个优势就是对各种数据的文件类型进行了抽象与统一，也就是使用特定的函数/类完成数据的加载以后，后续的所有的处理操作都可以复用。Reader的意思就是从磁盘文件进行数据解析、加载的函数，**一般是Graph定义的第一个步骤**。几个常见的Reader是：

* nvidia.dali.fn.readers.mxnet()，读入`RecordIO`格式的数据，输入`.rec / .idx`文件路径
* nvidia.dali.fn.readers.tfrecord()，读入`TFRecord`格式数据；这里需要先调用`tfrecord2idx`脚本来生成`.idx`文件，然后作为参数使用。`tfrecord2idx`脚本的实现可以帮助理解一下tfrecord文件的布局，整体思路与 RecordIO 的类似，但是应该是使用了 Protobuf 进行了编码，所以解析过程不够直观
* nvidia.dali.fn.readers.caffe()，读入`LMDB`格式数据
* nvidia.dali.fn.readers.file()解码文件目录等

这些Reader的创建函数都会接收两个参数，`shard_id`以及`num_shards`，后者表示将原始数据分成多少份、前者表示当前是第几部分，可以用于多进程训练时数据的分配。此外，在定义过程中一般使用`name`参数来指定Reader的名称，可用于后面的Iterator等。

Reader函数都接收`random_shuffle`参数，用于表明是否对数据集进行随机打乱。这里所说的随机打乱并非进行 Global 层次的打乱，而是在参数`initialize_fill`参数指定的 buffer 里进行打乱，也就是Local 层次的随机打乱；当然如果有多个文档输入的时候，会现在文档层次进行打乱。

另一个比较特殊的是`dali.ops.ExternalSource`可调用类或`dali.fn.external_source()`函数的使用，在上面 Pipeline 的代码实例里也提到了，可以接受一个 Python 实现的 Iterator 来产生数据，用于封装在 DALI 中使用，主要的参数就是`num_outputs`这个了，用于表明这个 Iterator 有几个输出。

### Iterator

上面提到，Reader作为数据来源，是 Graph 定义的第一个步骤；然后就是其他操作的定义以及 Graph 的编译；第三个步骤是可选的，主要是用一个新的 Iterator 进行封装，然后迭代产生**特定于DL框架**的数据格式，比如 MXNet 的 NDArray、Torch的Tensor等。以MXNet为例进行说明，DALI提供了两个主要的类，一个用于生成简单分类任务的数据，输出只有两个变量：Image / Label；另一个是更加通用的迭代器，可以生成多个多个输出。

* nvidia.dali.plugin.mxnet.DALIClassificationIterator()

  该类用于分类任务，只输出两个变量，分别是 data 与 label，类型是MXNet中的 DataBatch of NDArrays。

* nvidia.dali.plugin.mxnet.DALIGenericIterator()

  更加通用的 Iterator，可以输出任意数量的MXNet's DataBatch of NDArrays格式的数据。

注意，这两种 Iterator 返回的 DataBatch 数据的所有权仍然属于DALI，并且只在当前的 Iteration 里有效，如果想在其他 Iteration 中使用，需要将它拷贝到其他 NDArray 里保存才行。

至于针对 Pytorch 提供的 Iterator 也是这两个，功能与MXNet类似。

### 其他一些操作

其他的包括数据解码函数：

* nvidia.dali.fn.decoders.audio()
* nvidia.dali.fn.decoders.image()
* nvidia.dali.fn.decoders.image_crop()
* nvidia.dali.fn.decoders.image_random_crop()，比先使用`image()`函数进行解码然后使用`crop()`的方式会更高效一些，即使用`libjpect-turbo / nvJPEG`等库提供的 ROI 解码函数，也就是只解码特定区域的图像数据
* nvidia.dali.fn.decoders.image_slice()
* 

生成随机数的函数：

* nvidia.dali.fn.random.coin_flip()
* nvidia.dali.fn.random.normal()
* nvidia.dali.fn.random.uniform()

数据简单变换函数：

* nvidia.dali.fn.transforms.combine()
* nvidia.dali.fn.transforms.crop()
* nvidia.dali.fn.transforms.rotation(0)
* nvidia.dali.fn.transforms.scale()
* nvidia.dali.fn.transforms.shear()
* nvidia.dali.fn.transforms.translation()

## 支持的图像增广计算

DALI提供了也还算多的图像增广系列函数，包括：

* 对比度调整
* 颜色空间转换
* HSV，通过设置参数可以实现随机灰度化的功能
* 插值算法
* Resize操作
* Warp Affine
* 3D Transforms操作等

其他通用的操作提供了`fn.normalize()`、`nvidia.dali.fn.crop_mirror_normalize()`等函数来实现Normalize。像后者一样，DALI还提供一些融合多个功能的Operator，这个函数实现的是随机裁剪、翻转、Normalize，而且这个函数支持改变输出数据的Layout，比如输入是 HWC，可以指定输出的Layout格式是CHW，如果只想实现Layout的转换，则可以将Crop / Mirror / Normalize对应的参数设置为不起作用即可，至于如何设置可以参考API文档，保持默认值即可。

可以发现，DALI目前还没有RandAug / AutoAug / CutMix / Mixup 等复杂操作的实现的，文档里也提供了相应的自己实现新 Op 的说明，可以说还是非常的yin性化的。

## 一些具体的例子

这个例子以实现 Torch 框架下模型使用 RecordIO / TFRecord 等源数据文件格式进行训练的方式进行说明。

首先是定义数据预处理的 Graph，这部分可以分为两部分，第一部分是针对两种文件格式的数据加载函数、另一部分是对加载的数据进行处理的部分，后者可以公用。

公用的数据处理方式：

``` python
def common_pipeline(jpegs, labels):
    images = fn.decoders.image(jpegs, device='mixed')
    images = fn.resize(
        images,
        resize_shorter=fn.random.uniform(range=(256, 480)),
        interp_type=types.INTERP_LINEAR)
    images = fn.crop_mirror_normalize(
        images,
        crop_pos_x=fn.random.uniform(range=(0.0, 1.0)),
        crop_pos_y=fn.random.uniform(range=(0.0, 1.0)),
        dtype=types.FLOAT,
        crop=(227, 227),
        mean=[128., 128., 128.],
        std=[1., 1., 1.])
    return images, labels
```

读取RecordIO数据：

``` python
@pipeline_def
def mxnet_reader_pipeline(num_gpus):
    jpegs, labels = fn.readers.mxnet(
        path=[db_folder+"train.rec"],
        index_path=[db_folder+"train.idx"],
        random_shuffle=True,
        shard_id=Pipeline.current().device_id,
        num_shards=num_gpus,
        name='Reader')

    return common_pipeline(jpegs, labels)
```

读取TFrecord数据：

``` python
import nvidia.dali.tfrecord as tfrec

@pipeline_def
def tfrecord_reader_pipeline(num_gpus):
    inputs = fn.readers.tfrecord(
        path = tfrecord,
        index_path = tfrecord_idx,
        features = {
            "image/encoded" : tfrec.FixedLenFeature((), tfrec.string, ""),
            "image/class/label": tfrec.FixedLenFeature([1], tfrec.int64,  -1)},
        random_shuffle=True,
        shard_id=Pipeline.current().device_id,
        num_shards=num_gpus,
        name='Reader')

    return common_pipeline(inputs["image/encoded"], inputs["image/class/label"])
```

最后是数据迭代：

``` python
import numpy as np
from nvidia.dali.plugin.pytorch import DALIGenericIterator


pipe_types = [
    [mxnet_reader_pipeline, (0, 999)],
    [tfrecord_reader_pipeline, (1, 1000)]]

for pipe_t in pipe_types:
    pipe_name, label_range = pipe_t
    print ("RUN: "  + pipe_name.__name__)
    pipes = [pipe_name(
        batch_size=BATCH_SIZE, num_threads=2, device_id=device_id, num_gpus=N) for device_id in range(N)]
    dali_iter = DALIGenericIterator(pipes, ['data', 'label'], reader_name='Reader')

    for i, data in enumerate(dali_iter):
        # Testing correctness of labels
        for d in data:
            label = d["label"]
            image = d["data"]
            ## labels need to be integers
            assert(np.equal(np.mod(label, 1), 0).all())
            ## labels need to be in range pipe_name[2]
            assert((label >= label_range[0]).all())
            assert((label <= label_range[1]).all())
    print("OK : " + pipe_name.__name__)
```

至于更多的细节可以参考DALI的官方文档。
