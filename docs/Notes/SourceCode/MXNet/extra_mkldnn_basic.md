---
title: MKLDNN 概念整理
---

# MKLDNN 概念整理

smh

2019.12.11 - 12.17

[[toc]]

## 简介

## Basic Concepts

* primitives

  * `dnnl::primitive`
  * A `primitive` is a functor object that encapsulates a particular computation such as forward convolution, backward LSTM computations, or a data transformation operation.
  * 也可能是复杂的多个Operator的融合操作作为一个`primitive`
  * 但是与普通`pure function`的区别在于，`primitives`还可以保存状态，至于那些状态呢？可以是下面的例子：
    * immutable state
      比如`convolution primitives`可以保留参数的形状、预计算依赖的参数，如`cache blocking`. 后者回根据实际计算进行调整，从而优化性能。而且这个预计算的开销只有一次，后面可以重复使用这次预计算的结果。
    * `mutable stats`
      又被称为`scratchpad`，一个`primitive`可以使用这些区域作为临时存储，这些资源可以被`primitive`或运行时参数所掌握。
    *  
  
* engines

  * `dnnl::engine`
  * is an abstractioin of a computational device: a CPU, a specific GPU card in the system etc
  * 就是具体处理单元的抽象，
  * 大部分`primitive`都是在特定的`engine`上完成计算，处理`reorder primitives`或者两个`engine`之间的数据传递
  *  

* streams

  * `dnnl::stream`
  * encapsulate executioin context tied to a particular engine
  * 也就是 用于在`engine`上保存计算的上下文
  *  

* memory objects

  * `dnnl::memory`
  * encapsulate handles to memory allocated on a specific engine, tensor dimensions, data type, and memory format - the way tensor indices map to offsets in linear memory space
  * 也就是说，`memory format`就是`tensor`在存储中索引的映射关系
  * `memory`就是在特定`engine`上保存存储指针、tensorc尺寸、数据类型、存储结构等信息
  * memory objects are passed to primitives during execution
  * 也就是说，memory objects作为参数传入到`primitives`，然后进行计算等操作

*  

### Levels of Abstraction

* logical level， DNN库提供了下面的一些抽象

  * memory descriptors

    * `dnnl::memory::desc`
    * define a tensor's logical dimensions, data type, and the format in which the data is laid out in memory
    * `any(dnnl::memory::format_tag::any)`表示数据的实际`format`会在后面才会定义
    *  

  * operation descriptors

    * 每个支持的`primitive`对应一个`operation descriptors`
    * 描述`operation`的无需制定的基本属性，例如得使用哪一个`engine`计算等。
    * for example, convolution descriptor describes shapes of source, destination, and weights tensors, propagation kind (forward, backward with respect to data or weights), and other implementation-independent parameters
    *  

  * primitive descriptors

    * `dnnl_primitive_desc_t`
    * 在C++ API中，每个`primitive`可能对应多个`primitive descriptors`

    * 抽象层次位于`operation descriptors`与`primitives`之间，可以用于：inspect details of a specific primitive implementation like expected memory formats via queries to implement memory format propagation without having to fully instantiate a primitive

    * 各个抽象层次之间的关系以及对应的类的关系如下图：

      ![image-20191216210723107](/Users/smher/Library/Application Support/typora-user-images/image-20191216210723107.png)

### Creating Memory Objects and Primitives

就是对应上面的两套抽象目标体系。

* Memory Objects
  * 基于`memory descriptor`来创建，但不能通过`dnnl::memory::format_tag::any`这个`memory descriptor`来创建`memory object`
  * 有两种方法来初始化`memory descriptors`
    * 使用`dnnl::memory::desc`构造函数，或使用`dnnl::memory::desc::submemory_desc`来从tensor的一部分来提取`memory descriptor`
    * 通过查询`an existing primitive descriptor`来生成对应`primitive's parameters`的`memory descriptor`，比如使用`dnnl::convolution_fowward::primitive_desc::src_desc`
    *  
  * 在创建`memory objects`时，可以接受一个用户提供的`void *on CPU`指针，或者不提供，那么`DNN`库会自己创建一个存储空间
  *  
* Primitives
  * 创建`primitives`的步骤如下
    * Create an operation descriptor via, for example, `dnnl::convolution_forward::desc`，这个创建的`operator descriptor`可以包含`memory descriptors with placeholder format_tag::any memory formats`，如果`operator primitive`支持这种`memory descriptors`的话
    * Create a primitive descriptor based on the operation descriptor and an engine handle
    * Create a primitive based on the primitive descriptor obtained in step2
    *  
  *  
* 最后是将上面两个东西结合起来完成计算，`memory objects`作为参数然后使用`primitive`进行计算

## Getting started

基本的DNNL编程模型：

* 创建`Engine & stream`，用于支持`primitive`进行计算
* 准备用户数据，并创建`DNNL memory objects`
  * 从用户`buffer`中将数据加载到`DNNL memory object`
  * 将`tensor's logical dimensions`与`memory object formats`进行关联
* 创建`DNNL primitives`
* 执行`DNNL primitives`

### Public Headers

* include `dnnl.hpp`
* `dnnl_debug.h, example_utils.hpp`里面包含了一些用于调试的辅助函数

### getting_started_tutorial() function

* Engine & stream
  
  * All DNNL primitives and memory objects are attached to a particular `dnnl::engine`,`engine`就是特定的计算硬件，`primitives, memory object`针对这个特定的硬件进行优化，所以，`engine, memory object`在不同硬件之间是不能共用的
  
  * 创建`engine`过程需要制定`kind & index`
  
    ``` cpp
    engine eng(engine_kink, 0);
    ```
  
    * Engine kink可选的有
      * any
      * cpu
      * pu
  
  * 除了`engine`，所有的`primitive`都还需要`dnnl::stream`进行计算，每个`stream`包含了运行的上下文并帮钓到特定的`engine`上
  
    ``` cpp
    stream engine_stream(eng);
    ```
  
  * 在最简单的情况，就是一个程序只使用一个`device`,那么，`engine & stream`可以只创建一次，然后整个程序都使用这个已经创建的，比如使用单例模式
  
  *  
  
* Data preparation (code outside of DNNL)

  * 在`DNNL`里面，所有的`primitives`都假设输入的数据有`batch`维，并且是第一个维度
  * 示例代码如下

    ``` cpp
    const int N = 1, H = 13, W = 13, C = 3;
        // Compute physical strides for each dimension
        const int stride_N = H * W * C;
        const int stride_H = W * C;
        const int stride_W = C;
        const int stride_C = 1;
        // An auxiliary function that maps logical index to the physical offset
        auto offset = [=](int n, int h, int w, int c) {
            return n * stride_N + h * stride_H + w * stride_W + c * stride_C;
        };
        // The image size
        const int image_size = N * H * W * C;
        // Allocate a buffer for the image
        std::vector<float> image(image_size);
        // Initialize the image with some values
        for (int n = 0; n < N; ++n)
            for (int h = 0; h < H; ++h)
                for (int w = 0; w < W; ++w)
                    for (int c = 0; c < C; ++c) {
                        int off = offset(
                                n, h, w, c); // Get the physical offset of a pixel
                        image[off] = -std::cos(off / 10.f);
                    }
    ```

* Wrapping data into a DNNL memory object
  
  这一步就是上一步的图像数据封装成`dnnl::memory`对象，然后被`DNNL primitives`使用。创建`dnnl::memory`需要两个步骤：
  
  * 初始化`dnnl::memory::desc`结构(也就是`memory descriptor`)，这个数据结构只是用于描述数据，但并不真正的保存具体的数据，`memory descriptor`主要是用来创建`dnnl::memory`对象，还有就是初始化`primitive descriptors`
  * 创建`dnnl::memory`对象本身，基于上一步的`dnnl::memory::desc` + `an engine` 或者可选的`a handle to data`来实现。`memory object`在`primitive`计算时被使用。
  
  通过使用C++中的`list initialization`，可以把上述两个步骤合并起来完成，只要`dnnl::memory ::desc`不被使使用时，除了创建`dnnl::memory`对象。下面是这两个步骤的具体实现示例。
  
  * memory descriptor
  
    为了初始化`dnnl::memory::desc`我们需要传递的参数有：
  
    * tensor的维度，维度的顺序由具体的`primitive`来确定
  
    * 注意，`memory::desc`或者`memory object`都对于所保存的数据没有特别的理解。
  
    * 数据类型：`dnnl::memory::data_type`
  
    * 存储tag，`dnnl::memory::format_tag`，用于描述数据在实际存储单元上是怎么存储的，也就是`memory format`，这是`primitive`正确处理数据的前提
  
    * 示例代码
  
      ``` cpp
      auto src_md = memory::desc({N, C, H, W},   // logical dims, the order is defined by a primitive
                                 memory::data_type::f32, // tensor's data type
                                 memory::format_tag::nhwc. // memory format, NHWC in this case
                                )
      ```
  
      * 注意，我们需要指定`logical order of dimension`以及`memory format`，后者用于描述`how logical indices map to the offset in memory`
      * 也就是说，`memory::format_tag`描述了输入的用户数据的保存`format`，而`logical dims`是被`primitive`处理的数据`format`
      *  
  
  * alternative way to create a memory descriptor
  
    除了使用上文的`memory::format_tag`来创建`memory::desc`，我们还可以使用直接通过指定每个维度的`strides`的方式来创建`memory::desc`，示例如下：
  
    ``` cpp
    auto alt_src_md = memory::desc(
      {N, C, H, W},  // logical dims
      memory::data_type::fp32,
      {stride_N, stride_C, stride_H, stride_W}
    )
    ```
  
    * 注意，这里的`strides`的顺序是按照`primitive`使用的数据`format`的顺序来给出的，而不是实际输入数据的`NHWC`格式！
  
  * creating a memory object
  
    有了`memory descriptor & engine`，就可以创建用于保存输入、输出的`memory objects`
  
    ``` cpp
    auto src_mem = memory(src_md, eng);
    write_to_dnnl_memory(image, data(), src_mem);

    // for dst mem
    auto dst_mem = memory(src_md, eng);
    ```
  
    需要注意的有两点
  
    * `DNNL`库会具有`dst_mem`的管理权限，用于自动释放，因此需要在`dst_mem`有效的时候使用这个变量
    * DNNL库分配的存储空间都具有良好的`alignment`，可以提高性能，一般来说，输入的用户分配的空间也应该具有良好的配齐，才能有好的性能
    *  
  
* Creating a ReLU primitive

  创建`primitive`的过程也分为三个步骤，虽然在某些情况下可以使用`C++11`的特性来合并某些步骤：

  * 初始化`operator descriptor`，这个例子中为`dnnl::eltwise_forward::desc`，这里面定义了`operation parameters`

  * 创建`primitive descriptor`，这里是`dnnl::eltwise_forward::primitive_desc`，包含了制定具体的算法来完成需要的`operation`。用户可以获取选择的算法的存储开销以及其它一些后面提到的内容(`memory format propagation`)。

  * 创建`primitive`，这里是`dnnl::eltwise_forward`，可以对参数`memory objects`的计算进行操作

  * 总的来说，就是之前提到的三个不同的抽象层次，每个抽象层次对应一个具体的步骤！`memory object`的创建过程也是对应两个抽象层次！！！而已。

  * 注意，`primitive`的创建非常耗时，所以应该只创建一次，然后多次使用！

  * 代码示例如下

    ``` cpp
    //  ReLU op descriptor (no engine- or implementation-specific information)
        auto relu_d = eltwise_forward::desc(
                prop_kind::forward_inference, algorithm::eltwise_relu,
                src_md, // the memory descriptor for an operation to work on
                0.f, // alpha parameter means negative slope in case of ReLU
                0.f // beta parameter is ignored in case of ReLU
        );
        // ReLU primitive descriptor, which corresponds to a particular
        // implementation in the library
        auto relu_pd
                = eltwise_forward::primitive_desc(relu_d, // an operation descriptor
                        eng // an engine the primitive will be created for
                );
        // ReLU primitive
        auto relu = eltwise_forward(relu_pd); // !!! this can take quite some time
    ```

  * 注意，简单的计算如`eltwise, BN`等，可以传入`exact tensor & memory format`，然后创建的`primitive`就使用对应的`memory descriptor`进行计算

  * 其它一些复杂计算，如`convolution`,会自己决定`memory format`

  *  

* Executing the ReLU primitive

  * 输入、输出数据的`memory objects`被传入给`execute()`，通过`<tag, memory>map`的方式。其中，`tag`表示每个`memory object`代表哪种的`tensor`，所有的`eltwise primitives`需要这个`map`包含至少输入、输出两个元素的`memory object`

  * 每个`primitive`在`stream`中执行，也就是`execute()`函数的第一个参数，与`stream`的类型相关，每个`execution`可能是`blocking or non-blocking`的，因此在获取计算结果之前，需要首先执行`dnnl::stream::wait`

    ``` cpp
    relu.execute(engine_stream, {{DNNL_ARG_SRC, src_mem}, // source tag & memory obj
                                 {DNNL_ARG_DST, dst_mem}, // destination tag and memory obj
                                });
    engine_stream.wait(); // wait the stream to complete the execution
    ```

  * `eltwise primitives`支持`in-place operations`，为了使用`in-place`，需要将输入、输出memory object标记成`DNNL_ARG_SRC, DNNL_ARG_DST`

     ``` cpp
     relu.execute(engine_stream, {
       {DNNL_ARG_SRC, src_mem},
       {DNNL_ARG_DST, src_mem},
     })
     ```

  *  

* Obtaining the result and validataion

  就是从`dst_mem`来获取指向结果数据的指针，并进行适当的类型转换，具体函数是`dnnl::memory::get_data_handle()`

  * 代码示例如下

    ``` cpp
        // Obtain a buffer for the `dst_mem` and cast it to `float *`.
        // This is safe since we created `dst_mem` as f32 tensor with known
        // memory format.
        std::vector<float> relu_image(image_size);
        read_from_dnnl_memory(relu_image.data(), dst_mem);
        // Check the results
        for (int n = 0; n < N; ++n)
            for (int h = 0; h < H; ++h)
                for (int w = 0; w < W; ++w)
                    for (int c = 0; c < C; ++c) {
                        int off = offset(
                                n, h, w, c); // get the physical offset of a pixel
                        float expected = image[off] < 0
                                ? 0.f
                                : image[off]; // expected value
                        if (relu_image[off] != expected) {
                            std::cout << "At index(" << n << ", " << c << ", " << h
                                      << ", " << w << ") expect " << expected
                                      << " but got " << relu_image[off]
                                      << std::endl;
                            throw std::logic_error("Accuracy check failed.");
                        }
                    }
    ```

*  

### main() function

* 可以使用异常来处理错误，`DNNL C++ API`可能抛出`dnnl::error`，包含了`error status (of type dnnl_status_t)`以及一个可以人为阅读的`error message accessible through regular what() method`

* 具体的代码示例如下

  ``` cpp
  int main(int argc, char **argv) {
      int exit_code = 0;
      try {
          engine::kind engine_kind = parse_engine_kind(argc, argv);
          getting_started_tutorial(engine_kind);
      } catch (dnnl::error &e) {
          std::cout << "DNNL error caught: " << std::endl
                    << "\tStatus: " << dnnl_status2str(e.status) << std::endl
                    << "\tMessage: " << e.what() << std::endl;
          exit_code = 1;
      } catch (std::string &e) {
          std::cout << "Error in the example: " << e << std::endl;
          exit_code = 2;
      }
      std::cout << "Example " << (exit_code ? "failed" : "passed") << std::endl;
      return exit_code;
  }
  ```

## Memory format propagation

正确使用DNNL库的前提是对`format propagation`具有很好的理解。

* 在创建一些`compute-intensive`的操作时，推荐使用`dnnl::memory::format_tag::any`(也就是placeholder memory format)。它会根据不用的硬件环境、卷积参数进行调整，选择性能最好的`memory format`

* 而像`Elementwise, LRN, BN`以及其它一些计算，在前向传播时，需要使用与前一层相同的`memory format`,这样可以避免不必要的`reorders`，这通常比较耗时。但在后向传播时，这些`primitives`需要与前向过程中使用的`memory format`一致，所以在初始化这些`primitives`用于后向计算时，需要使用`dnnl::memory::format_tag::any`！

* 下表给出了根据`operation description`初始化过程该不该使用`dnnl::memory::format_tag::any`的情况

  ![image-20191217132412361](/Users/smher/Library/Application Support/typora-user-images/image-20191217132412361.png)

*  

### Introduction to the Tutorial

* configure primitives to use optimized memory formats
* Determine whether data needs to be reordered from/to optimized memory formats

具体的实现步骤如下：

* initialize

  ``` cpp
  engine eng(engine_kind, 0);
  stream s(eng);
  ```

* create convolution and pooling primitives

  * 在创建`memory descriptors`时，使用`dnnl::memory::format_tag::any`的memory format。这种方法仅适用于部分`primitives`，如卷积、内积。也可以用于`dstination memory descriptors`中，表示`destination will have the same memory format as the source`!

  * 示例代码如下

    ``` cpp
        // Tensor and kernel dimensions. We use the same 3x3 kernel with padding=1
        // for both convolution and pooling primitives, which means that the
        // activation tensor shapes do not change.
        const int N = 1, H = 14, W = 14, IC = 256, OC = IC, KH = 3, KW = 3;
        auto conv_src_md = memory::desc({N, IC, H, W}, memory::data_type::f32,
                memory::format_tag::any // let convolution choose memory format
        );
        auto conv_weights_md = memory::desc(
                {IC, OC, KH, KW}, memory::data_type::f32,
                memory::format_tag::any // let convolution choose memory format
        );
        auto conv_dst_md = conv_src_md; // shape does not change
        auto pool_dst_md = conv_dst_md; // shape does not change
    ```

  *  

* create source & destination memory objects

  * 假设输入、输出的`memory format`是`NCHW`

  * 示例代码如下

    ``` cpp
        auto src_mem = memory(
                {{N, IC, H, W}, memory::data_type::f32, memory::format_tag::nchw},
                eng);
        auto weights_mem = memory({{IC, OC, KH, KW}, memory::data_type::f32,
                                          memory::format_tag::oihw},
                eng);
        auto dst_mem = memory(
                {{N, IC, H, W}, memory::data_type::f32, memory::format_tag::nchw},
                eng);
    ```

  * 注意`weight`的`memory format`是`memory::format_tag::oihw`,与`memory descriptor`中的logical dim order一致

  *  

* Determine if source & destination need to be reordered

  * 通过`primitive`所期望拿到的`memory format`与可用的`memory format`之间比较对应的`memory descriptor`确定

  * 示例代码如下

    ``` cpp
    bool need_reforder_src = conv_pd.src_desc() != src_mem.get_desc();
    ```

  * 注意，这里不能通过`memory format tags`的比较来实现，因为这个只提供了部分的`data layout in memory`，并没有描述`memory objects obtained via sub-memory constructor`

  * 重复这个判断在`weights, destination memory format descriptors`上

    ``` cpp
        bool need_reorder_weights
                = conv_pd.weights_desc() != weights_mem.get_desc();
        bool need_reorder_dst = conv_pd.dst_desc() != dst_mem.get_desc();
    ```

  *  

* Allocate intermediate buffers if necessary

  * 根据上面的结果，这一步对`source, weights data, output of the pooling`等数据进行必要的空间分配

  * 临时空间的创建是基于`memory descripors from the primitive descriptors`,来保证一致性

  * 示例代码如下

    ``` cpp
        auto conv_src_mem
                = need_reorder_src ? memory(conv_pd.src_desc(), eng) : src_mem;
        auto conv_weights_mem = need_reorder_weights
                ? memory(conv_pd.weights_desc(), eng)
                : weights_mem;
        auto conv_dst_mem = memory(conv_pd.dst_desc(), eng);
        auto pool_dst_mem
                = need_reorder_dst ? memory(pool_pd.dst_desc(), eng) : dst_mem;
    ```

  *  

* Perform reorders for source data if necessary

  * 开始实际执行，也就是`reorder data`

  * 在调用`reorder primitives`之前，调用`dnnl::stream::wait()`，在异步执行中需要这样做

    ``` cpp
        if (need_reorder_src) {
            auto reorder_src = reorder(src_mem, conv_src_mem);
            reorder_src.execute(
                    s, {{DNNL_ARG_FROM, src_mem}, {DNNL_ARG_TO, conv_src_mem}});
            s.wait(); // wait for the reorder to complete
        }
        if (need_reorder_weights) {
            auto reorder_weights = reorder(weights_mem, conv_weights_mem);
            reorder_weights.execute(s,
                    {{DNNL_ARG_FROM, weights_mem},
                            {DNNL_ARG_TO, conv_weights_mem}});
            s.wait(); // wait for the reorder to complete
        }  // s is stream
    ```

  *  

* Create and execute convolution and pooling primitives

  * 在执行完`reorder`之后，就可以进行`compute convoulution & pooling`

  * 示例代码如下

    ``` cpp
        auto conv_scratchpad_mem = memory(conv_pd.scratchpad_desc(), eng);
        auto conv = convolution_forward(conv_pd);
        conv.execute(s,
                {{DNNL_ARG_SRC, conv_src_mem}, {DNNL_ARG_WEIGHTS, conv_weights_mem},
                        {DNNL_ARG_DST, conv_dst_mem}});
        auto pool_scratchpad_mem = memory(pool_pd.scratchpad_desc(), eng);
        auto pool = pooling_forward(pool_pd);
        pool.execute(
                s, {{DNNL_ARG_SRC, conv_dst_mem}, {DNNL_ARG_DST, pool_dst_mem}});
        s.wait();
    ```

  *  

* Reorder destination data if necessary

  * 与输入数据的`reorder`一致

  * 示例代码如下

    ``` cpp
        if (need_reorder_dst) {
            auto reorder_dst = reorder(pool_dst_mem, dst_mem);
            reorder_dst.execute(
                    s, {{DNNL_ARG_FROM, pool_dst_mem}, {DNNL_ARG_TO, dst_mem}});
            s.wait();
        }
    ```

  *  

*  

## Inference & Training Aspects

### propagation kinds

* Inference

  * `dnnl::prop_kind::forward_inference`
  *  

* Training

  通常由下面几个步骤构成。

  * make prediction based on the current state of the model，虽然也是前向过程，但对应的`propagation kind`为`dnnl::prop_kind::forward_trainingJ`，注意与上面名字的后缀不同
  * compute an error between predicted and the actual answer
  * perform the backward propagation of errors to compute the weights (learnable parameters) gradient，对于给定的operation，后向传播过程童颜个可以分为两个步骤
    * propagating error with respect to data, *i.e.* computing `diff_src` from `diff_dst`，这个步骤对应的`propagation kind`为`dnnl::prop_kind::backward_data`
    * propagating error with respect to weights, *i.e.* computing `diff_weights` from `diff_dst`，对应的`propagation kind`为`dnnl::prop_kind::backward_weights`
    * 也就是对应的分别对权重、输入数据进行求导两个步骤
    *  
  * use computed gradients to modify the weights according to the chosen solver to. Improve the accuracy of the model
  *  

* 后向传播过程分为了对`diff_src, diff_weights`两个过程

*  

### difference between forward propagation on training and inference

* 在推理过程中，一些临时变量可以被服用，而训练过程中，临时变量可能需要被用于计算输入数据的梯度
* 使用时，`dnnl::prop_lind::forward_training`可能输出不止一个变量

### Inference-Specific Aspects

下面列出了推理过程中主要的`specifics`

* 使用`dnnl::prop_lind::forward_inference`作为`propagation kind`
* 尽可能使用`in-place`属性
* 创建一次`primitives`，然后在不同的`model invocations`之间进行复用
* 使用`post-ops attributes`来将多个不同的`primitives`串联或者融合，可以减轻带宽压力，提高性能

### Training-Specific Aspects

* 注意前面提到的训练时`propagation kind`，涉及三个类别啊
* 不同`model invocations`对`primitives`进行复用
* 不同的计算在后向传播时可能需要不同的`tensors`，这个就跟具体函数的梯度计算公式又关系了
* 对于`dnnl::memory::format_tag::any`标记的`memory format tag`，不能保证前向、后向计算使用相同的`memory format`，此时就需要使用`reorders`了
* 尽可能使用in-place属性，而且要对应的使用`dnnl::memory::format_tag:;any`作为`gradient tensors`的`memory format`
* 有些`primitive`在前向、后向传播之间还需要辅助数据，被称为`Workspace`
* 在创建后向传播过程的`primitive descriptors`时，可能还需要传入前线的`primitive descriptor`，这被称为`hint`，只有那些需要`workspace`的`primitive`需要
* when creating your working memory and mkl-dnn memory descriptor, specify the type of memory you want to work with

### Workspace

> the workspace is a tensor that the primitives fills in during forward propagation and that will then be used by the corresponding backward propagation operation.

使用`workspace`的流程如下：

* 在创建前向传播中的`primitive`时，使用`.workspace_desc()`来获取`primitive descriptor`对`workspace`的需求

  * 如果返回的`memory descriptor`是空的，即`dnnl::memory::desc() or dnnl::memory::desc::get_size() return 0`，那么就不会进行额外的操作
  * 如果需要新的`workspace`，那么就将得到的`memory descriptor`传递到`execution function`，使用的`map tag`为`DNNL_ARG_WORKSPACE`
  *  

* 在后向传播计算中，attach that same workspace memory during the execution as well. 注意，the state of the workspace memory after backward computations are done is undefined.

* 示例代码如下

  ``` cpp
  // FWD
  auto forward_primitive_desc = ...::primitive_desc(); // create a primitive desc
  auto workspace_md = forward_primitive_desc.workspace_desc(); // query workspace
  memory workspace(workspace_md, engine); // create a memory (even if empty)
  primitive_forward.execute(stream, {
          ...,
          {DNNL_ARG_WORKSPACE, workspace} // this is output
          });
  // The workspace contains required information for the backward propagation,
  // hence should not be used anywhere else.
  // ...
  // BWD
  primitive_backward.execute(stream, {
          ...,
          {DNNL_ARG_WORKSPACE, workspace} // this input/output
          });
  // The state of the workspace is undefined here
  ```

* 注意，这里的`workspace`与`primitive attributes: scratchpad`不同，后者仅在`primitive`计算过程中被产生，不会被不同的调用之间共享使用，即使在推理过程中也可能需要的。

*  

## Primitive Attributes

* 在创建`primitive descriptor`的时候被用到，并且是被复制过去的，所以在创建完`primitive descriptor`之后就可以删除这些`attributes`

* 回顾下创建`primitive`的三个抽象层次对应的三个步骤

  * 初始化`operation descriptor`
  * 基于`operation descriptor, engine, attributes`创建`primitive descriptor`。注意，在创建后向计算的`primitive`时，这一步还需要传入对应前向计算的`primitive`创建时使用的`primitive descriptor`。这里的`primitive descriptor`一旦创建就不能被修改了
  * 仅使用`primitive descriptor`就可以创建`primitive`了
  *  

* `operation descriptor`包含的信息有：operation  kind, propagationo kind, source, destination, and other tensors, the strides and so on

* 设置`attributes`时，必须使用对应的`setter`接口。`attributes`可以是空，此时使用默认的`attribute`来创建`primitive descriptor`

* C++中的使用代码示例如下

  ``` cpp
  // ### C++ API ###
  dnnl::primitive_attr attr;
  attr.set_SOMETHING(params);
  attr.set_SOMETHING_ELSE(params);
  primitive::primitive_desc pd(..., attr);
  // in C++ destroying of attr happens automatically
  ```

* 支持的ATTRIBUTES

  * Scratchpad behavior

    handling the intermediate emporary memory by the library or a user

  * Quantization

    Used in INT8 inference

  * Post-ops

    fuse a primitive with some operation applied to the primitive's result, used mostly for inference

* 异常处理

  * 可能会创建一些`primitive descriptor`时传入一些还不支持的`attributes`，因为`attributes`毕竟得考虑扩展性，此时用户就会得到一个`dnnl_unimplemented`得错误，在C++中，是`dnnl::error`，但目前阶段，库并不会提供具体的信息！

## Data Types

### 支持的Data Types

![image-20191216213656395](/Users/smher/Library/Application Support/typora-user-images/image-20191216213656395.png)

### Inference & training

![image-20191216213738621](/Users/smher/Library/Application Support/typora-user-images/image-20191216213738621.png)

* 注意，使用低精度计算时，可能需要修改模型的实现
* 不同的`primitive`可能对精度有不同的要求，需要参考具体支持的数据类型

### Hardware Limitations

 不同的硬件也支持不同的精度，`DNN`的实现也不同，可参考下图：

![image-20191216214047013](/Users/smher/Library/Application Support/typora-user-images/image-20191216214047013.png)

* 如果硬件不支持`bf16`，一般需要`AVX512 Byte and Word Instructions (AVX512BW)`的硬件、指令支持，否则使用`bf16`比使用`fp32`会慢3-4倍！

## Reorder between CPU & GPU engines

主要是CPU与GPU之间可能需要的`memory reordering`。

* 包含`dnnl.hpp`头文件
* 所有的`API types and functions`都在`dnnl`明明空间内

### Cross_engine_reorder_tutorial() function

* Engine & stream

  * 示例代码如下

    ``` cpp
    auto cpu_engine = engine(engine::kind::cpu, 0);
    auto gpu_engine = engine(engine::kind::gpu, 0);
    auto stream_gpu = stream(gpu_engine);
    auto stream_cpu = stream(cpu_engine);
    ```

* Wrapping data into DNNL GPU memory object

  * 示例代码如下

    ``` cpp
        const auto tz = memory::dims {2, 16, 1, 1};
        auto m_cpu
                = memory({{tz}, memory::data_type::f32, memory::format_tag::nchw},
                        cpu_engine);
        auto m_gpu
                = memory({{tz}, memory::data_type::f32, memory::format_tag::nchw},
                        gpu_engine);
        fill(m_cpu, tz);
        auto r1 = reorder(m_cpu, m_gpu);
    ```

* Creating a ReLU primitive

  * 创建GPU版的ReLU示例代码如下，其它过程与CPU的例子中一致，尤其是步骤

    ``` cpp
        //  ReLU op descriptor (uses a GPU memory as source memory.
        //  no engine- or implementation-specific information)
        auto relu_d = eltwise_forward::desc(prop_kind::forward,
                algorithm::eltwise_relu, m_gpu.get_desc(), 0.0f);
        // ReLU primitive descriptor, which corresponds to a particular
        // implementation in the library. Specify engine type for the ReLU
        // primitive. Use a GPU engine here.
        auto relu_pd = eltwise_forward::primitive_desc(relu_d, gpu_engine);
        // ReLU primitive
        auto relu = eltwise_forward(relu_pd);
    ```

* reorder & get the results from a DNNL GPU memory object

  ``` cpp
  auto r2 = reorder(m_gpu, m_cpu);
  ```

* 执行所有的primitives

  * 计算流程是`Reorder (CPU, GPU) -> ReLU -> Reorder(GPU, CPU)`

  * 示例代码如下

    ``` cpp
        // wrap source data from CPU to GPU
        r1.execute(stream_gpu, m_cpu, m_gpu);
        // Execute ReLU on a GPU stream
        relu.execute(stream_gpu, {{DNNL_ARG_SRC, m_gpu}, {DNNL_ARG_DST, m_gpu}});
        // Get result data from GPU to CPU
        r2.execute(stream_gpu, m_gpu, m_cpu);
        stream_gpu.wait();
    ```

    * 注意，所有的`primitives`在相同的`GPU stream`上完成

*  

## API

* 提供了C/C++ 相关的 API
* 需要注意的地方：可能引起非定义行为的情况
  * NaN 作为输入
  * 超过`s8, u8`范围的数导致累加器`overflow`
  * 使用`bf16 (16-bit floating point)`精度会比`32-bit floating point`下降很多
  *  
* 使用指针作为参数时
  * 在传入`MKLDNN`之前需要已经分配好空间
  * 保证`data buffers`之间不能相互重叠，除非明确允许`in-place`计算

## Understanding Memory Formats

* data format
  * one form of data representation that describes how multidimensional arrays (nD) are stored in linear (1D) memory address space and why this is important for DNNL。
  * data format 与 `layout`是等价的在本文中

### Data formats

输入数据有4个维度，包括batch, channel, spatial dimension。为了方便，仅以2D spatial为例进行说明。

* Plain data formats

  * 以`batch = 2, channel = 16, spatial = 5 * 4`

    ![image-20191217195832544](/Users/smher/Library/Application Support/typora-user-images/image-20191217195832544.png)

    * 那么在`(n, c, h, w)`位置的数值是

      ```reStructuredText
      value(n, c, h, w) = n * CHW + c * HW + h * W + w
      ```

    *  

  * 定义一个logical index (n, c, h, w) 到`address displacement to the location of the value`的函数

    ```reStructuredText
    offset : (int, int, int, int) --> int
    ```

    * NCHW

      ```reStructuredText
      offset_nchw(n, c, h, w) = n * CHW + c * HW + h * W + w
      ```

      >  We use `nchw` here to denote that `w` is the inner-most dimension, meaning that two elements adjacent in memory would share the same indices of `n`, `c`, and `h`, and their index of `w` would be different by `1`. This is of course true only for non-border elements. On the contrary, `n` is the outermost dimension here, meaning that if you need to take the same pixel `(c, h, w)` but on the next image, you have to jump over the whole image size `C*H*W`.
      > One can create memory with **NCHW** data layout using [dnnl_nchw](https://intel.github.io/mkl-dnn/group__dnnl__api__memory.html#gga395e42b594683adb25ed2d842bb3091da83a751aedeb59613312339d0f8b90f54) of the enum type [dnnl_format_tag_t](https://intel.github.io/mkl-dnn/group__dnnl__api__memory.html#ga395e42b594683adb25ed2d842bb3091d) defined in [dnnl_types.h](https://github.com/intel/mkl-dnn/blob/master/include/dnnl_types.h) for the C API, and [dnnl::memory::format_tag::nchw](https://intel.github.io/mkl-dnn/structdnnl_1_1memory.html#a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faded7ac40158367123c5467281d44cbeb) defined in [dnnl.hpp](https://github.com/intel/mkl-dnn/blob/master/include/dnnl.hpp) for the C++ API.

    * NHWC

      ```reStructuredText
      offset_nhwc(n, c, h, w) = n * HWC + h * WC + w * C + c
      ```

      >  For a single image (**N** = 1), this format is very similar to how [BMP-file format](https://en.wikipedia.org/wiki/BMP_file_format) works, where the image is kept pixel by pixel and every pixel contains all required information about colors (for instance, three channels for 24bit BMP).

      layout 对应 `dnnl_nhwc or dnnl::memory::format_tag::nhwc`

    * CHWN

    * 不同layout之间的数据示意图如下

      ![image-20191217200657137](/Users/smher/Library/Application Support/typora-user-images/image-20191217200657137.png)

    *  

*  

### Generalization of the plain data layout

也就是存在相邻两个像素索引不相邻，或者存在`strides`的情况，

![image-20191217200936742](/Users/smher/Library/Application Support/typora-user-images/image-20191217200936742.png)

此时，有

```reStructuredText
offset(n, c, h, w) = n * stride_n
                   + c * stride_c
                   + h * stride_h
                   + w * stride_w
```

* 用户可以基于stride来创建`memory descriptors`，就如前面提到的两一种方式创建的`memory descriptor`

  ``` cpp
  dnnl_dims_t dims = {N, C, H, W};
  dnnl_dims_t strides = {stride_n, stride_c, stride_h, stride_w};
  dnnl_memory_desc_t md;
  dnnl_memory_desc_init_by_strides(&md, 4, dims, dnnl_f32, strides);
  ```

* DNNL支持strides via blocking structure，伪代码如下

  ``` cpp
  memory_desc_t md; // memory descriptor object
  // logical description, layout independent
  md.ndims = 4;           // # dimensions
  md.dims = {N, C, H, W}; // dimensions themselves
  // physical description
  md.format_kind = dnnl_blocked; // generic blocked format
  md.format_desc.blocking.strides = {
      stride_n, stride_c, stride_h, stride_w
  };
  ```

### Blocked layout

* 目的：为了更好的向量化，或者提高cache的复用，DNNL引入了Blocked layout，也就是将某一维或某些维度分成多个固定大小，比如`nChw16c on AVX512+, nChw8c on SSE4.1+`，也就是将`channel`维分成了16或8个block

* 对应的offset function如下，以`nChw8c`为例

  ``` cpp
  offset_nChw8c(n, c, h, w) = n * CHW
                            + (c / 8) * HW*8
                            + h * W*8
                            + w * 8
                            + (c % 8)
  ```

  ![image-20191217201626327](/Users/smher/Library/Application Support/typora-user-images/image-20191217201626327.png)

  * 也即是说，相邻的8个`channel`称为innerest维度了，在存储空间中相同block内的相邻channel的像素称为相邻的了，其次是spatial维度，在然后是`channel block`，最后是`batch`维度
  * We use lower- and uppercase letters in the formats to distinguish between the blocks (e.g. 8c) and the remaining co-dimension (**C** = channels / 8).
  * 背后的原理见论文：[Distributed Deep Learning Using Synchronous Stochastic Gradient Descent](https://arxiv.org/pdf/1602.06709v1.pdf)
  *  

* 对应的伪代码如下

  ``` cpp
  memory_desc_t md;
  // logical description, layout independent
  md.ndims = 4;           // # dimensions
  md.dims = {N, C, H, W}; // dimensions themselves
  // physical description
  md.memory_format = dnnl_blocked; // blocked layout
  ptrdiff_t stride_n = C*H*W;
  ptrdiff_t stride_C = H*W*8;
  ptrdiff_t stride_h =   W*8;
  ptrdiff_t stride_w =     8;
  md.format_desc.blocking.strides = { // strides between blocks
      stride_n, stride_C, stride_h, stride_w
  };
  md.format_desc.inner_nblks = 1; // number of blocked dimensions;
                                  // 1, since only channels are blocked
  md.format_desc.inner_idxs[0] = 1; // Only the 1st (c) dimension is blocked
                                    // n -- 0st dim, w -- 3rd dim
  md.format_desc.inner_blks[0] = 8; // This 1st dimensions is blocked by 8
  ```

* 但如果channel的个数不是16或8 的倍数怎么办

  * 一种方法是先把前面能整除的所有channel进行计算，然后将剩下不足8个或16个channel的`tail`进行计算，但tail情况是在太多了，对于DNNL不现实

  * DNNL采用的方法是zero-padding

    > The idea is to round the channels up to make them multiples of the block size and pad the resulting tail with zeros (in the example above, `24 = div_up(17, 8) * 8`). Then primitives like convolutions might work with a rounded-up number of channels instead of the original ones and compute the correct result (adding zeros doesn't change the result).

  * zero-padding处理示意图如下所示

    ![image-20191217202504654](/Users/smher/Library/Application Support/typora-user-images/image-20191217202504654.png)

  *  

* 处理不能整除时的一些陷阱

  * To keep *padded data are zeros* invariant, [dnnl_memory_set_data_handle()](https://intel.github.io/mkl-dnn/group__dnnl__api__memory.html#ga6888f8c17f272d6729c9bc258ed41fcf) and [dnnl::memory::set_data_handle()](https://intel.github.io/mkl-dnn/structdnnl_1_1memory.html#aaf3c13ed6f4af1b719ef7413ee36f63d) physically add zeros whenever the user attaches a pointer to a memory that uses zero padding. That might affect performance if too many unnecessary calls to these functions are made. We might consider extending our API in the future to allow attaching pointers without subsequent initialization with zeros if the user can guarantee that the padding is already filled correctly.

  * 因为补零会引入存储空间开销，The memory size required to keep the data cannot be computed by the formula `sizeof(data_type) * N * C * H * W` anymore. The actual size should always be queried via [dnnl_memory_desc_get_size()](https://intel.github.io/mkl-dnn/group__dnnl__api__memory.html#gaed039afa75d9f56763e2c1321f1563c4) in C and [dnnl::memory::desc::get_size()](https://intel.github.io/mkl-dnn/structdnnl_1_1memory_1_1desc.html#a3a12698f833b44ed55af9fc6621c4917) in C++.

  * Element-wise operations that are implemented in the user's code and directly operate on DNNL blocked layout like this:

    ``` cpp
    for (int e = 0; e < phys_size; ++e)
        x[e] = eltwise_op(x[e])
    ```

    are not safe if the data is padded with zeros and `eltwise_op(0) != 0`.

  *  

* 总的DNNL库的代码如下

  ``` cpp
  const int C = 17;
  const int C_padded = div_up(17, 8) * 8; // 24
  // logical description, layout independent
  const int ndims    = 4;            // # of dimensions
  dnnl_dims_t dims = {N, C, H, W}; // dimensions themselves
  memory_desc_t md;
  // initialize memory descriptor
  dnnl_memory_desc_init(&md, ndims,
                               dims,
                               dnnl_f32,   // single precision data type
                               dnnl_nChw8c // blocked layout
                               );
  ptrdiff_t expect_stride_n = C_padded*H*W;   // note C_padded here, not C
  ptrdiff_t expect_stride_C =          H*W*8;
  ptrdiff_t expect_stride_h =            W*8;
  ptrdiff_t expect_stride_w =              8;
  ptrdiff_t expect_stride_8c =             1;
  bool expect_true = true
      && true // logical dims stay as is
      && md.dims[0] == N
      && md.dims[1] == C
      && md.dims[2] == H
      && md.dims[3] == W
      && true // padded dims are rounded accordingly
      && md.padded_dims[0] == N
      && md.padded_dims[1] == C_padded
      && md.padded_dims[2] == H
      && md.padded_dims[3] == W
      && true // strides between blocks correspond to the physical layout
      && md.format_desc.blocking.strides[0] == expect_stride_n
      && md.format_desc.blocking.strides[1] == expect_stride_C
      && md.format_desc.blocking.strides[2] == expect_stride_h
      && md.format_desc.blocking.strides[3] == expect_stride_w
      && true // inner-most blocking
      && md.format_desc.blocking.inner_nblks == 1 // only 1 dim is blocked (c)
      && md.format_desc.blocking.inner_idxs[0] == 1  // 1st (c) dim is blocked
      && md.format_desc.blocking.inner_dims[0] == 8; // the block size is 8
  assert(expect_true);
  ```
