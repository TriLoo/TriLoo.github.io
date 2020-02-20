---
title: Convolution (0)
---

# Convolution Op 源码分析 (0)

smh

2019.12.21

## 简介

* 为了以一个具体的 `Op` 来将整个流程串起来，包括前向、后向计算整个流程，中间可能涉及一些`Engine`、`Graph`的一些东西

* 整个Convolution Op的实现细节共有5篇笔记，这是第一篇，也是最早的一篇，内容个比较混乱...

### 计算顺序

* 通过查看下面对cuBLAS,的具体使用，发现卷积的计算是；

  `data * weight`!!!

* 但是观察`MKLDNN`，计算顺序是：

  `weight * data !!!`???????

* **从代码上来看，数据结构是后一种！！！**

*  

**需要确定 MKL Blas, cuBlas 下 Conv 计算结果是否一致！！！**

* 需要注意的是FullyConnected的情况，那里的计算顺序是

`data * weight.T()`!!!!

## 源码

### 前向计算

``` cpp
  void Forward(const OpContext &ctx,
               const std::vector<TBlob> &in_data,
               const std::vector<OpReqType> &req,
               const std::vector<TBlob> &out_data) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(req[conv::kOut], kWriteTo);
    size_t expected = param_.no_bias ? 2 : 3;
    CHECK_EQ(in_data.size(), expected);
    CHECK_EQ(out_data.size(), 1U);
    CHECK_EQ(req[conv::kOut], kWriteTo);
    LayerSetUp(in_data[conv::kData].shape_, out_data[conv::kOut].shape_);
    Stream<xpu>* s = ctx.get_stream<xpu>();

    // initialize weight and col_buffer 3D tensors for using gemm
    index_t M = conv_out_channels_ / group_;
    index_t N = conv_out_spatial_dim_;
    index_t K = kernel_dim_;
    Tensor<xpu, 3, DType> weight_3d = in_data[conv::kWeight].get_with_shape<xpu, 3, DType>(
      Shape3(group_, M, K), s);
    Tensor<xpu, 4, DType> output_4d = out_data[conv::kOut].get_with_shape<xpu, 4, DType>(
      Shape4(num_, group_, M, N), s);

    // no need to allocating memory and reordering in memory
    if (is_1x1_) {
      Tensor<xpu, 4, DType> input_4d = in_data[conv::kData].get_with_shape<xpu, 4, DType>(
        Shape4(num_, group_, K, N), s);
      for (index_t n = 0; n < num_; ++n) {
        Tensor<xpu, 3, DType> input_3d = input_4d[n];
        Tensor<xpu, 3, DType> output_3d = output_4d[n];
        for (index_t g = 0; g < group_; ++g) {
          linalg_gemm(weight_3d[g], input_3d[g], output_3d[g], false, false, s, req[conv::kOut]);     // CPU 这边使用 dot()函数完成计算 (不使用 MKL 时)
        }
      }
    } else {
      // allocate workspace for col_buffer
      Tensor<xpu, 1, DType> workspace = ctx.requested[conv::kTempSpace]
        .get_space_typed<xpu, 1, DType>(Shape1(col_buffer_size_), s);
      // calculate the shape of col_buffer
      mxnet::TShape col_buffer_shape(num_spatial_axes_ + 1, 1);
      col_buffer_shape[0] = conv_in_channels_ * param_.kernel.Size();
      for (int i = 1; i < col_buffer_shape.ndim(); ++i) {
        col_buffer_shape[i] = out_data[0].shape_[i+1];
      }
      // create a column buffer using workspace and col_buffer_shape
      TBlob col_buffer(workspace.dptr_, col_buffer_shape, xpu::kDevMask, DataType<DType>::kFlag);
      Tensor<xpu, 3, DType> col_buffer_3d = col_buffer.get_with_shape<xpu, 3, DType>(
        Shape3(group_, K, N), s);
      for (index_t n = 0; n < num_; ++n) {
        // transform image to col_buffer in order to use gemm
        im2col(s, in_data[conv::kData].dptr<DType>()+n*input_dim_, in_data[conv::kData].shape_,
               col_buffer.shape_, param_.kernel, param_.pad, param_.stride, param_.dilate,
               col_buffer.dptr<DType>());
        Tensor<xpu, 3, DType> output_3d = output_4d[n];
        for (index_t g = 0; g < group_; ++g) {
          // Legacy approach shown here for comparison:
          //   Assign(output_3d[g], req[conv::kOut], dot(weight_3d[g], col_buffer_3d[g]));
          linalg_gemm(weight_3d[g], col_buffer_3d[g], output_3d[g], false, false, s,
            req[conv::kOut]);
        }
      }
    }

    if (bias_term_) {
      Tensor<xpu, 1, DType> bias = in_data[conv::kBias].get<xpu, 1, DType>(s);
      Tensor<xpu, 3, DType> output_3d = out_data[conv::kOut].get_with_shape<xpu, 3, DType>(
        Shape3(num_, conv_out_channels_, conv_out_spatial_dim_), s);
      // has bias term, broadcast it to the same shape of output_3d in channel dim
      output_3d += mshadow::expr::broadcast<1>(bias, output_3d.shape_);
    }
  }
```

* 在`is_1x1`的情况下（即`kenrnel[i] == 1 && stride[i] == 1 && pad[i] == 0 for i in kernel.dim()`）
  * 直接调用`linalg_gemm(...)`函数进行计算
  * 注意计算过程是`weight * data`
  *  
* 其他情况下，见后面笔记中的分析

### linalg_gemm()函数

``` cpp
template<typename xpu, typename DType>
inline void linalg_gemm(const Tensor<xpu, 2, DType>& A,
                        const Tensor<xpu, 2, DType>& B,
                        const Tensor<xpu, 2, DType>& C,
                        bool tA, bool tB, Stream<xpu> *s,
                        mxnet::OpReqType req) {
  using namespace mxnet;
  switch (req) {
    case kNullOp:
      break;
    case kWriteTo:
    case kWriteInplace:
      linalg_gemm(A, B, C, DType(1.0), DType(0.0), tA, tB, s);
      break;
    case kAddTo:
      linalg_gemm(A, B, C, DType(1.0), DType(1.0), tA, tB, s);
      break;
    default:
      LOG(FATAL) << "not reached";
  }
}
```

* 第13、16行的`linalg_gemm(...)`函数是一个模板函数，根据`xpu`特例化了两个函数，而且比较有趣的是，这里的先定义一个宏，然后在使用这个宏定义出两个函数：

  ``` cpp
  #define LINALG_CPU_GEMM(fname, DType) \
  template<> inline \
  void linalg_gemm<cpu, DType>(const Tensor<cpu, 2, DType>& A, const Tensor<cpu, 2, DType>& B, \
                               const Tensor<cpu, 2, DType>& C, DType alpha, DType beta, \
                               bool tA, bool tB, Stream<cpu> *s) { \
    check_gemm(A, B, C, alpha, beta, tA, tB); \
    cblas_##fname(CblasRowMajor, (tA ? CblasTrans : CblasNoTrans), (tB ? CblasTrans : CblasNoTrans), \
                  C.size(0), C.size(1), (tA ? A.size(0) : A.size(1)), alpha, \
                  A.dptr_, A.stride_, B.dptr_, B.stride_, beta, C.dptr_, C.stride_); \
  }
  // ...
  LINALG_CPU_GEMM(sgemm, float)
  LINALG_CPU_GEMM(dgemm, double)
  ```

  * 注意，这里都是重载了上面的`linalg_gemm()`模板函数，且进行了特例化，函数模板不存在偏特例化！
  * 从这里也可以看出来，最终调用的是`cblas_sgemm, cblas_dgemm`两个`MKLDNN`里面的函数了。如果不使用`mkl blass`，那么具体实现就是使用一个`dot(...)`函数来实现了，这里先忽略`dot(...)`函数的实现
  * 这里需要注意的是，在CPU端，这里使用的是`MKL Blas`完成的计算，而还有一个是`ExCPU...`的那个函数，那里是使用`MKLDNN`实现的，而且那里的优先级高于这里的优先级。 

*  

#### cblas_sgemm (CPU)

 `MKL Blas`说明文档[mklman-11.2 - MKL](https://software.intel.com/sites/default/files/managed/9d/c8/mklman.pdf)里面解释如下：

* 包含的头文件`mkl.h`

* 函数类型

  ```c
  void cblas_sgemm (const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE transa, const
  CBLAS_TRANSPOSE transb, const MKL_INT m, const MKL_INT n, const MKL_INT k, const float
  alpha, const float *a, const MKL_INT lda, const float *b, const MKL_INT ldb, const
  float beta, float *c, const MKL_INT ldc);
  // cblas_dgemm
  // cblas_cgemm
  // cblas_zgemm
  ```

  * 具体参数的说明见文档吧！有点多。
  * API的风格与`NVIDIA cuBLAS`的风格很相近啊。。。
  *  

* 具体的操作

  $$C := \alpha * op(A) * op(B) + \beta * C$$

  * `op(X)`可以是$X, X^T, X^H$等，即转置
  *  

* 在MXNet中的实现与调用过程如下

  ``` cpp
  cblas_##fname(CblasRowMajor, (tA ? CblasTrans : CblasNoTrans), (tB ? CblasTrans : CblasNoTrans), \
                  C.size(0), C.size(1), (tA ? A.size(0) : A.size(1)), alpha, \
                  A.dptr_, A.stride_, B.dptr_, B.stride_, beta, C.dptr_, C.stride_); \

  // 调用， A weight, B, input, C output
          linalg_gemm(A, B, C, DType(1.0), DType(0.0), tA, tB, s);
  ```

  * `fname`可以是`sgemm, dgemm`等
  * 至此，是CPU端 `1x1` 情况下卷积的计算过程

*  

#### cublas_sgemm (GPU)

当`__CUDACC__`为真，且不使用`cuDNN`时，则使用这里的`cuBLAS`库里面的函数完成计算。具体代码我就不贴了，与CPU情况类似，也是先定义一个宏，然后在定义两个特例化函数分别调用`cublasSgemm, cublasDgemm`。注意，当CUDA_VERSION >= 7050 时，则使用`cublasSgemmEx(...)`代替`cublasSgemm`完成计算！

`cublas?gemm`的实现可以参考[CUBLAS_Library - 10.1 - CUDA](https://docs.nvidia.com/cuda/pdf/CUBLAS_Library.pdf)

细节这里就省略了吧。

### cuDNN Convolution

具体实现在`src/operator/nn.cudnn/cudnn_convolution-inl.h`，会在`convolution.cu::ConvoulutionCompute<gpu>`这个特例化函数里面使用并完成计算。

这个函数的注册是：

``` cpp
NNVM_REGISTER_OP(Convolution)
.set_attr<FCompute>("FCompute<gpu>", ConvolutionCompute<gpu>);
```

### 1x1 情况的计算

* 不需要调用`im2col()`，直接对数据进行计算

* 这里是分Batch内各个图像以及 Group 来计算的。

* 计算过程就是，将空间维压缩成1维，这样输出的空间维也不变了关键是下面公式：

  $(M, N) = (M, K) * (K, N)$

  * 其中`M`为`output_channel / group`
  * `K`为`input_channel / group * ks * ks`，这里`ks = 1`
  * `N`为`fm_h * fm_h`，这里也就是`input_h * input_w`了
  *  

* 配图后面再说吧，有点麻烦啊

* 接下来就是非`1x1`的情况了

*  

### im2col()函数

主要的前提，卷积的计算过程是这样进行的：

* 分图像、分组 进行卷积！！！

从上面`Convolution`计算过程来看，这一步的主要作用是：将数据换成`[(in_c / g * ks * ks), (fm_h * fm_w)]`，关键是第一项的数值那种结构！也就是一个两维的数据，并且输出feature map 的空间维是最后一维。

* 从实际使用`linalg_gemm(...)`进行计算时，`im2col(...)`的作用确实如上一句所述

  ``` cpp
  linalg_gemm(weight_3d[g], col_buffer_3d[g], output_3d[g], false, false, ...);
  ```

  * 其中，`col_buffer_3d[g].shape`为`[K, N]`，其中`K = inc/g * ks * ks`，也就是分组卷积中计算一个输出像素所需的参数或输入数据的数量
  *  

* 所以，`im2col(...)`的主要作用就是将输入的**每幅**图像变为`(g, inc/g * ks * ks, x)`的尺寸，考虑到具体计算还是，所以这里`x`的具体数值就是`in_h / ks* in_w / ks`，但实际上却是`out_h * out_w`，这就涉及`im2col(...)`的具体实现了，应该是前者，我猜在`im2col(...)`里面肯定有些输入的像素进行重新安排调整，**有些元素是重复的或被舍弃**。

*  

* 现在就看`im2col(...)`源码了

*  

然后就看下代码是不是这个意思吧！！！

* 调用

  ``` cpp
     im2col(s, in_data[conv::kData].dptr<DType>()+n*input_dim_, in_data[conv::kData].shape_,
                 col_buffer.shape_, param_.kernel, param_.pad, param_.stride, param_.dilate,
                 col_buffer.dptr<DType>());
  ```

* 实现（CPU）

  ``` cpp
  template <typename DType>
  inline void im2col(mshadow::Stream<cpu>* s,
                     const DType* data_im, const mxnet::TShape& im_shape,
                     const mxnet::TShape& col_shape, const mxnet::TShape& kernel_shape,
                     const mxnet::TShape& pad, const mxnet::TShape& stride,
                     const mxnet::TShape& dilation, DType* data_col) {
    if (2 == kernel_shape.ndim()) {
      im2col_cpu(data_im, im_shape[1], im_shape[2], im_shape[3],
                 kernel_shape[0], kernel_shape[1], pad[0], pad[1],
                 stride[0], stride[1], dilation[0], dilation[1], data_col);
    } else {
      im2col_nd_core_cpu(data_im, true, im_shape, col_shape,
                         kernel_shape, pad, stride, dilation, data_col);
    }
    template <typename DType>
  inline void im2col_cpu(const DType* data_im, const int channels,
      const int height, const int width, const int kernel_h, const int kernel_w,
      const int pad_h, const int pad_w,
      const int stride_h, const int stride_w,
      const int dilation_h, const int dilation_w,
      DType* data_col) {
    const int output_h = (height + 2 * pad_h -
      (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int output_w = (width + 2 * pad_w -
      (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    const int channel_size = height * width;
  
    for (int channel = channels; channel--; data_im += channel_size) {
      for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
        for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
          int input_row = -pad_h + kernel_row * dilation_h;
          for (int output_rows = output_h; output_rows; output_rows--) {
            if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
              for (int output_cols = output_w; output_cols; output_cols--) {
                *(data_col++) = 0;
              }
            } else {
              int input_col = -pad_w + kernel_col * dilation_w;
              for (int output_col = output_w; output_col; output_col--) {
                if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                  *(data_col++) = data_im[input_row * width + input_col];
                } else {
                  *(data_col++) = 0;
                }
                input_col += stride_w;
              }
            }
            input_row += stride_h;
          }
        }
      }
    }
  }
  ```

  * `CPU`的实现在`src/operator/nn/im2col.h`，`GPU`的实现在`src/operator/nn.im2col.cuh`
  * 这里就以`CPU`为例进行说明，`GPU`的实现后面补充
  * 首先需要考虑输入的`kernel`的形状是`M, K`，其中，`K`维度是按照`spatial -> channel` 这个顺序存储的，所以对输入数据进行的处理也是：先按照空间维，然后在`channel`维进行处理。具体体现在，将计算一个输出像素所需要的输入数据保存成一列
  * 在结果数据中，结果变量里面，相邻的两个地址空间（或者用相邻的两列）对应的是输入的相邻两个像素为中心的卷积窗口（三维），对应的而且还是输出的两个相邻的像素
  * 在实际操作中，先进行`output channel`维，也就是`inc * ks * ks`，然后按照空间维`Row-major`的方式遍历。注意，这里在`channel`维并没有分组的概念，因为`channel`是最外层，所以只需要在形状解释的时候调整即可
  * 可以看出来，现在`pad`的数值只支持常数0的pad
  * 代码主体是一个5层循环，外面3层对应的是结果数据的一列；里面两层对应的是结果数据的一行。`input_row`就是结果的每一行的起始数据对应输入数据的哪一行，也就是对应输入的第`input_row`行！而对应的行的参数就是变量`input_col`的数值；而对应输入数据的`channel`参数，则是在最外层循环中计算，也就是说`data_im`就是输入数据当前`channel`的起始指针。至此，基于以上三个参数，就可以计算正确的输入数据了！！！
  * 需要注意的是，第`31`行的`input_row`只是当前输出行所对应的输入数据的起始行，这个数据会在里面一层的循环中进行更新，完成所有输入行的遍历！`input_col`类似。想不明白，画个示意图就行！
  * 更多的解释，见源码中的解释！至此就结束了卷积前向计算过程CPU端
  *  

* 实现 （GPU）

  ``` cpp
  
  ```

### col2im()函数

源码如下：

``` cpp
template <typename DType>
inline void col2im_cpu(const DType* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    DType* data_im, OpReqType req) {
  if (mxnet::kNullOp == req) return;
  if (mxnet::kAddTo != req) {
    std::fill(data_im, data_im+height*width*channels, static_cast<DType>(0));
  }
  const int output_h = (height + 2 * pad_h -
    (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w = (width + 2 * pad_w -
    (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const int channel_size = height * width;

  for (int channel = channels; channel--; data_im += channel_size) {
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int input_row = -pad_h + kernel_row * dilation_h;
        for (int output_rows = output_h; output_rows; output_rows--) {
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
            data_col += output_w;
          } else {
            int input_col = -pad_w + kernel_col * dilation_w;
            for (int output_col = output_w; output_col; output_col--) {
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                data_im[input_row * width + input_col] += *data_col;
              }
              data_col++;
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
}

/*!\brief
 * cpu function of col2im algorithm
 * \param s device stream
 * \param data_col start pointer of the column buffer to be filled
 * \param im_shape input image shape in dimensions (N, C, H, W,)
 * \param col_shape column buffer shape
 * \param kernel_shape kernel filter shape
 * \param pad pad shape
 * \param stride stride shape
 * \param dilation dilation shape
 * \param data_im pointer of a image (C, H, W,...) in the image batch
 */
template <typename DType>
inline void col2im(mshadow::Stream<cpu>* s,
                   const DType* data_col, const mxnet::TShape& im_shape,
                   const mxnet::TShape& col_shape, const mxnet::TShape& kernel_shape,
                   const mxnet::TShape& pad, const mxnet::TShape& stride,
                   const mxnet::TShape& dilation, DType* data_im, OpReqType req) {
  int num_spatial_axes = kernel_shape.ndim();
  if (2 == num_spatial_axes) {
    col2im_cpu(data_col, im_shape[1], im_shape[2], im_shape[3],
               kernel_shape[0], kernel_shape[1], pad[0], pad[1],
               stride[0], stride[1], dilation[0], dilation[1], data_im, req);
  } else {
    im2col_nd_core_cpu(data_col, false, im_shape, col_shape,
                       kernel_shape, pad, stride, dilation, data_im, req);
  }
}
```

* CPU端，过程就是上面`im2col()`函数的反过程

## Op Register & Call 过程

用于前向计算注册时的模板函数如下。

``` cpp
template<typename xpu>
void ConvolutionCompute(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx, const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
  const ConvolutionParam& param = nnvm::get<ConvolutionParam>(attrs.parsed);
  MSHADOW_REAL_TYPE_SWITCH(inputs[conv::kData].type_flag_, DType, {
    ConvolutionOp<xpu, DType> op;
    op.Init(param);
    op.Forward(ctx, inputs, req, outputs);
  });
}
```

* 如果是`CPU`则直接使用这个模板注册；如果是`gpu`，则使用这个函数的`gpu`特例化版本，那里会调用`cuDNN`或`cuBLAS`等完成计算，具体过程见上面代码分析部分。

* 在注册过程中，还注册了`FComputeEx<cpu>`版本的计算，这里对应的函数是`ConvolutionComputeExCPU()`，这个函数里面使用MKLDNN里面的`MKLDNNConvolutionFoward(...)`来完成具体计算，而在这个函数里面，经过一系列的操作，就可以发现是最终是`shared_ptr<mkldnn::convolution_forward> fwd_`完成了计算，而这个函数在MKLDNN中的说明在[convolution_forward structure - mkldnn](https://intel.github.io/mkl-dnn/structdnnl_1_1convolution__forward.html)

* 从这个文件`src/imperative/imperative.cc`可以看出来，`FComputeEx<cpu>`要比`FCompute<cpu>`的优先级要高！！！

  ``` cpp
    FCompute fn = common::GetFCompute<FCompute>(op, "FCompute", ctx);
    FComputeEx fn_ex = common::GetFCompute<FComputeEx>(op, "FComputeEx", ctx);
  
    // FComputeEx is dispatched only when dispatch_mode is DispatchMode::kFComputeEx
    CHECK(dispatch_mode != DispatchMode::kUndefined);
    bool dispatch_fcompex = dispatch_mode == DispatchMode::kFComputeEx;
    if (fn_ex && dispatch_fcompex) {
      PushFComputeEx(fn_ex, op, attrs, ctx, read_vars, write_vars,
          requested, inputs, outputs, req);
    } else if (fn) {
      PushFCompute(fn, op, attrs, ctx, read_vars, write_vars,
          requested, inputs, outputs, mutate_idx, req);
      // ...
  ```

*  

### Register (CPU)

``` cpp
NNVM_REGISTER_OP(Convolution)
.describe(
  //...
.set_num_inputs([](const NodeAttrs& attrs) {
  const ConvolutionParam& params = nnvm::get<ConvolutionParam>(attrs.parsed);
  return params.no_bias ? 2 : 3;
})
.set_num_outputs(1)
.set_attr_parser(ConvolutionParamParser)
.set_attr<nnvm::FListInputNames>("FListInputNames",
    [](const NodeAttrs& attrs) {
  const ConvolutionParam& params = nnvm::get<ConvolutionParam>(attrs.parsed);
  if (params.no_bias)
    return std::vector<std::string>{"data", "weight"};
  else
    return std::vector<std::string>{"data", "weight", "bias"};
})
.set_attr<nnvm::FListOutputNames>("FListOutputNames",
    [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"output"};
})
.set_attr<mxnet::FInferShape>("FInferShape", ConvolutionShape)
.set_attr<nnvm::FInferType>("FInferType", ConvolutionType)
#if MXNET_USE_MKLDNN == 1
.set_attr<FInferStorageType>("FInferStorageType", ConvStorageType)
#endif
.set_attr<FCompute>("FCompute<cpu>", ConvolutionCompute<cpu>)
#if MXNET_USE_MKLDNN == 1
.set_attr<bool>("TIsMKLDNN", true)
.set_attr<FComputeEx>("FComputeEx<cpu>", ConvolutionComputeExCPU)
#endif
.set_attr<nnvm::FGradient>("FGradient", ConvolutionGrad{"_backward_Convolution"})
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.add_argument("data", "NDArray-or-Symbol", "Input data to the ConvolutionOp.")
.add_argument("weight", "NDArray-or-Symbol", "Weight matrix.")
.add_argument("bias", "NDArray-or-Symbol", "Bias parameter.")
.add_arguments(ConvolutionParam::__FIELDS__());
```

* 这里注册了CPU端完成卷积计算的函数注册
  * 使用`set_attr<FCompute>("FCompute<cpu>", ConvolutionCompute<cpu>)`注册普通的卷积计算函数，使用`mkl blas or dot()`完成计算
  * 使用`set_attr<FComputeEx>("FComputeEx<cpu>", ConvolutonComputeExCPU)`注册使用`MKLDNN`实现的卷积计算函数

### Register (GPU)

``` cpp
NNVM_REGISTER_OP(Convolution)
.set_attr<FCompute>("FCompute<gpu>", ConvolutionCompute<gpu>);
```

* 实现`ConvolutionCompute()`函数的`GPU`特例化版本，实现在GPU上的卷积计算
* 这个函数里面要么会调用`ConvolutionOp`这个类的`Forward()`函数完成前向计算，要么就是调用`cuDNN`的库函数完成计算。在前者里面，最终是会调用`cuBLAS::cublasSgemm()`等函数完成计算

### Call

## Caffe Convolution Implementation

待补充...
