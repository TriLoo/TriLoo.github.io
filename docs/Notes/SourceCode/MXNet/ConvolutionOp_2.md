---
title: Convolution (1)
---
# Convolutioin Op 源码分析 (1)

smh

2019.12.27

## 简介

* 反向传播时需要对数据、权重、bias都计算梯度，这里只负责计算梯度，更新过程在调用`step()`中才发生
* 前向过程对应的计算是`z = w * x + b`，反向求导就是对这三个输入数据分别求偏导，对输入求导是为了反向传播。计算过程的输入还包括从输出数值的梯度，也就是这里的`out_grad`，形状与`z`的形状相同，`b`的梯度就是传入的输出值的梯度。
* 反向求导的过程，得基于前向过程进行考虑，在计算权重的梯度的时候，需要的是输入数据的 `inc / g * ks * ks`大小的数据；计算输入数据的梯度时，需要的是`inc / g * ks * ks`形状的权重数据。另一方面，对于前向输出数据的梯度需要的是：`out_grad * weight`；而对于权重的梯度的计算的是`out_grad * input`，这两个顺序是根据源码确定的。而且先计算`data`的梯度，然后计算`weight`的梯度。

## 反向传播计算

源码如下：

``` cpp
// ...
    index_t M = kernel_dim_;                            // inc / g * ks * ks
    index_t N = conv_out_spatial_dim_;                  // fm_h * fm_w
    index_t K = conv_out_channels_ / group_;            //
    // kernel 的数据尺寸是： (inc / g) * ks * ks
    // 其中最后一个维度，一列对应的就对应一个 feature map，然后就一共 group_ * K 个 feature map
    Tensor<xpu, 3, DType> weight_3d = in_data[conv::kWeight].get_with_shape<xpu, 3, DType>(
      Shape3(group_, K, M), s);
    Tensor<xpu, 4, DType> out_grad_4d = out_grad[conv::kOut].get_with_shape<xpu, 4, DType>(     // out_grad 的尺寸其实就是 z 的尺寸
      // num_: batch size
      Shape4(num_, group_, K, N), s);
    // For computing dLoss/dWeight
    Tensor<xpu, 3, DType> dweight_3d = in_grad[conv::kWeight].get_with_shape<xpu, 3, DType>(
      Shape3(group_, K, M), s);
//...
```

* 反向过程中，重新调用`LayerSetUp(...)`设置类成员变量

### 1x1 的情况

``` cpp
    if (is_1x1_) {
      Tensor<xpu, 4, DType> input_4d = in_data[conv::kData].get_with_shape<xpu, 4, DType>(
        Shape4(num_, group_, M, N), s);
      Tensor<xpu, 4, DType> in_grad_4d = in_grad[conv::kData].get_with_shape<xpu, 4, DType>(
        Shape4(num_, group_, M, N), s);
      // 对于 batch 中的每一幅图像分别计算
      for (index_t n = 0; n < num_; ++n) {
        Tensor<xpu, 3, DType> input_3d = input_4d[n];           // shape: (group_, M, N)
        Tensor<xpu, 3, DType> in_grad_3d = in_grad_4d[n];       // shape: (group_, M, N)
        Tensor<xpu, 3, DType> out_grad_3d = out_grad_4d[n];     // shape: (group_, K, N), K = out_c / group_
        // gradient w.r.t. input data
        for (index_t g = 0; g < group_; ++g) {
          //
          linalg_gemm(weight_3d[g], out_grad_3d[g], in_grad_3d[g], true, false, s);
          auto request = (n == 0) ? req[conv::kWeight] : kAddTo;
          linalg_gemm(out_grad_3d[g], input_3d[g], dweight_3d[g], false, true, s, request);
        }
      }
    }
```

* 首先计算对输入数据的梯度，结果保存在`in_grad_3d`，对应的是`in_grad[conv::kData]`；然后再计算当前卷积权重参数的梯度，结果保存在`dweight_3d`里面，对应的是`in_grad[conv::kWeight]`
* 在计算输入数据的梯度时，计算过程为：`weight.T() * out_grad -> in_data_grad`，也就是说，需要对权重数据进行一次转置。具体原因为，根据前向计算过程，`out_grad_3d[g]`的列维度对应的是输入数据的一列，而一行对应的是`weight`的一行，这句话的意思是，前向计算中，一个输出像素对应输入的权重的一行，对应输入数据的一列；或者说，输入数据的每个元素都要与输入权重的一列进行计算；另一方面，对于权重数据中的每一个元素，都需要参与一行的输入数据进行计算计算输出数据。前向过程如下图所示（TODO）。
* 基于上面的分析，计算输入数据的梯度的时候，因为权重数据在前面，输出数据的梯度在后面，所以需要对权重进行转置，这是因为输出数据梯度的一列对应的是权重数据的一列（前向计算时，权重数据在前面）；输出数据梯度的一行对应的是输入数据的一行。至此是计算输入数据的梯度了。
* 计算权重的梯度的过程与上述过程类似，也就是说，权重数据中的每一个元素参与输入数据的一行，同时输入数据是计算权重梯度时的第二项，所以需要对输入数据进行转置，这样第一项的输出数据的梯度乘以转置后的输入数据才是对应输入数据的一行。另一方面，权重数据中的每一个元素，参与的是输出数据的一行（因为前向时权重在前、输入数据在后，矩阵乘的结果中的一行就由权重的一行决定了），所以计算权重梯度时，输出数据梯度的一行用于计算。这里计算过程是：`out_grad * input.T()`。
* 至此，`1x1`的计算过程就分析完了

### 3x3的情况

``` cpp
    else {
      // allocate workspace for col_buffer
      Tensor<xpu, 1, DType> workspace = ctx.requested[conv::kTempSpace]
        // col_buffer_size_ = inc * ks * ks * fm_h * fm_w, fm -> feature map
        // 这个col_buffer_size_ 就是计算一个输出 feature map 所需的空间？为啥还要考虑 fm_h * fm_w。其实是用来保存输出数据的形状的
        .get_space_typed<xpu, 1, DType>(Shape1(col_buffer_size_), s);
      // calculate the shape of col_buffer
      mxnet::TShape col_buffer_shape(num_spatial_axes_ + 1, 1);           // num_spatial_axes， 也就是空间维的个数，通常为2
      col_buffer_shape[0] = conv_in_channels_ * param_.kernel.Size();     // inc * ks * ks
      for (int i = 1; i < col_buffer_shape.ndim(); ++i) {
        col_buffer_shape[i] = out_grad[conv::kData].shape_[i+1];
      }
      // create a column buffer using workspace and col_buffer_shape
      TBlob col_buffer(workspace.dptr_, col_buffer_shape, xpu::kDevMask, DataType<DType>::kFlag);
      Tensor<xpu, 3, DType> col_buffer_3d = col_buffer.get_with_shape<xpu, 3, DType>(
        Shape3(group_, M, N), s);
      for (index_t n = 0; n < num_; ++n) {
        Tensor<xpu, 3, DType> out_grad_3d = out_grad_4d[n];
        // gradient w.r.t. input data
        for (index_t g = 0; g < group_; ++g) {
          // Legacy approach shown here for comparison:
          //   col_buffer_3d[g] = dot(weight_3d[g].T(), out_grad_3d[g]);
          linalg_gemm(weight_3d[g], out_grad_3d[g], col_buffer_3d[g], true, false, s);
        }
        col2im(s, col_buffer.dptr<DType>(), in_grad[conv::kData].shape_, col_buffer.shape_,
               param_.kernel, param_.pad, param_.stride, param_.dilate,
               in_grad[conv::kData].dptr<DType>()+n*input_dim_, req[conv::kData]);

        // gradient w.r.t. weight, dWeight should accumulate across the batch and group
        im2col(s, in_data[conv::kData].dptr<DType>()+n*input_dim_, in_data[conv::kData].shape_,
               col_buffer.shape_, param_.kernel, param_.pad, param_.stride, param_.dilate,
               col_buffer.dptr<DType>());
        for (index_t g = 0; g < group_; ++g) {
          auto request = (n == 0) ? req[conv::kWeight] : kAddTo;
          // Legacy approach shown here for comparison:
          //   Assign(dweight_3d[g], request, dot(out_grad_3d[g], col_buffer_3d[g].T()));
          linalg_gemm(out_grad_3d[g], col_buffer_3d[g], dweight_3d[g], false, true, s, request);
        }
      }
    }
```

* 首先是对输入数据计算梯度，注意这个计算包括两个步骤！首先对输入数据计算梯度的过程与上面`1x1`的情况相同，然后在调用`col2im(...)`将`im2col(...)`过程中相同输入数据像素计算多个输出的过程但在这里需要合并的数值进行合并
* 所以这里可以看出来`col2im(...)`函数的作用就是将`im2col(...)`过程中相同输入像素的梯度结果合并，得到最终的像素的梯度，`col2im(...)`函数的实现见下文。
* 然后就是计算权重的梯度，也包含两个步骤。首先将输入数据调用`im2col(...)`，结果保存在临时空间`col_buffer`中；第二步是计算权重的梯度，也就是`out_grad * input.T()`了
* 计算权重梯度时，需要对输入数据首先调用`im2col(...)`，其实很好理解，就是要恢复前向计算时的数据结构，才能计算梯度，其他的就没了，所以这里对`im2col(...)`的调用与上面前向计算过程中的`im2col(...)`完全一致的
* 至此，就只剩下`col2im(...)`函数的具体实现了，注意这个函数的主要作用就是合并输入数据在`im2col(...)`结果中相同像素的梯度值！这一点可以从参数上看出来，`col2im(...)`函数输入数据是临时空间`col_buffer`，而函数的结果保存在`in_grad[conv::kData].dptr<DType>() + n * input_dim_`中了。具体就看下面的源码分析吧。

### Bias的梯度计算

上面，提到计算`bias`的梯度时，就是对输出数据梯度进行按照`channel`进行求和即可

``` cpp
    // gradient w.r.t bias
    if (bias_term_) {
      Tensor<xpu, 1, DType> dbias = in_grad[conv::kBias].get<xpu, 1, DType>(s);
      Tensor<xpu, 3, DType> dout = out_grad[conv::kOut].get_with_shape<xpu, 3, DType>(
          Shape3(num_, conv_out_channels_, conv_out_spatial_dim_), s);
      ASSIGN_DISPATCH(dbias, req[conv::kBias], sumall_except_dim<1>(dout));
    }
```

* 其实就是一个求和的过程

### col2im() 函数 (CPU)

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
```

* 首先调用`std::fill(...)`将`in_grad[conv::kData]`内容全部初始化为0，如果`req != kAddTo`
* 然后计算输出结果的形状，这里计算得到的形状其实就是传入的`col_buf_`的尺寸，也是这个卷积计算输出的空间尺寸了
* 最后就是主体，主体是一个五层循环，与`im2col(...)`过程相同；其中外面3层循环对应`data_col`的一列，里面2层循环对应`data_col`的一行
* `input_row`对应也是当前输出行所对应的输入数据的第`input_row`行，`input_col`的含义类似，加上最外层的`channel`参数，这三个参数就可以确定一个输入数据的像素位置了。
* 与`imcol(...)`的主要区别在于第`29`行，即调换了等号两边的两个计算数，同时运算符变为`+=`
* 需要注意的是，在与第`24`行，如果当前数据对应的输入数据的行pad部分，那么直接将 `data_col`的指针直接向前移动`output_w`的大小，即考虑下一个输出行；另一方面，即如果是处于列pad部分，则直接将`data_col++`即可。
* 至此，就分析完`col2im()`CPU端代码了
