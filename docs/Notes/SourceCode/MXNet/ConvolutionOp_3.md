---
title: Convolution (2)
---

# ConvolutionOp源码分析 (2)

smh

2019.12.27

## 简介

* 这个文件主要是用于讨论`GPU`端`im2col, col2im`的实现，这两个函数是重载的函数模板，也就是只有第一个参数的类型不同。
* 看GPU源码，一方面是算法的并行实现，另一方面是映射到`GPU`时的线程分配

## im2col() GPU

### 调用

``` cpp
template <typename DType>
inline void im2col(mshadow::Stream<gpu>* s,
                   const DType* data_im, const mxnet::TShape& im_shape,
                   const mxnet::TShape& col_shape, const mxnet::TShape& kernel_shape,
                   const mxnet::TShape& pad, const mxnet::TShape& stride,
                   const mxnet::TShape& dilation, DType* data_col) {
  // num_axes should be smaller than block size
  index_t num_spatial_axes = kernel_shape.ndim();
  CHECK_LT(num_spatial_axes, mshadow::cuda::kBaseThreadNum);
  index_t num_kernels = im_shape[1] * col_shape.ProdShape(1, col_shape.ndim());
  using namespace mxnet_op;
  switch (num_spatial_axes) {
  case 1:
    im2col_nd_gpu_kernel<DType, 1>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<cuda_get_num_blocks(num_kernels), mshadow::cuda::kBaseThreadNum,
           0, mshadow::Stream<gpu>::GetStream(s)>>>(
        num_kernels, data_im, im_shape.get<3>(), col_shape.get<2>(),
        kernel_shape.get<1>(), pad.get<1>(), stride.get<1>(), dilation.get<1>(), data_col);
    break;
  case 2:
    im2col_gpu_kernel<DType> // NOLINT_NEXT_LINE(whitespace/operators)
        <<<cuda_get_num_blocks(num_kernels), mshadow::cuda::kBaseThreadNum,
           0, mshadow::Stream<gpu>::GetStream(s)>>>(
        num_kernels, data_im, im_shape[2], im_shape[3], kernel_shape[0], kernel_shape[1],
        pad[0], pad[1], stride[0], stride[1], dilation[0], dilation[1],
        col_shape[1], col_shape[2], data_col);
    break;
  case 3:
    im2col_nd_gpu_kernel<DType, 3>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<cuda_get_num_blocks(num_kernels), mshadow::cuda::kBaseThreadNum,
           0, mshadow::Stream<gpu>::GetStream(s)>>>(
        num_kernels, data_im, im_shape.get<5>(), col_shape.get<4>(),
        kernel_shape.get<3>(), pad.get<3>(), stride.get<3>(), dilation.get<3>(), data_col);
    break;
  default:
    LOG(FATAL) << "im2col_nd_gpu does not support computation with "
               << num_spatial_axes << " spatial axes";
  }
  MSHADOW_CUDA_POST_KERNEL_CHECK(im2col_nd_gpu_kernel);
}
```

* 这里仅以2D卷积为例进行说明，这里直接调用 Kernel 函数进行计算了。
* 其他的情况，都依赖于`im2col_nd_gpu_kernel(...)`这个 Kernel函数
* `cuda_get_num_blocks(...)`的实现如下

  ``` cpp
  inline int cuda_get_num_blocks(const int N) {
    using namespace mshadow::cuda;
    return std::min(kMaxGridNum, (N + kBaseThreadNum - 1) / kBaseThreadNum);
  }
  ```

  * 这里用于计算`numBlock`这个参数
  *  

* 这里传入`cuda_get_num_blocks(...)`的参数是`num_kernels`，对应的数值是`inc * oh * ow`，因为目标尺寸是`inc * ks * ks * oh * ow`，所以每个线程计算`ks * ks`目标值

### im2col_gpu()函数实现

``` cpp
template <typename DType>
inline void im2col_gpu(mshadow::Stream<gpu>* s,
                       const DType* data_im, const int channels,
                       const int height, const int width,
                       const int kernel_h, const int kernel_w,
                       const int pad_h, const int pad_w,
                       const int stride_h, const int stride_w,
                       const int dilation_h, const int dilation_w,
                       DType* data_col) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int height_col = (height + 2 * pad_h -
      (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  int width_col = (width + 2 * pad_w -
      (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  int num_kernels = channels * height_col * width_col;
  using namespace mxnet_op;
  // NOLINT_NEXT_LINE(whitespace/operators)
  im2col_gpu_kernel<DType><<<cuda_get_num_blocks(num_kernels), mshadow::cuda::kBaseThreadNum,
                             0, mshadow::Stream<gpu>::GetStream(s)>>>(
      num_kernels, data_im, height, width, kernel_h, kernel_w, pad_h,
      pad_w, stride_h, stride_w, dilation_h, dilation_w, height_col,
      width_col, data_col);
  MSHADOW_CUDA_POST_KERNEL_CHECK(im2col_gpu_kernel);
}
```

* 最后也是调用`im2col_gpu_kernel<DType>`的Kernel函数，这里只是多了根据输入数据尺寸计算输出数据尺寸的步骤

### Kernel - im2col_gpu_kernel(...)

GPU端实际计算用到的

``` cpp
template <typename DType>
__global__ void im2col_gpu_kernel(const int n, const DType* data_im,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int height_col, const int width_col,
    DType* data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    const int h_index = index / width_col;
    const int h_col = h_index % height_col;
    const int w_col = index % width_col;
    const int c_im = h_index / height_col;
    const int c_col = c_im * kernel_h * kernel_w;
    const int h_offset = h_col * stride_h - pad_h;
    const int w_offset = w_col * stride_w - pad_w;
    DType* data_col_ptr = data_col;
    data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;
    const DType* data_im_ptr = data_im;
    data_im_ptr += (c_im * height + h_offset) * width + w_offset;
    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        int h_im = h_offset + i * dilation_h;
        int w_im = w_offset + j * dilation_w;
        *data_col_ptr =
            (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ?
            data_im_ptr[i * dilation_h * width + j * dilation_w] : static_cast<DType>(0);
        data_col_ptr += height_col * width_col;
      }
    }
  }
}
```

* `CUDA_KERNEL_LOOP`的实现如下

  ``` cpp
  #define CUDA_KERNEL_LOOP(i, n) \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
        i < (n); \
        i += blockDim.x * gridDim.x)
  ```

  * 其实就是更新`i`，一个线程可以处理索引是`blockDim.x * gridDim.x`整数倍的数
  * `i`的起始值是`blockIdx.x * blockDim.x + threadIdx.x`

* 根据前面的代码，每个线程计算`ks * ks`个输出，也就是单线程内两层循环的作用了。在考虑实际实现的时候，可以想象一个由`inc * ks * ks * oh * ow`个像素块组成的二维矩形，长为`inc * ks * ks`，宽为`oh * ow`，其中每个像素块有一个颜色，不同颜色对应的不同的线程。这个大的二维矩形在列的方向，被分成`ks * ks`个小的矩形，每个小矩形块的高是`inc`，在每个小的矩形块中，是`inc * oh * ow`个小的像素块，注意，在每个小矩形块中，不同列的像素块的颜色不相同，但小矩形块中的一列的像素块的颜色是相同的，不同小矩形块相同列对应不同的颜色，表示一次循环，计算所有小矩形块中的同一行但属于不同列不同小矩形块的输出！

* 根据上面的解释，每个线程在每次循环对应的目标地址是``
* `data_col_ptr`就保存了存储结果的目标指针，当前线程当前循环的存储目的起始地址是：`data_col+w_col`

#### 小结

* 首先计算时，每`ks * ks`个输出数据使用一个线程完成计算，然后这些输出数据是连续的一列，一共有`inc * oh * oh`个线程。这个是实现代码的关键！
* 在`Kernel`函数内，`h_col, w_col`保存了输出数据的二维位置索引，`c_im`保存了输入数据的`channel`索引，`c_col`保存了在输出数据中`channel`的偏移量，这个参数与上面的`h_col, w_col`构成了输出数据的位置索引；`h_offset, w_offset`与`c_im`构成了输入数据的位置索引！
* 然后就是根据上面一步的参数计算输出、输入数据的指针
  * 输出数据的偏移地址是：`(c_col * height_col + h_col) * width_col + w_col` ，其实展开来看就是`c_col * oh * ow + h_col * ow + w_col`，或者吧：`c_col * heihgt_col`就是当前`h_col`之前存在这么多的行，以3维数据的结构为例更容易想
  * 输入数据的偏移地址是：`(c_im * height + h_offset) * width + w_offset`，理由同上！
* 最后就是里面两层循环完成单线程`ks * ks`个输出元素的赋值了，其他就没了

## col2im() GPU

* 整体实现思路与`im2col()`相同，这里就直接看下`kernel`函数怎么实现的吧
* 这里每个线程计算一个输出像素，注意这里的输出像素就是前向输入数据，形状是`inc * inh * inw`，只不过每个线程需要累加`ks * ks`个输入数据，也就是计算得到原始输入数据的梯度了！注意看`Backward（）`函数理解这句话就行了。
* 调用代码

  ``` cpp
      col2im_gpu_kernel<DType><<<cuda_get_num_blocks(im_size), mshadow::cuda::kBaseThreadNum,
                                 0, mshadow::Stream<gpu>::GetStream(s)>>>(
          im_size, data_col, im_shape[1], im_shape[2], im_shape[3],
          kernel_shape[0], kernel_shape[1], pad[0], pad[1], stride[0], stride[1],
          dilation[0], dilation[1], col_shape[1], col_shape[2], data_im, req);
      MSHADOW_CUDA_POST_KERNEL_CHECK(col2im_gpu_kernel);
  ```

### col2im_gpu_kernel()函数实现

``` cpp
template <typename DType>
__global__ void col2im_gpu_kernel(const int n, const DType* data_col,
    const int channels, const int height, const int width,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int height_col, const int width_col,
    DType* data_im, OpReqType req) {
  CUDA_KERNEL_LOOP(index, n) {
    DType val = 0;
    const int w_im = index % width + pad_w;
    const int h_im = (index / width) % height + pad_h;
    const int c_im = index / (width * height);
    int kernel_extent_w = (kernel_w - 1) * dilation_w + 1;
    int kernel_extent_h = (kernel_h - 1) * dilation_h + 1;
    // compute the start and end of the output
    const int w_col_start =
        (w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) / stride_w + 1;
    const int w_col_end = min(w_im / stride_w + 1, width_col);
    const int h_col_start =
        (h_im < kernel_extent_h) ? 0 : (h_im - kernel_extent_h) / stride_h + 1;
    const int h_col_end = min(h_im / stride_h + 1, height_col);
    // TODO(caffe): use LCM of stride and dilation to avoid unnecessary loops
    for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
      for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
        int h_k = (h_im - h_col * stride_h);
        int w_k = (w_im - w_col * stride_w);
        if (h_k % dilation_h == 0 && w_k % dilation_w == 0) {
          h_k /= dilation_h;
          w_k /= dilation_w;
          int data_col_index = (((c_im * kernel_h + h_k) * kernel_w + w_k) *
                                height_col + h_col) * width_col + w_col;
          val += data_col[data_col_index];
        }
      }
    }
    KERNEL_ASSIGN(data_im[index], req, val);
  }
}
```

* `w_im, h_im, c_im`对应的输入数据的位置索引，`w_col_start, w_col_end, h_col_start, h_col_end`对应的是输出数据的位置索引，这里的输入输出都是在`im2col`中的概念来说的
* 这里还需要考虑下 `im2col()`函数的实现，尤其是输入数据的过程。在那个函数中，输入数据的尺寸是`inc * inh * inw`，而输出的尺寸是`[inc/g * ks * ks, oh, ow]`，这个过程中的像素位置变化需要弄清楚。简单来说，输入数据中卷积窗口大小`inc / g * ks * ks`中的空间维像素`ks * ks`编程列了，然后不同`channel`维的`ks * ks`个数据在输出数据的列方向上连接起来，就是这个过程而已，所以才有了上面`im2col()`中的每个小矩形块的高是`ks * ks`，然后小矩形块的数量是`inc`。另一方面，这个过程导致输入数据中的相同像素在输出数据中不同列的不同行位置重复出现，这里不同列不同行的关系是，同一像素在相邻的两次卷积计算中属于不同的空间位置，所以在输出数据中体现在不同的行，具体相邻的两次卷积计算列相差一个像素；相邻的两次卷积计算在输出数据中体现为不同的列，所以同一像素在输出数据中页会出现在不同的列里面，一共重复出现`ks * ks`次。
* 补充一下上面的过程，输出数据中，`im2col()`结果中的一列对应的就是卷积计算结果中的一个像素，所以一个卷积核需要在输出数据列的方向上计算`oh * ow`次，完成一个`feature map`的计算。
* 有了上面的讨论，而`col2im(...)`就是上述过程的反过程中。所以具体操作就是，将输出数据中相邻的`ks * ks`列的相邻`ks * ks`行的数据进行累加，得到一个输入像素的梯度值。
* 再计算输出数据的位置索引时，可以通过三维数据的视角进行理解，这样就好一些理解了
* `h_k, w_k`中的`k`是`kernel`的意思，这两个数值表示的是`kernel`数据位置坐标体系中的位置索引。而`(c_im * kernel + h_k) * kernel_w + w_k`表示的是输出数据的`channel`索引，加上上面的`w_col, h_col`参数就可以唯一确定输出数据的位置索引了！这里需要注意的是这个`channel`维索引的计算，其具体计算原理是：`h_im - h_col * stride_h`计算出这个`kernel`的当前计算中目标输入像素所在对应`kernel`数据位置体系中的位置，具体分两步，首先是通过`h_im - h_col * stride_h`计算出存在`dilation`时的行位置，这个公式的原理可以简单思考下即可（`h_col * stride_h`相当于当前卷积窗口的中心位置！），第二步是去除这个`dilation`的影响。
* 总的来说，就是根据输入数据的位置，计算输出数据的位置，中间为了计算输出数据的`channel`维位置，需要考虑当前输入像素在卷积计算过程中对应`kernel`的哪个位置。整体流程就是这样了。更多的可以参考下代码中的注释。
* 这里本来就不需要`atomic`操作吧，因为一个线程对应一个不同的目的存储地址，而剩下的都是读操作，所以不需要互斥操作了。

## im2col_nd_gpu_kernel() 函数的实现

看这个函数代码很复杂，瞄一眼是怎么实现的吧，作为单独一节。源码如下：

``` cpp
template <typename DType, int num_axes>
__global__ void im2col_nd_gpu_kernel(const int n, const DType* data_im,
    const Shape<num_axes+2> im_shape, const Shape<num_axes+1> col_shape,
    const Shape<num_axes> kernel_shape, const Shape<num_axes> pad, const Shape<num_axes> stride,
    const Shape<num_axes> dilation, DType* data_col) {
  int d_temp[num_axes];  // NOLINT(runtime/arrays)
  int d_iter[num_axes];  // NOLINT(runtime/arrays)

  __shared__ int shared_dilation[num_axes];
  __shared__ int shared_kernel_shape[num_axes];
  __shared__ int shared_pad[num_axes];
  __shared__ int shared_stride[num_axes];
  __shared__ int shared_col_shape[num_axes + 1];
  __shared__ int shared_im_shape[num_axes + 1];

  if (threadIdx.x < num_axes) {
    shared_dilation[threadIdx.x] = dilation[threadIdx.x];
    shared_kernel_shape[threadIdx.x] = kernel_shape[threadIdx.x];
    shared_pad[threadIdx.x] = pad[threadIdx.x];
    shared_stride[threadIdx.x] = stride[threadIdx.x];
  }
  if (threadIdx.x < num_axes + 1) {
    shared_col_shape[threadIdx.x] = col_shape[threadIdx.x];
    shared_im_shape[threadIdx.x] = im_shape[threadIdx.x+1];  // skip batch dim
  }
  __syncthreads();

  int i;
  CUDA_KERNEL_LOOP(index, n) {
    // Initialize channel_in, computed in the loop below, with intermediate
    // computations used to compute the spatial indices.
    int channel_in = index;
    int channel_out = 1;
    for (i = num_axes - 1; i >= 0; --i) {
      d_temp[i] = channel_in % shared_col_shape[i + 1];
      channel_in /= shared_col_shape[i + 1];
      channel_out *= shared_kernel_shape[i];
    }
    channel_out *= channel_in;
    int data_col_inc = 1;
    for (i = 0; i < num_axes; ++i) {
      channel_out *= shared_col_shape[i + 1];
      channel_out += d_temp[i];
      d_temp[i] = d_temp[i] * shared_stride[i] - shared_pad[i];
      channel_in *= shared_im_shape[i + 1];
      channel_in += d_temp[i];
      data_col_inc *= shared_col_shape[i + 1];
      d_iter[i] = 0;
    }
    DType* data_col_ptr = data_col + channel_out;
    const DType* data_im_ptr = data_im + channel_in;
    bool incremented;
    do {
      bool in_range = true;
      for (i = 0; i < num_axes; ++i) {
        const int d_iter_im = d_iter[i] * shared_dilation[i] + d_temp[i];
        in_range &= d_iter_im >= 0 && d_iter_im < shared_im_shape[i + 1];
        if (!in_range) { break; }
      }
      if (in_range) {
        int data_im_offset = d_iter[0] * shared_dilation[0];
        for (i = 1; i < num_axes; ++i) {
          data_im_offset *= shared_im_shape[i + 1];
          data_im_offset += d_iter[i] * shared_dilation[i];
        }
        *data_col_ptr = data_im_ptr[data_im_offset];
      } else {
        *data_col_ptr = 0;
      }
      data_col_ptr += data_col_inc;
      incremented = false;
      for (i = num_axes - 1; i >= 0; --i) {
        const int d_max = shared_kernel_shape[i];
        if (d_iter[i] == d_max - 1) {
          d_iter[i] = 0;
        } else {  // d_iter[i] < d_max - 1
          ++d_iter[i];
          incremented = true;
          break;
        }
      }  // for (int i = num_axes - 1; i >= 0; --i)
    } while (incremented);  // do
  }  // CUDA_KERNEL_LOOP(index, n)
}
```

* 算了，以后再说吧

## 至此，`ConvolutionOp`的实现大概流程就是这样子了
