---
title: Convolution (3)
---

# ConvolutionOp源码分析 (3)

smh

2020.01.06

## 简介

* 这篇笔记主要是`ConvolutionOp`注册过程涉及到的函数的具体实现，具体内容见源码

## 源码分析

### Op注册代码

实际注册代码如下，这里忽略`Parameter`的注册过程，因为只需要一句话就行。

``` cpp
NNVM_REGISTER_OP(Convolution)
.describe(R"code(Compute *N*-D convolution on *(N+2)*-D input.

- **workspace**: A large number leads to more (GPU) memory usage but may improve
  the performance.

)code" ADD_FILELINE)
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

NNVM_REGISTER_OP(_backward_Convolution)
.set_num_outputs([](const NodeAttrs& attrs) {
  const ConvolutionParam& params = nnvm::get<ConvolutionParam>(attrs.parsed);
  return params.no_bias ? 2 : 3;
})
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
#if MXNET_USE_MKLDNN == 1
.set_attr<FInferStorageType>("FInferStorageType", BackwardConvStorageType)
#endif
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.set_attr_parser(ConvolutionParamParser)
#if MXNET_USE_MKLDNN == 1
.set_attr<bool>("TIsMKLDNN", true)
.set_attr<FComputeEx>("FComputeEx<cpu>", ConvolutionGradComputeExCPU)
#endif
.set_attr<FCompute>("FCompute<cpu>", ConvolutionGradCompute<cpu>);
```

* 包括了`Convolution, _backward_Convolution`两个`Op`的注册，在注册梯度计算`Op`时，不同的是多出了下面的代码：

  ``` cpp
  .set_attr<nnvm::TIsBackward>("TIsBackward", true)
  ```

  但是少了下面的代码：

  ``` cpp
  .set_attr<mxnet::FInferShape>("FInferShape", ConvolutionShape)
  .set_attr<nnvm::FInferType>("FInferType", ConvolutionType)
  ```

* 这侧过程中涉及到的具体实现函数见下面内容，主要分为前向、后向计算过程

## 前向过程

### set_num_inputs()

``` cpp
.set_num_inputs([](const NodeAttrs& attrs) {
  const ConvolutionParam& params = nnvm::get<ConvolutionParam>(attrs.parsed);
  return params.no_bias ? 2 : 3;
})
```

* 没什么要说的，就是根据是否有`bias`返回输入数据的个数
* 这里使用的是一个`[](const NodeAttr& attrs){}` Lambda函数的实现的，下面可以看到，这里也可以直接穿进去一个数值

### set_num_outputs()

``` cpp
.set_num_outputs(1)
```

* 这里直接穿进去一个数字`1`，而没有像上面那样使用`Lambda`

### set_attr_parser(ConvolutionParamParser)

* `ConvolutionParamParser()`是一个函数，定义如下

  ``` cpp
  void ConvolutionParamParser(nnvm::NodeAttrs* attrs) {
    using namespace mshadow;
    ConvolutionParam param_;
    try {
      param_.Init(attrs->dict);
    } catch (const dmlc::ParamError& e) {
      std::ostringstream os;
      os << e.what();
      os << ", in operator " << attrs->op->name << "("
         << "name=\"" << attrs->name << "\"";
      for (const auto& k : attrs->dict) {
        os << ", " << k.first << "=\"" << k.second << "\"";
      }
      os << ")";
      throw dmlc::ParamError(os.str());
    }
  
    if (param_.kernel.ndim() == 1) {
      param_.layout = param_.layout? param_.layout.value() : mshadow::kNCW;
      if (param_.stride.ndim() == 0) param_.stride = Shape1(1);
      if (param_.dilate.ndim() == 0) param_.dilate = Shape1(1);
      if (param_.pad.ndim() == 0) param_.pad = Shape1(0);
    } else if (param_.kernel.ndim() == 2) {
      param_.layout = param_.layout ? param_.layout.value() : mshadow::kNCHW;
      if (param_.stride.ndim() == 0) param_.stride = Shape2(1, 1);
      if (param_.dilate.ndim() == 0) param_.dilate = Shape2(1, 1);
      if (param_.pad.ndim() == 0) param_.pad = Shape2(0, 0);
    } else {
      CHECK_EQ(param_.kernel.ndim(), 3U) << param_.kernel.ndim() << "D convolution not supported";
      param_.layout = param_.layout ? param_.layout.value(): mshadow::kNCDHW;
      if (param_.stride.ndim() == 0) param_.stride = Shape3(1, 1, 1);
      if (param_.dilate.ndim() == 0) param_.dilate = Shape3(1, 1, 1);
      if (param_.pad.ndim() == 0) param_.pad = Shape3(0, 0, 0);
    }
    CHECK_EQ(param_.kernel.ndim(), param_.stride.ndim())
      << "Stride must have the same number of dimensions with kernel_size,"
      << "but kernel_size is set to " << param_.kernel << " while stride is "
      << param_.stride;
    CHECK_EQ(param_.kernel.ndim(), param_.dilate.ndim())
      << "Dilate must have the same number of dimensions with kernel_size,"
      << "but kernel_size is set to " << param_.kernel << " while dilate is "
      << param_.dilate;
    CHECK_EQ(param_.kernel.ndim(), param_.pad.ndim())
      << "Padding must have the same number of dimensions with kernel_size,"
      << "but kernel_size is set to " << param_.kernel << " while padding is "
      << param_.pad;
    attrs->parsed = std::move(param_);
  }
  ```

  * 首先也是调用`Init()`函数将`attrs->dict()`中的信息先保存在新建的`ConvolutionParam param_`里面
  * 然后在将剩下的参数进行设置，最后进行一次必要的检查
  * 最后将新建的`param_`保存在`attrs->parsed`里面
  *  

* 另一种实现是（以`FullyConnected`）为例进行示范

  ``` cpp
  .set_attr_parser(ParamParser<FullyConnectedParam>)
  ```

  * 这里使用了`ParamParser`函数来实现相同的功能，这个函数里面会调用`FullyConnectedParam.Init()`函数，其实这个函数是基类`Parameter`内实现的，而实际上最终调用的就是`ParamManager::RunInit()`函数实现具体的赋值，最最终，调用的是`FieldAccessEntry::Set(), Check()`等函数，`ParamParser()`函数的实现如下：

    ``` cpp
    template<typename PType>
    inline void ParamParser(nnvm::NodeAttrs* attrs) {
      PType param;
      try {
        param.Init(attrs->dict);
      } catch (const dmlc::ParamError& e) {
        std::ostringstream os;
        os << e.what();
        // op->name 就是 Op::name 成员
        os << ", in operator " << attrs->op->name << "("
           << "name=\"" << attrs->name << "\"";
        for (const auto& k : attrs->dict) {
          os << ", " << k.first << "=\"" << k.second << "\"";
        }
        os << ")";
        throw dmlc::ParamError(os.str());
      }
      attrs->parsed = std::move(param);
    }
    ```

* 这个属性为可选操作，但一般都有

### set_attrs()系列函数

这一系列调用就是为新增加的`Operator`添加各种属性，具体的属性包括：

* `FListInputNames`
* `FListOutputNames`
* `FInferShape`
* `nnvm::FInferType`
* `FCompute`
* `FComputeEx`
* `TIsMKLDNN`
* `FGradient`
* `FResourceRequest`
* 可选的`FInplaceOption`等

### FListInputNames, FListOutputNames

``` cpp
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
```

* 比较简单，关键是这里`Lambda`匿名函数的类型

  ``` cpp
  std::vector<std::string>(const NodeAttrs& attrs)
  ```

* 其它的应该就没了吧。。。

### set_attr\<mxnet::FInferShape\>(...)实现

``` cpp
static bool ConvolutionShape(const nnvm::NodeAttrs& attrs,
                             mxnet::ShapeVector *in_shape,
                             mxnet::ShapeVector *out_shape) {
  using namespace mshadow;
  const ConvolutionParam& param_ = nnvm::get<ConvolutionParam>(attrs.parsed);
  if (!param_.no_bias) {
    CHECK_EQ(in_shape->size(), 3U) << "Input:[data, weight, bias]";
  } else {
    CHECK_EQ(in_shape->size(), 2U) << "Input:[data, weight]";
  }
  // CHECK_EQ(out_shape->size(), 1) << "Output: [output]";
  out_shape->resize(1, mxnet::TShape());
  const mxnet::TShape &dshp = (*in_shape)[conv::kData];
  if (!mxnet::ndim_is_known(dshp)) return false;

  if (param_.kernel.ndim() == 1) {
    // 1d conv
    CHECK_EQ(dshp.ndim(), 3U) << "Input data should be 3D in batch-num_filter-x";
    Shape<3> dshape = ConvertLayout(dshp.get<3>(), param_.layout.value(), kNCW);
    Shape<3> wshape = Shape3(param_.num_filter / param_.num_group,
        mxnet::dim_size_is_known(dshape, 1) ? dshape[1] / param_.num_group : -1,
        param_.kernel[0]);
    wshape = ConvertLayout(wshape, kNCW, param_.layout.value());
    if (wshape[0] >= 0) {
      wshape[0] *= param_.num_group;
    }
    // 实际调用 mxnet::op::sahpe_assign(...) 完成操作，这个函数里面，首先会 检查 *in_shape[kWeight] 的 ndim() 是否已知，若未知，则直接用 wshape 赋值给 *in_shape[kWeight]
    //   如果ndim()已知，则按照每个维度的尺寸进行赋值
    SHAPE_ASSIGN_CHECK(*in_shape, conv::kWeight, wshape);
    if (!param_.no_bias) {
      SHAPE_ASSIGN_CHECK(*in_shape, conv::kBias, Shape1(param_.num_filter));
    }

    const index_t dilated_ksize_x = param_.DilatedKernelSize(0);
    if (dshape[1] != -1) {
      CHECK_EQ(dshape[1] % param_.num_group, 0U) << "input num_filter must divide group size";
    }
    CHECK_EQ(param_.num_filter % param_.num_group, 0U) \
      << "output num_filter must divide group size";
    CHECK_GT(param_.kernel.Size(), 0U) \
      << "incorrect kernel size: " << param_.kernel;
    CHECK_GT(param_.stride.Size(), 0U) \
      << "incorrect stride size: " << param_.stride;
    CHECK_GT(param_.dilate.Size(), 0U) \
      << "incorrect dilate size: " << param_.dilate;
    Shape<3> oshape;
    oshape[0] = dshape[0];
    oshape[1] = param_.num_filter;
    oshape[2] = dshape[2] != -1 ?
      (AddPad(dshape[2], param_.pad[0]) - dilated_ksize_x) / param_.stride[0] + 1 : -1;
    SHAPE_ASSIGN_CHECK(*out_shape, 0, ConvertLayout(oshape, kNCW, param_.layout.value()));
    // Perform incomplete shape inference. Fill in the missing values in data shape.
    // 1) We can always fill in the batch_size.
    // 2) We can back-calculate the input height/width if the corresponding stride is 1.
    oshape = ConvertLayout((*out_shape)[0].get<3>(), param_.layout.value(), kNCW);
    dshape[0] = oshape[0];
    if (oshape[2] != -1 && param_.stride[0] == 1) {
      dshape[2] = oshape[2] + dilated_ksize_x - 1 - 2 * param_.pad[0];
    }
    SHAPE_ASSIGN_CHECK(*in_shape, conv::kData,
        ConvertLayout(dshape, kNCW, param_.layout.value()));
    // Check whether the kernel sizes are valid
    if (dshape[2] != -1) {
      CHECK_LE(dilated_ksize_x, AddPad(dshape[2], param_.pad[0])) << "kernel size exceed input";
    }
    return true;
  } else if (param_.kernel.ndim() == 2) {
    // 2d conv
    CHECK_EQ(dshp.ndim(), 4U) \
      << "Input data should be 4D in batch-num_filter-y-x";
    Shape<4> dshape = ConvertLayout(dshp.get<4>(), param_.layout.value(), kNCHW);
    Shape<4> wshape = Shape4(param_.num_filter / param_.num_group,
        mxnet::dim_size_is_known(dshape, 1) ? dshape[1] / param_.num_group : -1,
        param_.kernel[0], param_.kernel[1]);
    wshape = ConvertLayout(wshape, kNCHW, param_.layout.value());
    if (wshape[0] >= 0) {
      wshape[0] *= param_.num_group;
    }
    SHAPE_ASSIGN_CHECK(*in_shape, conv::kWeight, wshape);
    if (!param_.no_bias) {
      SHAPE_ASSIGN_CHECK(*in_shape, conv::kBias, Shape1(param_.num_filter));
    }

    const index_t dilated_ksize_y = param_.DilatedKernelSize(0);
    const index_t dilated_ksize_x = param_.DilatedKernelSize(1);
    if (dshape[1] != -1) {
      CHECK_EQ(dshape[1] % param_.num_group, 0U) << "input num_filter must divide group size";
    }
    CHECK_EQ(param_.num_filter % param_.num_group, 0U) \
      << "output num_filter must divide group size";
    CHECK_GT(param_.kernel.Size(), 0U) \
      << "incorrect kernel size: " << param_.kernel;
    CHECK_GT(param_.stride.Size(), 0U) \
      << "incorrect stride size: " << param_.stride;
    CHECK_GT(param_.dilate.Size(), 0U) \
      << "incorrect dilate size: " << param_.dilate;
    Shape<4> oshape;
    oshape[0] = dshape[0];
    oshape[1] = param_.num_filter;
    oshape[2] = dshape[2] != -1 ?
      (AddPad(dshape[2], param_.pad[0]) - dilated_ksize_y) / param_.stride[0] + 1 : -1;
    oshape[3] = dshape[3] != -1 ?
      (AddPad(dshape[3], param_.pad[1]) - dilated_ksize_x) / param_.stride[1] + 1 : -1;
    SHAPE_ASSIGN_CHECK(*out_shape, 0, ConvertLayout(oshape, kNCHW, param_.layout.value()));
    // Perform incomplete shape inference. Fill in the missing values in data shape.
    // 1) We can always fill in the batch_size.
    // 2) We can back-calculate the input height/width if the corresponding stride is 1.
    oshape = ConvertLayout((*out_shape)[0].get<4>(), param_.layout.value(), kNCHW);
    dshape[0] = oshape[0];
    if (oshape[2] != -1 && param_.stride[0] == 1) {
      dshape[2] = oshape[2] + dilated_ksize_y - 1 - 2 * param_.pad[0];
    }
    if (oshape[3] != -1 && param_.stride[1] == 1) {
      dshape[3] = oshape[3] + dilated_ksize_x - 1 - 2 * param_.pad[1];
    }
    SHAPE_ASSIGN_CHECK(*in_shape, conv::kData,
        ConvertLayout(dshape, kNCHW, param_.layout.value()));
    // Check whether the kernel sizes are valid
    if (dshape[2] != -1) {
      CHECK_LE(dilated_ksize_y, AddPad(dshape[2], param_.pad[0])) << "kernel size exceed input";
    }
    if (dshape[3] != -1) {
      CHECK_LE(dilated_ksize_x, AddPad(dshape[3], param_.pad[1])) << "kernel size exceed input";
    }
    return true;
  } else if (param_.kernel.ndim() == 3) {
    // 3d conv
    CHECK_EQ(dshp.ndim(), 5U) \
      << "Input data should be 5D in batch-num_filter-depth-y-x";
    Shape<5> dshape = ConvertLayout(dshp.get<5>(), param_.layout.value(), kNCDHW);
    Shape<5> wshape = Shape5(param_.num_filter / param_.num_group,
        mxnet::dim_size_is_known(dshape, 1) ? dshape[1] / param_.num_group : -1,
        param_.kernel[0], param_.kernel[1], param_.kernel[2]);
    wshape = ConvertLayout(wshape, kNCDHW, param_.layout.value());
    if (wshape[0] >= 0) {
      wshape[0] *= param_.num_group;
    }
    SHAPE_ASSIGN_CHECK(*in_shape, conv::kWeight, wshape);
    if (!param_.no_bias) {
      SHAPE_ASSIGN_CHECK(*in_shape, conv::kBias, Shape1(param_.num_filter));
    }

    // Note: 3D dilation currently not supported.
    // Calculations below done to preserve symmetry with 1D/2D code.
    const index_t dilated_ksize_d = param_.DilatedKernelSize(0);
    const index_t dilated_ksize_y = param_.DilatedKernelSize(1);
    const index_t dilated_ksize_x = param_.DilatedKernelSize(2);
    if (dshape[1] >= 0) {
      CHECK_EQ(dshape[1] % param_.num_group, 0U) << "input num_filter must divide group size";
    }
    CHECK_EQ(param_.num_filter % param_.num_group, 0U)
      << "output num_filter must divide group size";
    CHECK_GT(param_.kernel.Size(), 0U) \
      << "incorrect kernel size: " << param_.kernel;
    CHECK_GT(param_.stride.Size(), 0U) \
      << "incorrect stride size: " << param_.stride;
    CHECK_GT(param_.dilate.Size(), 0U) \
      << "incorrect dilate size: " << param_.dilate;
    CHECK_EQ(param_.dilate.Size(), 1U)
      << "Dilate is not supported in 3d convolution";
    Shape<5> oshape;
    oshape[0] = dshape[0];
    oshape[1] = param_.num_filter;
    oshape[2] = dshape[2] != -1 ?
      (AddPad(dshape[2], param_.pad[0]) - dilated_ksize_d) / param_.stride[0] + 1 : -1;
    oshape[3] = dshape[3] != -1 ?
      (AddPad(dshape[3], param_.pad[1]) - dilated_ksize_y) / param_.stride[1] + 1 : -1;
    oshape[4] = dshape[4] != -1 ?
      (AddPad(dshape[4], param_.pad[2]) - dilated_ksize_x) / param_.stride[2] + 1 : -1;
    SHAPE_ASSIGN_CHECK(*out_shape, 0, ConvertLayout(oshape, kNCDHW, param_.layout.value()));
    // Perform incomplete shape inference. Fill in the missing values in data shape.
    // 1) We can always fill in the batch_size.
    // 2) We can back-calculate the input depth/height/width if the corresponding stride is 1.
    oshape = ConvertLayout((*out_shape)[0].get<5>(), param_.layout.value(), kNCDHW);
    dshape[0] = oshape[0];
    if (oshape[2] != -1 && param_.stride[0] == 1) {
      dshape[2] = oshape[2] + dilated_ksize_d - 1 - 2 * param_.pad[0];
    }
    if (oshape[3] != -1 && param_.stride[1] == 1) {
      dshape[3] = oshape[3] + dilated_ksize_y - 1 - 2 * param_.pad[1];
    }
    if (oshape[4] != -1 && param_.stride[2] == 1) {
      dshape[4] = oshape[4] + dilated_ksize_x - 1 - 2 * param_.pad[2];
    }
    SHAPE_ASSIGN_CHECK(*in_shape, conv::kData,
        ConvertLayout(dshape, kNCDHW, param_.layout.value()));
    // Check whether the kernel sizes are valid
    if (dshape[2] != -1) {
      CHECK_LE(dilated_ksize_d, AddPad(dshape[2], param_.pad[0])) << "kernel size exceed input";
    }
    if (dshape[3] != -1) {
      CHECK_LE(dilated_ksize_y, AddPad(dshape[3], param_.pad[1])) << "kernel size exceed input";
    }
    if (dshape[4] != -1) {
      CHECK_LE(dilated_ksize_x, AddPad(dshape[4], param_.pad[2])) << "kernel size exceed input";
    }
    return true;
  } else {
    LOG(FATAL) << "Unknown convolution type";
    return false;
  }
}
```

* 重点是里面的`Shape_ASSIGN_CHECK()`宏函数的实现

  ``` cpp
  #define SHAPE_ASSIGN_CHECK(shape_array, index, shape)                       \
    {                                                                         \
      if (!::mxnet::op::shape_assign(&(shape_array)[index], mxnet::TShape(shape))) { \
        std::ostringstream os;                                                \
        os << "Shape inconsistent, Provided = " << (shape_array)[index] << ','\
           << " inferred shape=" << shape;                                    \
        throw ::mxnet::op::InferShapeError(os.str(), index);                  \
      }                                                                       \
    }
  ```

  * 可以看出来，实际调用的是`mxnet::op::shape_assign()`这个函数

* `mxnet::op::shape_assign()`函数的实现

  ``` cpp
  inline bool shape_assign(mxnet::TShape *y, const mxnet::TShape& x) {
    // y.ndim 未知，直接赋值
    if (!mxnet::ndim_is_known(*y)) {
      *y = x;
      return true;
    } else if (y->ndim() != x.ndim()) {
      return !mxnet::ndim_is_known(x);
      // y.ndim 已知，则按照对应的对应的每个维度进行赋值
    } else {
      for (int i = 0; i < y->ndim(); ++i) {
        // dim_size_is_known() 实际是 *y[i] != -1 返回的结果
        if (!mxnet::dim_size_is_known(*y, i)) {
          (*y)[i] = x[i];
        } else if ((*y)[i] != x[i] && x[i] >= 0) {      // 已知的尺寸不匹配
          return false;
        }
      }
      return true;
    }
  }
  ```

  * 分了三种情况：
    * `y.ndim()`未知，则直接将`x`的值赋值给`y`
    * `y.ndim()`已知，但与`x.ndim()`不一致，若`x.ndim()`可能未知的情况，则直接返回正确，否则返回错误
    * 如果`y.ndim() == n.ndim()`，即`y.ndim()`已知且与`x.ndim()`一致，则将`x`中每个维度的尺寸信息复制给`y`对应的维度

* `ConvolutionShape()`函数的实现过程，其实就是分了几种情况，这里以`param_.kernel.ndim() == 2`的情况为例进行说明

  * wshape

    ``` cpp
        Shape<4> wshape = Shape4(param_.num_filter / param_.num_group,
            mxnet::dim_size_is_known(dshape, 1) ? dshape[1] / param_.num_group : -1,
            param_.kernel[0], param_.kernel[1]);
    ```

    * 所以，实际`weight`的形状是`outc / g, inc / g, ks, ks`

  * bias，比较简单，见源码
  * 然后就是四个`CHECK`语句了
  * 然后就是计算输出的尺寸，这里使用`AddPad()`等函数，实际计算就是按照卷积输出输出空间尺寸的计算公式进行的，这里需要注意的是，后面还会根据输出数据的尺寸去初始化输入数据的尺寸，做到前向后向的形状推理，因为有写尺寸可能需要根据输出数据的尺寸进行计算！

    ``` cpp
        Shape<4> oshape;
        oshape[0] = dshape[0];        // bs
        oshape[1] = param_.num_filter;    // outc
        oshape[2] = dshape[2] != -1 ?
          (AddPad(dshape[2], param_.pad[0]) - dilated_ksize_y) / param_.stride[0] + 1 : -1;     // 计算输出的空间尺寸
        oshape[3] = dshape[3] != -1 ?
          (AddPad(dshape[3], param_.pad[1]) - dilated_ksize_x) / param_.stride[1] + 1 : -1;
        SHAPE_ASSIGN_CHECK(*out_shape, 0, ConvertLayout(oshape, kNCHW, param_.layout.value()));
        // Perform incomplete shape inference. Fill in the missing values in data shape.
        // 1) We can always fill in the batch_size.
        // 2) We can back-calculate the input height/width if the corresponding stride is 1.
        // 根据输出数据计算输入数据的尺寸
        oshape = ConvertLayout((*out_shape)[0].get<4>(), param_.layout.value(), kNCHW);
        dshape[0] = oshape[0];
        if (oshape[2] != -1 && param_.stride[0] == 1) {
          dshape[2] = oshape[2] + dilated_ksize_y - 1 - 2 * param_.pad[0];
        }
        if (oshape[3] != -1 && param_.stride[1] == 1) {
          dshape[3] = oshape[3] + dilated_ksize_x - 1 - 2 * param_.pad[1];
        }
        SHAPE_ASSIGN_CHECK(*in_shape, conv::kData,
            ConvertLayout(dshape, kNCHW, param_.layout.value()));
    ```

  * 最后就是对`kernel`的尺寸与输入数据的尺寸进行检查，也就是说输入数据的长、宽要大于等于考虑`pad`之后的输入数据的空间尺寸，注意，这里要**大于等于**

    ``` cpp
        if (dshape[2] != -1) {
          CHECK_LE(dilated_ksize_y, AddPad(dshape[2], param_.pad[0])) << "kernel size exceed input";
        }
        if (dshape[3] != -1) {
          CHECK_LE(dilated_ksize_x, AddPad(dshape[3], param_.pad[1])) << "kernel size exceed input";
        }
    ```

  * 其它的就没了

### set_attr<nnvm::FInferType>(...)函数

``` cpp
static bool ConvolutionType(const nnvm::NodeAttrs& attrs,
                            std::vector<int> *in_type, std::vector<int> *out_type) {
  const ConvolutionParam& param_ = nnvm::get<ConvolutionParam>(attrs.parsed);
  CHECK_GE(in_type->size(), 1U);
  int dtype = (*in_type)[0];
  CHECK_NE(dtype, -1) << "First input must have specified type";
  for (size_t i = 0; i < in_type->size(); ++i) {
    if ((*in_type)[i] == -1) {
      (*in_type)[i] = dtype;
    } else {
      UNIFORM_TYPE_CHECK((*in_type)[i], dtype, ListArguments(param_)[i]);
    }
  }
  out_type->clear();
  out_type->push_back(dtype);
  return true;
}
```

* 首先是对输入`in_type`中所有元素的`data type`设置对，这里就是使用`in_type[0]`的`dtype`作为所有输入的`dtype`
* 然后就是设置输出的`dtype`，具体类型也是`in_type[0]`所表示的`dtype`，注意这个值不能是`-1`，也就是说必须是已知的

这个属性还有另一种实现，同样以`FullyConnected`为例：

``` cpp
static bool FullyConnectedType(const nnvm::NodeAttrs& attrs,
                               std::vector<int> *in_type, std::vector<int> *out_type) {
  CHECK_GE(in_type->size(), 1U);
  return ElemwiseAttr<int, type_is_none, type_assign, true, type_string>(
      attrs, in_type, out_type, -1);
}
```

* 这里调用了`ElemwiseAttr()`这个模板函数进行实现，这个函数的实现为：

  ``` cpp
  template<typename AttrType, bool (*is_none)(const AttrType&),
           bool (*assign)(AttrType*, const AttrType&), bool reverse_infer,
           std::string (*attr_string)(const AttrType&),
           index_t n_in = -1, index_t n_out = -1>
  inline bool ElemwiseAttr(const nnvm::NodeAttrs& attrs,
                           std::vector<AttrType> *in_attrs,
                           std::vector<AttrType> *out_attrs,
                           const AttrType& none) {
    AttrType dattr = none;
    size_t in_size = in_attrs->size();
    size_t out_size = out_attrs->size();
    if (n_in != -1)
      in_size = static_cast<size_t>(n_in);
    if (n_out != -1)
      out_size = static_cast<size_t>(n_out);
  
    CHECK_LE(in_size, in_attrs->size());
    CHECK_LE(out_size, out_attrs->size());
    auto deduce = [&](const std::vector<AttrType>& vec, size_t size, const char *name) {
        for (size_t i = 0; i < size; ++i) {
          CHECK(assign(&dattr, vec.at(i)))
            << "Incompatible attr in node " << attrs.name << " at " << i << "-th "
            << name << ": " << "expected " << attr_string(dattr)
            << ", got " << attr_string(vec.at(i));
        }
      };
    deduce(*in_attrs, in_size, "input");
    if (reverse_infer)
        deduce(*out_attrs, out_size, "output");
  
    auto write = [&](std::vector<AttrType> *vec, size_t size, const char *name) {
        for (size_t i = 0; i < size; ++i) {
          CHECK(assign(&(vec->at(i)), dattr))
            << "Incompatible attr in node " << attrs.name << " at " << i << "-th "
            << name << ": " << "expected " << attr_string(dattr)
            << ", got " << attr_string(vec->at(i));
        }
      };
    write(in_attrs, in_size, "input");
    write(out_attrs, out_size, "output");
  
    if (is_none(dattr))
        return false;
    return true;
  }
  ```

### FInferStorageType

``` cpp
inline static bool ConvStorageType(const nnvm::NodeAttrs& attrs,
                                   const int dev_mask,
                                   DispatchMode* dispatch_mode,
                                   std::vector<int>* in_attrs,
                                   std::vector<int>* out_attrs) {
  const ConvolutionParam& param = nnvm::get<ConvolutionParam>(attrs.parsed);
  uint32_t in_expected = param.no_bias ? 2 : 3;
  CHECK_EQ(in_attrs->size(), in_expected);
  CHECK_EQ(out_attrs->size(), 1);

  return MKLDNNStorageType(attrs, dev_mask, true, dispatch_mode, in_attrs,
                           out_attrs);
}
```

* 这个函数首先进行类型检查，然后调用`MKLDNNStorage(...)`函数，这个函数的实现如下：

  ``` cpp
  bool MKLDNNStorageType(const nnvm::NodeAttrs &attrs,
                         const int dev_mask,
                         bool support_mkldnn,
                         DispatchMode *dispatch_mode,
                         std::vector<int> *in_attrs,
                         std::vector<int> *out_attrs) {
    for (int& v : *in_attrs)
      if (v == - 1) v = kDefaultStorage;
  
    DispatchMode wanted_mode;
  #if MXNET_USE_MKLDNN == 1
    if (dev_mask == mshadow::cpu::kDevMask && !MKLDNNEnvSet())
      wanted_mode = DispatchMode::kFComputeFallback;
    else if (dev_mask == mshadow::cpu::kDevMask && support_mkldnn)
      wanted_mode = DispatchMode::kFComputeEx;
    else
  #endif
      wanted_mode = DispatchMode::kFCompute;
  
    bool dispatched = false;
    if (!dispatched && common::ContainsOnlyStorage(*in_attrs, kDefaultStorage)) {
      dispatched = op::storage_type_assign(out_attrs, mxnet::kDefaultStorage,
                                           dispatch_mode, wanted_mode);
    }
    if (!dispatched) {
      dispatched = op::dispatch_fallback(out_attrs, dispatch_mode);
    }
    return dispatched;
  }
  ```

  * `DispatchMode`枚举类定义了`kUndefined, kFCompute, kFComputeEx, kFComputeFallback, kVariable`几个成员
  * `MKLDNNEnvSet()`函数返回`MXNET_MKLDNN_ENABLED`系统变量 的值，默认为`true`
  * 这个函数，其实就是决定使用`kFComputeFallback, kFComputeEx, kFCompute`这几种的计算类型中的哪一个
  * `ContainsOnlyStorage()`函数的实现

    ``` cpp
    inline bool ContainsOnlyStorage(const StorageTypeVector& vstorage,
                                    const NDArrayStorageType stype) {
      if (!vstorage.empty()) {
        for (const auto& i : vstorage) {
          if (i != stype) return false;
        }
        return true;
      }
      return false;
    }
    ```

    * 也就是传进来的`vstorage`非空，且所有元素都等于`stype`时才返回`true`

  * `storage_type_assign()`函数的实现

    ``` cpp
    inline bool storage_type_assign(StorageTypeVector* stypes,
                                    const NDArrayStorageType target_stype,
                                    DispatchMode* dispatch,
                                    const DispatchMode target_dispatch) {
      CHECK_GT(stypes->size(), 0);
      bool success = true;
      for (int& stype : *stypes) {
        if (!type_assign(&stype, target_stype)) {
          success = false;
        }
      }
      if (success) {
        DISPATCH_MODE_ASSIGN_CHECK(dispatch, 0, target_dispatch);
      }
      return success;
    }
    ```

    * `type_assign()`函数的实现如下

      ``` cpp
      inline bool type_assign(int *y, const int& x) {
        if (*y == -1) {
          *y = x;
          return true;
        } else if (*y != x && x != -1) {
          return false;
        }
        return true;
      }
      ```

      * 就是将`y`根据`x`进行赋值

    * `storage_type_assign()`函数就是将`stypes`中的每个元素赋值为`target_stype`，然后调用`DISPATCH_MODE_ASSIGN_CHECK(...)`

      ``` cpp
      #define DISPATCH_MODE_ASSIGN_CHECK(type_array, index, type)                 \
        {                                                                         \
          if (!::mxnet::op::dispatch_mode_assign(&(type_array)[index], type)) {   \
            std::ostringstream os;                                                \
            os << "Dispatch mode inconsistent, Provided = "                       \
               << common::dispatch_mode_string((type_array)[index]) << ','        \
               << " inferred mode = " << common::dispatch_mode_string(type);      \
            throw ::mxnet::op::InferStorageTypeError(os.str(), index);            \
          }                                                                       \
        }
      ```

      * `dispatch_mode_assign()`就是将`type`赋值给`&(type_array)[index]`
      * 其它的没了

  * `dispatch_fallback()`函数

    ``` cpp
    inline bool dispatch_fallback(StorageTypeVector* stypes, DispatchMode* dispatch) {
      for (auto& stype : *stypes) {
        type_assign(&stype, kDefaultStorage);
      }
      DISPATCH_MODE_ASSIGN_CHECK(dispatch, 0, DispatchMode::kFComputeFallback);
      return true;
    }
    ```

    * 就是将`stypes`中的所有元素设置为`kDefaultStorage`，而将`dispatch[0]`设置为`kFComputeFallback`

* 暂时没有了

### FCompute, FComputeEx

* 这一个就是前向计算函数了，具体实现见前面的代码分析

### FGradient

``` cpp
.set_attr<nnvm::FGradient>("FGradient", ConvolutionGrad{"_backward_Convolution"})
```

* `ConvolutionGrad`的实现如下：

  ``` cpp
  struct ConvolutionGrad {
    const char *op_name;
    std::vector<nnvm::NodeEntry> operator()(const nnvm::NodePtr& n,
                                            const std::vector<nnvm::NodeEntry>& ograds) const {
      const ConvolutionParam& param = nnvm::get<ConvolutionParam>(n->attrs.parsed);
      std::vector<nnvm::NodeEntry> heads(ograds.begin(), ograds.end());
      heads.push_back(n->inputs[conv::kData]);
      heads.push_back(n->inputs[conv::kWeight]);
      if (!param.no_bias)
        heads.push_back(n->inputs[conv::kBias]);
      return MakeGradNode(op_name, n, heads, n->attrs.dict);
    }
  };
  ```

  * 包含一个`op_name`表示后向传播过程所使用的`Operator`的名字
  * 然后就是创建一个`Node`了，`MakeGradNode`的实现如下

    ``` cpp
    inline std::vector<nnvm::NodeEntry> MakeGradNode(
        const char* op_name, const nnvm::NodePtr& n,
        const std::vector<nnvm::NodeEntry>& inputs,
        const std::unordered_map<std::string, std::string>& dict) {
      auto p = MakeNode(op_name, n->attrs.name + "_backward",
                        &inputs, &dict, &n);
      std::vector<nnvm::NodeEntry> ret;
      for (uint32_t i = 0; i < p->num_outputs(); ++i) {
        ret.emplace_back(p, i, 0);
      }
      return ret;
    }
    ```

    * 实际调用的是`MakeNode(...)`函数，在这个函数里面，会根据传入的后向计算的`op name`以及设置`Node::attrs, comtrol_deps`等成员创建一个`Node`，然后返回
    * 注意这里，返回的是一个`vector<nnvm::NodeEntry>`也就是后向传播计算函数的所有输出的`NodeEntry`，并且这个向量里面每个元素（类型是`NodeEntry`）的`NodePtr node`成员都指向这个后向传播计算节点
  * 另外一点，这个可调用类的重载的`()`运算符的声明是

    ``` cpp
    std::vector<nnvm::NodeEntry>(const nnvm::NodePtr& n,
                                 const std::vector<nnvm::NodeEntry>& ograds)
    ```

* 所以这里会创建一个`Node`来实现后向传播计算，且这个注册的函数的返回的是后向传播过程计算的结果，即`vector<nnvm::NodeEntry>`

另一种实现是使用`ElemwiseGradUseIn{op_name}, ElemwiseGradUseOut{op_name}, ElemwiseGradUseNone{op_name}`，分别对应后向传播需要输入数据、输出数据、或者无需操作。具体实现如下：

``` cpp
// 见另一篇笔记吧
```

* 见下一篇笔记，单独分析

### FResourceRequest

``` cpp
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
```

* 主要是就是这个`Lambda`的实现了，然后资源管理可以参考对应的代码分析笔记

### add_argument()系列了

* 就是将`Operator`需要的操作数按照顺序添加进去即可，没有啥要说的

## 后向过程

``` cpp
NNVM_REGISTER_OP(_backward_Convolution)
.set_num_outputs([](const NodeAttrs& attrs) {
  const ConvolutionParam& params = nnvm::get<ConvolutionParam>(attrs.parsed);
  return params.no_bias ? 2 : 3;
})
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
#if MXNET_USE_MKLDNN == 1
.set_attr<FInferStorageType>("FInferStorageType", BackwardConvStorageType)
#endif
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.set_attr_parser(ConvolutionParamParser)
#if MXNET_USE_MKLDNN == 1
.set_attr<bool>("TIsMKLDNN", true)
.set_attr<FComputeEx>("FComputeEx<cpu>", ConvolutionGradComputeExCPU)
#endif
.set_attr<FCompute>("FCompute<cpu>", ConvolutionGradCompute<cpu>);
```

* 首先名字可以看出来，就是前向计算过程中`FGradient`里面的名字，那里使用了一个`ConvolutionGrad`可调用类来实现，具体实现过程见上面对这个类的代码分析
* `set_num_outputs()`，具体参考实现代码
* `set_attr<nnvm::TIsBackward>("TIsBackward", true)`

  这个属性前向过程不需要设置

* `set_attr<FInferStorageType>`

  ``` cpp
  inline static bool BackwardConvStorageType(const nnvm::NodeAttrs& attrs,
                                             const int dev_mask,
                                             DispatchMode* dispatch_mode,
                                             std::vector<int> *in_attrs,
                                             std::vector<int> *out_attrs) {
    const ConvolutionParam& param = nnvm::get<ConvolutionParam>(attrs.parsed);
    uint32_t in_expected = param.no_bias ? 3 : 4;
    uint32_t out_expected = param.no_bias ? 2 : 3;
    CHECK_EQ(in_attrs->size(), in_expected);
    CHECK_EQ(out_attrs->size(), out_expected);
  
    return MKLDNNStorageType(attrs, dev_mask, true, dispatch_mode, in_attrs,
                             out_attrs);
  }
  ```

  * 首先对输入进行检查
  * 然后调用`MKLDNNStorageType(...)`函数，这个函数的实现见上面前向过程的分析

* `set_attr<FResourceRequest>`

  ``` cpp
  .set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
  ```

  * 表明后向计算中需要临时空间

* `set_attr_parser`，与前向过程中用到的函数一致
* `FComputeEx, FCompute`分别对应的是`ConvolutionGradComputeExCPU, ConvolutionGradCompute<cpu>`两个函数
