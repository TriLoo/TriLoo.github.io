---
title: Convolution (4)
---

# ConvolutionOp源码分析 (4)

smh

2020.01.06

## 简介

* 这个文件主要是分析注册后向传播计算函数时的具体实现，在上一篇中已经给出了一种`Convolution`的实现，就是自定义了一个`ConvolutionGrad`类，重载的`()`运算符类型是：

  ``` cpp
  std::vector<nnvm::NodeEntry>(const nnvm::NodePtr& n,
                               const std::vector<nnvm::NodeEntry>& ograds)
  ```

  * 在这个函数里面，会创建一个`GradNode`来完成计算，返回的是这个梯度计算函数所有的输出构成的`vector<nnvm::NodeEntry>`
* 主要是`ElemwiseGradUseIn{op_name}, ElemwiseGradUseOut{op_name}, ElemwiseGradUseNone{op_name}`几个类的实现

## ElemwiseGradUseIn

这种情况会用到输入数据来计算梯度。以`abs`操作为例进行说明。

``` cpp
struct ElemwiseGradUseIn {
  const char *op_name;
  std::vector<nnvm::NodeEntry> operator()(const nnvm::NodePtr& n,
                                          const std::vector<nnvm::NodeEntry>& ograds) const {
    return MakeNonlossGradNode(op_name, n, ograds, n->inputs, n->attrs.dict);
  }
};
```

* 主要就是调用`MakeNonlossGradNode()`函数

  ``` cpp
  // make gradient node that doesn't add to objective.
  // i.e. igrads are always zero when ograds are zero.
  inline std::vector<nnvm::NodeEntry> MakeNonlossGradNode(
      const char* op_name, const nnvm::NodePtr& n,
      const std::vector<nnvm::NodeEntry>& ograds,
      const std::vector<nnvm::NodeEntry>& inputs,
      const std::unordered_map<std::string, std::string>& dict) {
    if (CheckGradAllZero(ograds))
      return MakeZeroGradNodes(n, ograds);
    auto p = MakeNode(op_name, n->attrs.name + "_backward",
                      nullptr, &dict, &n);
    p->inputs.insert(p->inputs.end(), ograds.begin(), ograds.end());
    p->inputs.insert(p->inputs.end(), inputs.begin(), inputs.end());
    std::vector<nnvm::NodeEntry> ret;
    for (uint32_t i = 0; i < p->num_outputs(); ++i) {
      ret.emplace_back(p, i, 0);
    }
    return ret;
  }
  ```

  * 其实就是创建一个新的`Node`，其`op`为`op_name`指定，然后就是返回这个`Op`的所有输出`NodeEntry`构成的`vector`
  * 没了
* 对应的`_backward_abs`的实现为

  ``` cpp
  MXNET_OPERATOR_REGISTER_BINARY_WITH_SPARSE_CPU(_backward_abs, unary_bwd<mshadow_op::sign>)
  ```

  * 宏展开后为

    ``` cpp
    #define MXNET_OPERATOR_REGISTER_BINARY_WITH_SPARSE_CPU(__name$, __kernel$)              \
      MXNET_OPERATOR_REGISTER_BINARY(__name$)                                               \
      .set_attr<FInferStorageType>("FInferStorageType",                                     \
        ElemwiseStorageType<2, 1, true, true, true>)                                        \
      .set_attr<FCompute>("FCompute<cpu>", ElemwiseBinaryOp::Compute<cpu, __kernel$>)       \
      .set_attr<FComputeEx>("FComputeEx<cpu>", ElemwiseBinaryOp::ComputeEx<cpu, __kernel$>) \
      .set_attr<FResourceRequest>("FResourceRequest",  /* For Sparse CSR */ \
        [](const NodeAttrs& attrs) { \
          return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};})
    ```

    * 就是一些这些属性的注册的快捷方式
    * 宏`MXNET_OPERATOR_REGISTER_BINARY()`实现为

      ``` cpp
      #define MXNET_OPERATOR_REGISTER_BINARY(name)                        \
        NNVM_REGISTER_OP(name)                                            \
        .set_num_inputs(2)                                                \
        .set_num_outputs(1)                                               \
        .set_attr<nnvm::FListInputNames>("FListInputNames",               \
          [](const NodeAttrs& attrs) {                                    \
            return std::vector<std::string>{"lhs", "rhs"};                \
          })                                                              \
        .set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<2, 1>)  \
        .set_attr<nnvm::FInferType>("FInferType", ElemwiseType<2, 1>)     \
        .set_attr<nnvm::FInplaceOption>("FInplaceOption",                 \
          [](const NodeAttrs& attrs){                                     \
            return std::vector<std::pair<int, int> >{{0, 0}, {1, 0}};     \
          })                                                              \
        .add_argument("lhs", "NDArray-or-Symbol", "first input")          \
        .add_argument("rhs", "NDArray-or-Symbol", "second input")
      ```

      * 其实就是同样注册一些属性，包括`set_num_inputs, set_num_outputs, set_attr<nnvm::FListInputNames>, set_attr<mxnet::FInferShape>`等等注册`Op`时必须的属性

  * 宏`MXNET_OPERATOR_REGISTER_UNARY_WITH_RSP_CSR`的定义与之类似，唯一的区别就是`BINARY, UNARY`的区别

* 所以实际是`mshadow_op::sign()`函数完成计算

  ``` cpp
  struct sign : public mxnet_op::tunable {
    template<typename DType>
    MSHADOW_XINLINE static typename enable_if<!is_unsigned<DType>::value, DType>::type
    Map(DType a) {
      if (a < DType(0)) return DType(-DType(1));
      if (a > DType(0)) return DType(1);
      return DType(0);
    }
    template<typename DType>
    MSHADOW_XINLINE static typename enable_if<is_unsigned<DType>::value, DType>::type
    Map(DType a) {
      if (a > DType(0)) return DType(1);
      return DType(0);
    }
  };
  ```

  * 主要是静态成员函数`Map()`，这个函数的返回值是

    ``` cpp
    typename enable_if<!is_unsigned<DType>::value, DType>::type
    ```

    * `enable_if`模板的实现是

      ``` cpp
      template<bool B, class T = void>
      struct enable_if {};

      template<class T>
      struct enable_if<true, T> { typedef T type; };
      ```

      * 后面是一个便特例化的实现，`enable_if`利用了`SFINAE`，具体介绍另一篇笔记
      * 整体思路就是，如果`!is_unsigned<DType>::value`返回的是`true`的化，那么`type`就是`DType`，否则，就是空？按照`cppreference`的说法

        > This metafunction is a convenient way to leverage [SFINAE](https://en.cppreference.com/w/cpp/language/sfinae) to conditionally remove functions from [overload resolution](https://en.cppreference.com/w/cpp/language/overload_resolution) based on type traits and to provide separate function overloads and specializations for different type traits. **std::enable_if** can be used as an additional function argument (not applicable to operator overloads), as a return type (not applicable to constructors and destructors), or as a class template or function template parameter.

      * 换句话说，如果`enable_if<A, B>`的第一个模板参数`A`是`false`，那么就不定义这个函数；而如果是`true`的话才定义这个函数，并且能够被重载

  * `sign`类其实包含了两个`Map()`函数的定义，并根据`enable_if<!is_unsigned<DType>::value, DType>`等手段进行函数模板的重载，`is_unsigned`是C++提供的模板`struct`

  * `sign`类里面的`Map()`函数实现逻辑是
    * 如果传入的参数`a`大于0，那么返回1，否则返回0，（a 是 `unsigned`的情况）
    * 返回的是`a`的符号，如果`a`大于0，返回1，否则返回-1；如果`a=0`，则返回0
    * 需要注意的是，这里的`a`是`abs()`的输入，因为`abs()`计算后的结果都是正的，没法知道输入到底有没有取反了啊再计算`abs()`的时候
  *  

* 实际应该是`unary_bwd<mshadow_op::sign>`函数实现了`abs()`的梯度计算

  ``` cpp
  template<typename GRAD_OP>
  using unary_bwd = ::mxnet::op::mxnet_op::backward_grad_tuned<GRAD_OP>;
  ```

  * `backward_grad_tuned<GRAD_OP>`的实现如下

    ``` cpp
    template<typename GRAD_OP>
    struct backward_grad_tuned : public backward_grad<GRAD_OP>, public tunable {
      using backward_grad<GRAD_OP>::Map;
    };
    ```

  * `backward_grad<GRAD_OP>`的定义为

    ``` cpp
    template<typename GRAD_OP>
    struct backward_grad {
      /* \brief Backward calc with grad
       * \param a - output grad
       * \param args... - data to grad calculation op (what this is -- input, output, etc. -- varies)
       * \return input grad
       */
      template<typename DType, typename ...Args>
      MSHADOW_XINLINE static DType Map(DType a, Args... args) {
        return DType(a * GRAD_OP::Map(args...));
      }
    };
    ```

  * 所以，这里有实际计算是`a * GRAD_OP::Map(args...)`，这里的`GRAD_OP::Map(args...)`就是`mshadow_op::sign::Map()`函数了

## ElemwiseGradUseOut

这种情况会用到输出数据用于计算梯度，以`relu`为例进行说明。

``` cpp
struct ElemwiseGradUseOut {
  const char *op_name;
  std::vector<nnvm::NodeEntry> operator()(const nnvm::NodePtr& n,
                                          const std::vector<nnvm::NodeEntry>& ograds) const {
    std::vector<nnvm::NodeEntry> heads;
    uint32_t n_out = n->num_outputs();
    for (uint32_t i = 0; i < n_out; ++i) {
      heads.emplace_back(n, i, 0);
    }
    return MakeNonlossGradNode(op_name, n, ograds, heads, n->attrs.dict);
  }
};
```

* `MakeNonlossGradNode()`的实现见上文
* 这里与上面`MakeNonlossGradIn`的区别在于`MakeNonlossGrad()`函数的第三个参数，这里传入的是`heads`，而不是上面的`n->inputs`，而这里的`heads`其实就是根据传入的`n`里面包含的计算的所有输出构成的`vector<nnvm::NodeEntry>`变量了
* 在`relu`的实现中

  ``` cpp
  MXNET_OPERATOR_REGISTER_BINARY_WITH_SPARSE_CPU(_backward_relu, unary_bwd<mshadow_op::relu_grad>)
  ```

* `mshadow_op::relu_grad`的实现为

  ``` cpp
  struct relu_grad : public mxnet_op::tunable {
    template<typename DType>
    MSHADOW_XINLINE static DType Map(DType a) {
      if (isnan_typed::IsNan(a)) {
        return a;
      } else {
        return a > DType(0) ? DType(1) : DType(0);
      }
    }
  };
  ```

  * `Map()`函数的实现逻辑是，如果传入的参数`a`大于0，则返回1，否则返回0。需要注意的是，这里`a`依赖于`relu()`的输出，与上面的`abs()`相反；其实我觉着这里把`a`当作输入也可以，然后使用`ElemwiseGradUseOut`来实现
  * 没了

## ElemwiseGradUseNone

这种情况不会进行梯度计算，以`_copy`为例进行说明。

``` cpp
struct ElemwiseGradUseNone {
  const char *op_name;
  std::vector<nnvm::NodeEntry> operator()(const nnvm::NodePtr& n,
                                          const std::vector<nnvm::NodeEntry>& ograds) const {
    return MakeNonlossGradNode(op_name, n, ograds, {}, n->attrs.dict);
  }
};
```

* 直接调用`MakeNonlossGradNode()`函数，区别在于第三个参数是空的，即`{}`，表示不依赖于输入或者输出数据。体现在代码里面就是新建的`Node* p->inputs`只会`append(ograds)`，其它就没有区别了
* 其它的都是一致的

## 与`ConvolutionOp`的对比

* `Convolution`中使用了`MakeGradNode()`函数
  * 函数声明为

    ``` cpp
    inline std::vector<nnvm::NodeEntry> MakeGradNode(
        const char* op_name, const nnvm::NodePtr& n,
        const std::vector<nnvm::NodeEntry>& inputs,
        const std::unordered_map<std::string, std::string>& dict);
    ```

* 这个文件里面的例子使用了`MakeNonlossGradNode()`函数
  * 函数声明为

    ``` cpp
    inline std::vector<nnvm::NodeEntry> MakeNonlossGradNode(
        const char* op_name, const nnvm::NodePtr& n,
        const std::vector<nnvm::NodeEntry>& ograds,
        const std::vector<nnvm::NodeEntry>& inputs,
        const std::unordered_map<std::string, std::string>& dict);
    ```

  *  

* `MakeGradNode, MakeNonlossGradNode`的区别
  * 后者会多一个参数`ograds`，但在`Convolution`中，传进来的`heads`给`inputs`里面已经包含了`ograds`了
  * 细节见下面单独的一小节
* 此外，还有几个类似的函数

  ``` cpp
  // quick helper to make gradient nodes that simply pass back zero. could be used in output ops.
  inline std::vector<nnvm::NodeEntry> MakeZeroGradNodes(
      const nnvm::NodePtr& n,
      const std::vector<nnvm::NodeEntry>& ograds) {
    std::vector<nnvm::NodeEntry> ret;
    for (uint32_t i = 0; i < n->num_inputs(); ++i) {
      std::ostringstream os;
      if (1 == n->num_inputs()) {
        os << n->attrs.name << "_backward";
      } else {
        os << n->attrs.name << "_in" << i << "_backward";
      }
      ret.emplace_back(MakeNode("zeros_like", os.str(), {n->inputs[i]}, nullptr, &n));
    }
    return ret;
  }
  ```

  * 这个函数没有`op_name`这个参数，也没有使用传入的参数`ograds`，而是直接返回与`n->num_inputs()`相同数量的`zeros_like`为`Node`成员构成的`NodeEntry`向量
  * 其它就没了
  *  

* `CheckGradAllZero()`函数

  ``` cpp
  // check whether all output grads are zero.
  inline bool CheckGradAllZero(const std::vector<nnvm::NodeEntry>& ograds) {
    static const auto zero_op = nnvm::Op::Get("_zeros");
    static const auto zero_like_op = nnvm::Op::Get("zeros_like");
    if (ograds.empty())
      return false;
    for (const auto& grad : ograds) {
      if (!grad.node)
        return false;
      if (grad.node->op() != zero_op && grad.node->op() != zero_like_op )
        return false;
    }
    return true;
  }
  ```

  * 实现逻辑是如果 `ograds`是空，则返回`false`；如果`ograds.node`为空也返回`false`，如果`op`不是`_zeros`或者`zeros_like`则也返回`false`；其它情况才返回`true`

### MakeGradNode, MakeNonlossGradNode区别

``` cpp
// quick helper to make node
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

// ------------------------------------------------------------------
// ------------------------------------------------------------------

// make gradient node that doesn't add to objective.
// i.e. igrads are always zero when ograds are zero.
inline std::vector<nnvm::NodeEntry> MakeNonlossGradNode(
    const char* op_name, const nnvm::NodePtr& n,
    const std::vector<nnvm::NodeEntry>& ograds,
    const std::vector<nnvm::NodeEntry>& inputs,
    const std::unordered_map<std::string, std::string>& dict) {
  if (CheckGradAllZero(ograds))
    return MakeZeroGradNodes(n, ograds);
    // 第三个参数 nullptr 表示创建 p 时的传入的输入，这里是空的
  auto p = MakeNode(op_name, n->attrs.name + "_backward",
                    nullptr, &dict, &n);
  p->inputs.insert(p->inputs.end(), ograds.begin(), ograds.end());      // 先保存更深一层传进来的 输出数据的梯度
  p->inputs.insert(p->inputs.end(), inputs.begin(), inputs.end());      // 在保存这里传入的 inputs 数据

  // 下面就是构造这个op_name 计算的输出构成的 vector<nnvm::NodeEntry>
  std::vector<nnvm::NodeEntry> ret;
  for (uint32_t i = 0; i < p->num_outputs(); ++i) {
    ret.emplace_back(p, i, 0);
  }
  return ret;     // 返回 op_name  的所有输出构成的向量
}
```

* 感觉最大的区别不就是后一个函数多了一个`CheckGradAllZero(ograds)`的过程，此时使用`MakeZeroGradNodes(n, ograds)`来返回，其它的两个函数基本都是一致的啊！所以，这里也就是为什么后者需要单独的把`ograds`单独作为一个参数传入，而不像`Convolution`中把`ograds`直接放在`inputs`里面，然后调用`MakeGradNode()`函数
* 先暂时这样吧
