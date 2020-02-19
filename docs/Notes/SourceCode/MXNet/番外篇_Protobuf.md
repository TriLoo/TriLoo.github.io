# MXNet源码分析番外篇 - Protobuf

smh

2019.09.12

## 简介

参考：

[Protobuf Developer Guide](https://developers.google.com/protocol-buffers/docs/overview)

[Protobuf Basics C++](https://developers.google.com/protocol-buffers/docs/cpptutorial)

...



## Guide

*  主要目的

  > serializing structured data for use in communications protocols, data storage, and more.

*  message中的数据类型

  *  Numbers: integer or floating-point
  *  booleans
  *  strings
  *  raw bytes
  *  Other protobuf message types
  *  

*  

## C++ Basics

### .proto文件

*  `package`对应的是`namespace(C++)`

*  `message` 对应的是一个聚合类，里面的内容是`typed fileds`，每个`type`可以是下面几种类型

  *  `bool, int32, float, double, string`
  *  `enum`结构
  * `other message types`
  *  还可以在一个`message`里面**定义**其它`message`
  *  

*  `=1, =2 ...`等标记

  *  被`fields`用于二进制编码中

  * `1-15`之间的小数标记的`fields`会比大于等于16的标记的`fields`在二进制编码中少一个`byte`。这容易理解，因为`1-15`之间的数用4位二进制就可以表示了，而大于15的数都需要至少8位二进制才行，（这里假设是4的倍数），但到底是不是这个意思，还不确定

  *  所以可以把那些经常用到的`fields`或重复的`repeated`的`fields`等

  * `repeated`修饰的`fields`中，这些`tag`都会被重复编码，所以这是这些小`tag`最好的使用场景

    >  Each element in a repeated field requires re-encoding the tag number, so repeated fields are particularly good candidates for this optimization.

*  每个`field`必须被下面三个修饰词中的一个所修饰

  *  `required`
    *  修饰的`field`必须提供值，否则就报`unitialized`的错误
    *  除了上面这个区别之外，`required`与`optional`没有其它区别了
    *  
  *  `optional`
    * 对应的`field`可被设置或不被设置
    *   
  *  `repeated`
    *  可以把`repeated`修饰的`fields`比作长度可变的`array`来理解
    *  
  *  建议：**尽量不使用required，因为貌似坏处大于好处。**
  *  

*  

### C++ API

*  protobuf
*  io
* util
*  compiler
*  

### C++ Generated Code





## Language Guide

### Defining A message Type

*  `.proto`文件的语法声明（`syntax = "proto3"`）第一个非空、非注释行

*  `fields`就是`name/value pairs`，

  > each field has a name and a type

*  Specifying field types

  *  可以是`enumerations`或其它的`message types`
  *  

*  Assigning Field Numbers

  *  如果是`1-15`之间的数字，就只用1个bytes同时编码`field number ^ field type`
  *  `16-2047`之间的数字用2个`bytes`
  *  可以使用的数字范围最小时1，最大时$2^{29}. - 1$ ，此外`19000 - 19999`之间的数字是保留数字，也不能使用
  *  

*  message fileds可以是下面两种类型中的一种

  *  `singular`
    *  `proto3`中的默认类型，可以有零个或最多一个对应的`fields`
  *  `repeated`
    *  这种`fields`可以重复存在多次，包括零次，顺序也会被保留
    * 在`proto3`中`repeated fields of scalar numeric types`使用`packed`编码方式
    *  
  *  一个`.proto`文件中可以定义多个`message`类型

*  `adding comments`

  *  注释与C/C++注释相同
    *  `//`
    * `/* ... */`
  *  

*  `reserved fields`

  *  如果自己对之前存在的`message`注释掉了或删掉了，并且在新增的`message`中使用了这些原先被使用的`field number`，那么在**旧**的`.proto`文件时，就会引起严重的`data corruption, privacy bugs ...`。坚决办法就是把所有之前使用但现在删掉的`fields`标记成`reserved`

  *  如果后面有人想使用之前使用但现在删除的`fields number`时，编译器就会报错

    ```protobuf
    message Foo{
    	reserved 2, 15, 9, to 11;  // field numbers & field nmaes 必须在不同的reserved 语句中
    	reserved "foo", "bar";
    }
    ```

*  what's Generated From Your `.proto`

  *  `getting & setting field values`
  *  `serializing your messages to an output stream`
  *  `parsing your messages from an input stream`
  *  对于`C++`，产生一个`.h, .cc`文件，每个`Message`对应一个`class`
  *  

*  

### Scalar Value Types

`Scalar Message`可以有下面的类型，对应的是`.proto`类别与之对应的`C++`中的类型。

*  `int32`使用变长的编码，对于负数效率较低；如果使用负数的话，使用`sint32`类型
*  `sint32`，即`signed int`，对于负数，比`int32s`编码更高效
*  `fixed32`，总是使用4个字节进行编码
*  `float`
* `double`
* `bool`
*  `string`
* `bytesss`，可以包含任意长度的bytes数据，最长$2^32$，在`C++`中对应`string`类型
*  所有的类型，在使用`setting values`时，都会进行类型检查

### Default Values

*  如果`scalar message fileds`被设置成默认值，那么这个`value`就不会被序列化
*  

### Enumerations

当希望某个`fields`只能具有之前预设的值的时候，就可以定义一个`enumeration`

*  每个`enum`的第一个`element`必须映射到0

*  设置`option allow_alias = True`，此时才可以同一个`enum`里面包含多个`enum constants`到同一个`same value`

*  可以在`message`内部或外部定义`enum`，这两种方式定义的`enum`都可以在`.proto`的其它地方被使用

*  在反序列化时，如果遇到不认识的`enum values`，也会被保留，而且在C++中，取出的数值也是这个不认识的数的原样返回

*  `Reserved Values`

  * 在`enum`中同样使用`reserved`关键字避免后面使用之前存在但现在被删了的`numeric values and/or names`

    ```protobuf
    enum Foo{
    	reserved 2, 15,  9 to 11, 40 to max;
    	reserved "FOO", "BAR";
    }
    ```

    其中可以使用`max`来表示可能的最大值。

*  

### Other Message Types

*  Importing Definitions

  *  在`.proto`文件的顶部加入一句话

    ```protobuf
    import other_protos.proto;
    ```

    就可以直接import 进来其它`.proto`文件中的定义的`message`类型了。

  *  可以使用`import public`来转移对定义的依赖，也就是说在一个文件里面使用`import public other.proto;`来是的第三个文件可以用`import `当前文件来使用`other.proto`中的定义。

  *  

*  

### Nested Types

在一个`message`内部定义其它`message`类型。

*  在外面，使用`Parent.Type`来使用这种方式定义的`message`
*  可以嵌套任意层次或深度

### Updating A Message Type

当代码不更新，但`message`内容更新时，是可以的，而且还可以使用原来的代码。但需要注意下面几个规则：

*  不要修改所有已经存在的`fields`的`field numbers`
*  如果增加新的`field`，之前生成的`serialized messages`可以被新的代码使用，只是会返回默认值；当然，旧的代码也可以使用新生成的`message`
*  `int32, uint32, int64, uint64. bool`之间的类型更新不会影响代码的前向、后向兼容性
*  `sint32, sint64`之间相互兼容，但与上文提到的几个类型之间不存在兼容
*  `string, bytes`之间只要`bytes`是`UTF-8`，那么就是相互兼容的
*  `enum`与`int32, uint32, int64, uint64`之间是相互兼容的
*  

### Unknown fields

> Unknown fields are well-formed protocol buffer serialized data representing fields that the parser does not recognize.

*  在3.5版本之后，如果遇到不认识的`fields`，也会输出解析的结果
*  在之前，都会忽略这些不认识的`fields`
*  

### Any

> The Any message type lets you use messages as embedded types without having their `.proto` definition. An Any contains an arbitrary serialized message as `bytes`, along with a URL that acts as a globally unique identifier for and resolves to taht message's type.

* 需要`import goolge/protobuf/any.proto`;
* 使用`PackFrom(), UnpackTo()`来赋值、读取`Any`类型的参数
*   

### Oneof

*  使用方法：在关键字`oneof`后面加上一个`oneof name`，就是下面例子中的`test_oneof`

  ```protobuf
  message SampleMessage{
  	oneof test_oneof{
  		string name = 4;
  		SubMessage sub_message = 9;
  	}
  }
  ```

  

*  可以添加任意`fields of any type`，除了`repeated fields`

*  在使用`protoc`生成的代码中，`oneof fields`与其它普通`fields`具有相同的`getters, setters`

*  此外还有一个特别的函数`checking which value in the oneof is set`

*  

*  Oneof Features

  *  当设置oneof fields中的任意一个成员时，其它成员会被自动清除

    ```protobuf
    SampleMessage message;
    message.set_name("name");
    CHECK(message.has_name());
    message.mutable_sub_message();  // will clear name field
    CHECK(!message.has_name());
    ```

  *  如果解析器遇到同一个`oneof`的多个`members`时，只有最后遇到的那个是有效的，也就是`oneof`定义中的所有`fields`只有一个是有效的，这就是`oneof`名字的来源啊

  *  `oneof`不能是`repeated`

  *  `oneof`类型同样支持 `Reflection APIs`

  *  `oneof`中存在`fields`被设置成默认值，那么`case of that oneof field`就会被设置，并且这个值也会被序列化，我猜这个`case`就是`oneof`中的成员

  *  `C++`中，需要保证代码不会出现`memory crashes`，下面的代码就会引起`memory crash`，因为上面提到的`feature`，调用`set_anme("name")`之后`sub_message`就已经被清楚了

    ```protobuf
    SampleMessage message;
    SubMessage* sub_message = message.mutable_sub_message();
    message.set_name("name"); // will delete sub_message
    sub_message->set_...      // Crashed here
    ```

  *  同样的在`c++`中，使用`Swap()`之后，两个`oneof`的`case`就会互换。下面例子中，`msg1`会有`sub_message`，`msg2`会有`name`

  *  向后兼容性的问题

    > Be careful when adding or removing oneof fields. If checking the value of a oneof returns `None`/`NOT_SET`, it could mean that the oneof has not been set or it has been set to a field in a different version of the oneof. There is no way to tell the difference, since there's no way to know if an unknown field on the wire is a member of the oneof.

  *  

### Maps

*  Protobuf 提供了定义`associative map`的定义语法

  ```protobuf
  map<key_type, value_type> map_field = N;
  ```

  *  其中，`key_type`可以是`integral or string type`
  *  `value_type`可以是除了`map`之外的任意类型
  *  

*  `map fields`不能是`repeated`

*  `map`的序列化顺序是没定义行为的

*  当从`.proto`产生`text format`时，maps按照`key`进行排序

*  如果存在重复的`key`，只有最后一个是有效的

*  如果没有提供对应的`value`，在`C++`中返回默认值

*  Backwards compatibility

  *  `map`等价于下面的代码

    ```protobuf
    message MapFieldEntry{
    	key_type key = 1;
    	value_type value = 2;
    }
    repeated MapFieldEntry map_field = N;
    ```

  *  

### Packages









## Encoding

