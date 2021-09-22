---
title: "Binary Search Tree"
author: "triloon"
date: 2021-08-31T15:00:22+08:00
draft: false

tags : [
    "Data Structure",
    "Tree"
]

mathjax: true

excerpt_separator: <!--more-->
---
二叉搜索树相关笔记<!--more-->

## 定义

每个节点有两个子节点：左节点、右节点，即是一个二叉树；同时有顺序关系：左子树小于父节点，右子树大于父节点。

![binary-search-image structure example](/imgs/binary-search-tree/BSTSearch01.png)

查找复杂度：$\sim 2\ln(N)$，其中 N 是节点个数。查找未命中也是这个复杂度。

作为一种数据结构，主要功能就是：增删查改。floor() 也可以认为是一种“查”操作。

目前来看，相关应用可以分为两大类：遍历，排序。遍历就是包括前序、中序、后序进行遍历然后处理；排序一般就是按照中序进行处理。

## floor() 函数

```cpp {linenos=table, hl_lines=[0], linenostart=0}
Key floor(Key key)
{
    Node x = floor(root, key);
    if (x == nullptr)
        return nullptr;
    else
        return x;
}

Node floor(Node root, Key key)
{
    if (root == nullptr)
        return nullptr;
    
    if (root->key == key)
        return root;
    if (root->key > key)    // left branch
        return floor(root->left, key);
    // right branch: key > root->key
    Node t = floor(root->right, key);
    if (t == nullptr)
        return root;
    else
        return t;
}
```

为什么这里相当于是一个前序排序，其实本质上需要的是 `root->left` 分支的递归要在 `root->right` 的前面，这样才能保证走右子树的时候，可以最先判断这个子树上最小的数值。

这个函数关键是分清楚什么时候向左走，什么时候向右走，然后才能明白应该返回什么数值。向左走的时候，说明当前的 `root->key` 大于目标值，向右走的时候，说明当前的 `root->key` 小于目标值，而小于目标值则说明结果必定存在，因为大不了返回当前的 root 即可。因此，在递归过程向上走的时候，如果当前属于右子路的下一层(代码第19行)并且返回的是 nullptr，那么就返回当前的节点就行了(第20行)，否则返回递归返回的结果即可。总而言之，如果走上右分支，就必定有解；走上左分支，则原样返回即可。

**递归过程可以看成两个步骤：首先是沿着左子树或者右子树向下走，然后就是沿着树向上爬。** 向上爬的时候，需要注意返回结果，对树的状态进行更新，比如链接、具体成员变量等。另一方面，递归过程其实就是一个先进后出的过程，所以对应的非递归实现可以借助Stack来实现。向上走的时候基于向下走的时候的判断来保证正确。

## Delete()函数

首先是删除最小、最大值节点。

主要在于递归返回结果，**只需要将返回的链接赋给作为参数的链接**。

```cpp {linenos=table, hl_lines=[0], linenostart=0}
// ...
x.left = deleteMin(x.left);     // delete minimum
// ...
```

其次是删除中间某个节点，与删除最小、最大节点的区别在于，这个节点包含两个字节点，而且还需要保持二叉搜索树之间的顺序。这个问题常用的解法是：`Hibbard算法`。该算法表示在删除节点 x 之后，用它的后继节点填充它的位置，并且这个后继节点就是右子树中的最小节点。

对应的4个步骤如下：

1. 找到需要被删除的节点，用 t 表示
2. 然后找到被删除节点右（左，随机选择）分支的最小（大）值，表示为 x，即 `x = min(t.right)`
3. 然后更新 x 节点的左右分支，`x.right` 更新为 `deleteMin(t.right)`，保证右子树都大于 `x`
4. `x.left` 更新为 `t.left`，最后返回 x 作为原来 t 的父节点的子节点，即代替 t

实现示例如下：

```cpp {linenos=table, hl_lines=[0], linenostart=0}
Node delete(Key key)
{
    delete(root, key);
}

Node delete(Node t, key key)
{
    if (t == nullptr)       // not exists
        return nullptr;
    if (key < t->key)       // go left
    {
        t->left = delete(t->left, key);
        return t;
    }
    elif (key > t->key)
    {
        t->right = delete(t->right, key);
        return t;
    }
    else                    // equal
    {
        if (t.right == nullptr)
            return t.left;
        if (t.left == nullptr)
            return t.right;
        Node x = min(t.right);
        x.right = deleteMin(t.right);
        x.left = t.left;
        delete t;
        return x;
    }
    // other update
}
```

## 其他支持的函数

* Select()

  即返回第k小的节点。

  可以通过中序进行遍历，然后每次递归返回的时候，通过引用更新计数，当计数为0的时侯返回该节点即可。

* Rank()

  即给定一个key，返回该key在树中的位置。

  如果给定 key 小于当前节点，则正确的位置必定在右分支中，则最终的排序就是左分支的节点个数 + 1（当前节点） + 在右子树中的排序，而在右子树中的排序直接递归即可。

* 范围查找

  典型的中序遍历思路，即判断当前节点是否在制定的范围内，如果在范围内，则保存当前节点到一个队列即可。
