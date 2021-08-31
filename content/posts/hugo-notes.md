---
title: "Hugo Notes"
author: "triloon"
date: 2021-08-30T14:37:40+08:00
draft: false

excerpt_separator: <!--more-->
---

一些hugo blog搭建过程中的记录。<!--more-->

* 添加 table of contents

  参考博客：[How to Add Table Of Contents to a Hugo Blog](https://codingreflections.com/hugo-table-of-contents/)。目前用到的是第一种方式。
  包含两个步骤：
  * create shortcode
  * call the shortcode inside markdown

* 添加代码块并展示行号

  参考博客：[Syntax Highlighting](https://gohugo.io/content-management/syntax-highlighting/)，以及官方文档：[Configure Markup - highlight](https://gohugo.io/getting-started/configuration-markup#highlight)

  文中还给出了不同配置参数的解释、可用的 style 展示等。

* 添加Latex公式支持

  无用的博客，因为已经过时了，hugo不在支持 mmark 这种 content format 了。[KaTex Intergration - 玄冬Wong](https://dawnarc.com/2019/09/hugokatex-intergration/)。而且目前 mathjax 比 katex 支持的更全。

  另一篇博客是: [Config Hugo with Mathjax or Katex](https://rulenuts.netlify.app/post/config-hugo-with-mathjax-or-katex/)

  最终有用的博客是：[MathJax Support](https://www.gohugo.org/doc/tutorials/mathjax/) 以及 [Setting MathJax with Hugo](http://xuchengpeng.com/hexo-blog/2018/07/10/setting-mathjax-with-hugo/) 以及 [在Hugo中使用MathJax](https://note.qidong.name/2018/03/hugo-mathjax/) 以及[hugo使用katex](https://blog.csdn.net/weixin_42109159/article/details/105099962)

* 插入图片

  两个可行的参考是：[how-to-insert-image-in-my-post](https://discourse.gohugo.io/t/solved-how-to-insert-image-in-my-post/1473) 以及 [hugo-conent-management shortcodes](https://gohugo.io/content-management/shortcodes/#figure)

  具体使用：将下载的图片保存到 static 目录，比如按照 post tilte 命名的目录下面（例如，`/User/apple/myblogs/static/imgs/binary-search-tree/img00.png`，`/User/apple/myblogs`为项目根目录，即运行`hugo server`等命令的目录，`/User/apple/myblogs/public`为编译后的结果），然后在.md正文里指定图片地址（例如，`![image caption](/imgs/binary-search-tree/img00.png)`，注意下直接从`/imgs`开始）。

  第二个博客里给出了另外一种加载图片的方式，即使用shortcodes（Markdown语言的补充，方便建立网页。），而且也给出了其他多媒体信息的加载方式，比如vimeo, youtube, gist, tweet, instagram等信息。 

* 添加网页图标

  一个常见的博客是：[Favicon](https://www.enthuseandinspire.co.uk/blog/favicon/)，但是目前来看还没有起作用。但是在 `triloon.space` 上起作用了，后续需要补充上 windows 的格式。
