# 问题日志
## 001 在启动visdom服务时遇到如下情况，长时间无反应（2022年4月12日）
***
```powershell
Microsoft Windows [版本 10.0.18363.657]
(c) 2019 Microsoft Corporation。保留所有权利。

C:\Users\UserName>python -m visdom.server
D:\Anaconda3\lib\site-packages\visdom\server.py:39: DeprecationWarning: zmq.eventloop.ioloop is deprecated in pyzmq 17. pyzmq now works with default tornado and asyncio eventloops.
  ioloop.install()  # Needs to happen before any tornado imports!
Checking for scripts.
Downloading scripts, this may take a little while
```

### 解决办法
***
 1. 找到visdom模块安装位置
 	其位置为python或anaconda安装目录下`\Lib\site-packages\visdon`
  ```
  ├─static
│  ├─css
│  ├─fonts
│  └─js
├─__pycache__
├─__init__.py
├─__init__.pyi
├─py.typed
├─server.py
└─VERSION
  ```
 	可在python或anaconda安装目录下搜索找到
 2. 修改文件`server.py`
 修改函数`download_scripts_and_run`，将`download_scripts()`注释掉
 该函数位于全篇末尾，1917行


```python
def download_scripts_and_run():
    # download_scripts()
    main()


if __name__ == "__main__":
    download_scripts_and_run()
```

 3. 替换文件
    将resource中的visdom_static-master文件重命名并覆盖`\visdon\static`文件夹。


至此，该问题解决完毕。
使用命令`python -m visdom.server`开启服务，但是**画图无法自动更新**。
这里我通过谷歌浏览器自动刷新页面插件实现。

```powershell
Microsoft Windows [版本 10.0.18363.657]
(c) 2019 Microsoft Corporation。保留所有权利。

C:\Users\UserName>python -m visdom.server
D:\Anaconda3\lib\site-packages\visdom\server.py:39: DeprecationWarning: zmq.eventloop.ioloop is deprecated in pyzmq 17. pyzmq now works with default tornado and asyncio eventloops.
  ioloop.install()  # Needs to happen before any tornado imports!
It's Alive!
INFO:root:Application Started
You can navigate to http://localhost:8097
```

**一个终端保持服务开启，另外的终端就可以开始训练了。**

### 相关链接
https://blog.csdn.net/didi_ya/article/details/108364679


## 002 VsCode中插件pylint报错但运行没问题(2022年4月13日)
***
注意：这是为了解决强迫症而做的，代码运行没问题，但 pylint 就是在一个地方下面画红色波浪线，看着很不爽；而不是代码真的有问题而画的红线

pylint 提示 xx 模块没有方法/属性
```
(module) torch
The torch package contains data structures for multi-dimensional tensors and defines mathematical operations over these tensors. Additionally, it provides many utilities for efficient serializing of Tensors and arbitrary types, and other useful utilities.

It has a CUDA counterpart, that enables you to run your tensor computations on an NVIDIA GPU with compute capability >= 3.0.

Module 'torch' has no 'eq' member; maybe 'e'?pylint(no-member)
```

pylint 提示 xx 变量/属性 not callable
```
(constant) MODEL: LeNet5
MODEL is not callablepylint(not-callable)
```

### 解决方法
***
pylint 提示 xx 模块没有方法/属性

打开“settings.json”文件。在大括号中，添加一行：
```
"python.linting.pylintArgs": [
    "--generated-members=torch.*"
]
```
pylint是vscode的python语法检查器，pylint是静态检查，在用第三方库的时候有些成员只有在运行代码的时候才会被建立，它就找不到成员，在设置（settings.json）里添加。

这里一定不要改成 
```"python.linting.pylintArgs": ["--generate-members"],```， 不然会忽略所有的错误提示。
或者添加```"python.linting.pylintArgs": ["--errors-only"],```，不然所有warning也会消失。

***
pylint 提示 xx 变量/属性 not callable

解决方案：在错误行后面添加 ```# pylint:disable=E1102```，使 pylint 忽略这行的 not callable 警告。

这里一定不要改成 
```"python.linting.pylintArgs": ["--disable-msg=not-callable"]```， 不然会忽略所有的not-callable错误提示。

### 相关链接
https://blog.csdn.net/m0_49450715/article/details/123606344


## 其他常见问题
***
visdom相关

[下载失败,ERROR:tornado.general,打开蓝屏无导航条等问题](https://blog.csdn.net/Weary_PJ/article/details/122529587)

[启用visdom.server缓慢和空白蓝屏无导航栏](https://blog.csdn.net/qq_43280818/article/details/104241744)

[visdom不能自动更新-不能解决](https://www.csdn.net/tags/NtjaMg5sNjA0MDctYmxvZwO0O0OO0O0O.html)