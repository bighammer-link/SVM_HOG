# 使用介绍：

#### 			本项目使用SVM+HOG特征来实现螺钉和螺母的分类。

🎂食用方法：

​		1.安装`requirements.txt`中的依赖

​		2.进行训练，将`test.py`中的尾部代码`do_training('images/train/pos', 30, 'images/train/neg', 30)`取消注释，			运行`test.py `即可。训练结束后，训练所得权重会保存在weights.txt中。

​		3.测试，运行`UI.py`，点击**选择图片**，可以选择本文件夹`images\test`文件夹下的图片，也可以从网络自行下载测试图片。点击 **进行检测**。

👀数据集介绍：

​		数据集采用手机拍摄螺钉，螺母照片，分为正负样本。**正样本为螺钉，负样本为螺帽。**

❤友情提示：

​		本项目中weights.txt，是我训练好的权重，可以直接使用。

🐱‍👤运行结果：

![image](https://raw.githubusercontent.com/bighammer-link/My_Pictures/myblog/q.jpg)
