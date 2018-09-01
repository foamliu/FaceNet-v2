# FaceNet

这是 FaceNet 的Keras实现 [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832).

与 [FaceNet](https://github.com/foamliu/FaceNet) 主要差别：
1. 输入图片尺寸 139 -> 224。
2. 训练数据集 CelebA -> VGGFACE2。
3. 标定模型 Dlib -> [MTCNN](https://github.com/davidsandberg/facenet/tree/master/src/align).


## 软件兼容性
- Python 3.5
- Tensorflow 1.6.0
- Keras 2.2.0


## 数据集

VGGFACE2 是一个大型的人脸数据集，有9,000多个人物身份和330多万张人脸图像。

![image](https://github.com/foamliu/FaceNet-v2/raw/master/images/vggface2.png)

按照 [说明](http://zeus.robots.ox.ac.uk/vgg_face2/) 下载 VGGFACE2 数据集 vggface2_train.tar.gz 和 vggface2_test.tar.gz 放入 data 目录。

## 模型结构
![image](https://github.com/foamliu/FaceNet-v2/raw/master/images/model.png)

## 工作流程
处理单个输入图像的工作流程如下：

1. 人脸检测：使用 Dlib 中预先训练的模型检测面部。
2. 人脸校准：使用 Dlib 的实时姿势估计与 OpenCV 的仿射变换来尝试使眼睛和下唇在每个图像上出现在相同位置。
3. 卷积网络：使用深度神经网络把人脸图片映射为 128 维单位超球面上的一个点。

![image](https://github.com/foamliu/FaceNet-v2/raw/master/images/summary.jpg)
[图片来源](https://cmusatyalab.github.io/openface/)

## 性能评估

使用 Labeled Faces in the Wild (LFW) 数据集做性能评估:

- 13233 人脸图片
- 5749 人物身份
- 1680 人有两张以上照片

### 准备数据
下载 [LFW database](http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz) 放在 data 目录下:

```bash
$ wget http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz
$ tar -xvf lfw-funneled.tgz
$ wget http://vis-www.cs.umass.edu/lfw/pairs.txt
$ wget http://vis-www.cs.umass.edu/lfw/people.txt
```

### 评估脚本
```bash
$ python lfw_eval.py
```

### 测得结果
准确度: **89.27 %**.

## 如何使用
### 数据预处理
提取训练图像:
```bash
$ python pre-process.py

执行下面命令标定人脸：
```bash
$ python align/align_dataset_mtcnn.py data/train data/train_aligned --image_size 200 --margin 24
$ python align/align_dataset_mtcnn.py data/test data/test_aligned --image_size 200 --margin 24
```

### 训练
```bash
$ python train.py
```

要想可视化训练过程，执行下面命令：
```bash
$ tensorboard --logdir path_to_current_dir/logs
```

### DEMO

```bash
$ python demo.py
```

正(P) | 欧式距离 | 锚(A) | 欧式距离 | 反(N) |
|---|---|---|---|---|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/0_p_image.png)|0.3579|![image](https://github.com/foamliu/FaceNet/raw/master/images/0_a_image.png)|0.9037|![image](https://github.com/foamliu/FaceNet/raw/master/images/0_n_image.png)|
|1.2757|---|1.0052|---|1.0740|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/1_p_image.png)|0.7775|![image](https://github.com/foamliu/FaceNet/raw/master/images/1_a_image.png)|0.7634|![image](https://github.com/foamliu/FaceNet/raw/master/images/1_n_image.png)|
|1.7359|---|1.3784|---|0.7606|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/2_p_image.png)|0.3556|![image](https://github.com/foamliu/FaceNet/raw/master/images/2_a_image.png)|1.3154|![image](https://github.com/foamliu/FaceNet/raw/master/images/2_n_image.png)|
|1.1428|---|0.9498|---|0.7804|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/3_p_image.png)|0.5836|![image](https://github.com/foamliu/FaceNet/raw/master/images/3_a_image.png)|1.8520|![image](https://github.com/foamliu/FaceNet/raw/master/images/3_n_image.png)|
|2.0298|---|1.9031|---|1.9189|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/4_p_image.png)|0.3343|![image](https://github.com/foamliu/FaceNet/raw/master/images/4_a_image.png)|1.7794|![image](https://github.com/foamliu/FaceNet/raw/master/images/4_n_image.png)|
|1.3242|---|1.5383|---|0.6461|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/5_p_image.png)|0.2490|![image](https://github.com/foamliu/FaceNet/raw/master/images/5_a_image.png)|1.0756|![image](https://github.com/foamliu/FaceNet/raw/master/images/5_n_image.png)|
|1.2015|---|1.4692|---|0.8207|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/6_p_image.png)|0.3323|![image](https://github.com/foamliu/FaceNet/raw/master/images/6_a_image.png)|0.9058|![image](https://github.com/foamliu/FaceNet/raw/master/images/6_n_image.png)|
|1.1681|---|1.1821|---|0.5887|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/7_p_image.png)|0.6769|![image](https://github.com/foamliu/FaceNet/raw/master/images/7_a_image.png)|0.8985|![image](https://github.com/foamliu/FaceNet/raw/master/images/7_n_image.png)|
|1.1974|---|0.7921|---|1.3920|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/8_p_image.png)|0.3638|![image](https://github.com/foamliu/FaceNet/raw/master/images/8_a_image.png)|1.4623|![image](https://github.com/foamliu/FaceNet/raw/master/images/8_n_image.png)|
|0.8634|---|0.8968|---|0.5992|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/9_p_image.png)|0.3681|![image](https://github.com/foamliu/FaceNet/raw/master/images/9_a_image.png)|1.3735|![image](https://github.com/foamliu/FaceNet/raw/master/images/9_n_image.png)|


## 附录

### 样本数据
执行下面命令查看样本数据：
```bash
$ python data_generator.py
```
正(P) | 锚(A) | 反(N) |
|---|---|---|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/sample_p_0.jpg)|![image](https://github.com/foamliu/FaceNet/raw/master/images/sample_a_0.jpg)|![image](https://github.com/foamliu/FaceNet/raw/master/images/sample_n_0.jpg)|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/sample_p_1.jpg)|![image](https://github.com/foamliu/FaceNet/raw/master/images/sample_a_1.jpg)|![image](https://github.com/foamliu/FaceNet/raw/master/images/sample_n_1.jpg)|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/sample_p_2.jpg)|![image](https://github.com/foamliu/FaceNet/raw/master/images/sample_a_2.jpg)|![image](https://github.com/foamliu/FaceNet/raw/master/images/sample_n_2.jpg)|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/sample_p_3.jpg)|![image](https://github.com/foamliu/FaceNet/raw/master/images/sample_a_3.jpg)|![image](https://github.com/foamliu/FaceNet/raw/master/images/sample_n_3.jpg)|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/sample_p_4.jpg)|![image](https://github.com/foamliu/FaceNet/raw/master/images/sample_a_4.jpg)|![image](https://github.com/foamliu/FaceNet/raw/master/images/sample_n_4.jpg)|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/sample_p_5.jpg)|![image](https://github.com/foamliu/FaceNet/raw/master/images/sample_a_5.jpg)|![image](https://github.com/foamliu/FaceNet/raw/master/images/sample_n_5.jpg)|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/sample_p_6.jpg)|![image](https://github.com/foamliu/FaceNet/raw/master/images/sample_a_6.jpg)|![image](https://github.com/foamliu/FaceNet/raw/master/images/sample_n_6.jpg)|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/sample_p_7.jpg)|![image](https://github.com/foamliu/FaceNet/raw/master/images/sample_a_7.jpg)|![image](https://github.com/foamliu/FaceNet/raw/master/images/sample_n_7.jpg)|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/sample_p_8.jpg)|![image](https://github.com/foamliu/FaceNet/raw/master/images/sample_a_8.jpg)|![image](https://github.com/foamliu/FaceNet/raw/master/images/sample_n_8.jpg)|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/sample_p_9.jpg)|![image](https://github.com/foamliu/FaceNet/raw/master/images/sample_a_9.jpg)|![image](https://github.com/foamliu/FaceNet/raw/master/images/sample_n_9.jpg)|


### 数据增强
执行下面命令查看数据增强效果：
```bash
$ python data_generator.py
```
之前 | 之后 | 之前 | 之后 | 之前 | 之后 |
|---|---|---|---|---|---|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_before_0.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_after_0.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_before_1.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_after_1.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_before_2.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_after_2.png)|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_before_3.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_after_3.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_before_4.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_after_4.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_before_5.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_after_5.png)|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_before_6.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_after_6.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_before_7.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_after_7.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_before_8.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_after_8.png)|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_before_9.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_after_9.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_before_10.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_after_10.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_before_11.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_after_11.png)|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_before_12.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_after_12.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_before_13.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_after_13.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_before_14.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_after_14.png)|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_before_15.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_after_15.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_before_16.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_after_16.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_before_17.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_after_17.png)|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_before_18.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_after_18.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_before_19.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_after_19.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_before_20.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_after_20.png)|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_before_21.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_after_21.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_before_22.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_after_22.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_before_23.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_after_23.png)|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_before_24.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_after_24.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_before_25.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_after_25.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_before_26.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_after_26.png)|
|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_before_27.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_after_27.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_before_28.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_after_28.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_before_29.png)|![image](https://github.com/foamliu/FaceNet/raw/master/images/imgaug_after_29.png)|
