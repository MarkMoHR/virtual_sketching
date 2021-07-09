# General Virtual Sketching Framework for Vector Line Art - SIGGRAPH 2021

[[论文]](https://esslab.jp/publications/HaoranSIGRAPH2021.pdf) | [[项目主页]](https://markmohr.github.io/virtual_sketching/)

这份代码能用于实现：**线稿矢量化**、**粗糙草图简化**和**自然图像到矢量草图转换**。

<img src='docs/figures/muten.png' height=300><img src='docs/figures/muten-black-full-simplest.gif' height=300>

<img src='docs/figures/rocket.png' height=150><img src='docs/figures/rocket-blue-simplest.gif' height=150>&nbsp;&nbsp;&nbsp;&nbsp;<img src='docs/figures/1390.png' height=150><img src='docs/figures/face-blue-1390-simplest.gif' height=150>

## 目录
- [环境依赖](#环境依赖)
- [使用预训练模型测试](#使用预训练模型测试)
- [重新训练](#重新训练)
- [引用](#引用)

## 环境依赖
 - [Tensorflow](https://www.tensorflow.org/) (1.12.0 <= 版本 <=1.15.0)
 - [opencv](https://opencv.org/) == 3.4.2
 - [pillow](https://pillow.readthedocs.io/en/latest/index.html) == 6.2.0
 - [scipy](https://www.scipy.org/) == 1.5.2
 - [gizeh](https://github.com/Zulko/gizeh) == 0.1.11

## 使用预训练模型测试
### 模型下载与准备

在[这里](https://drive.google.com/drive/folders/1-hi2cl8joZ6oMOp4yvk_hObJGAK6ELHB?usp=sharing)下载模型：
  - `pretrain_clean_line_drawings` (105 MB): 用于线稿矢量化
  - `pretrain_rough_sketches` (105 MB): 用于粗糙草图简化
  - `pretrain_faces` (105 MB): 用于自然图像到矢量草图转换

然后，按照如下结构放置模型：
```
outputs/
    snapshot/
        pretrain_clean_line_drawings/
        pretrain_rough_sketches/
        pretrain_faces/
```

### 测试方法
在`sample_inputs/`文件夹下选择图像，然后根据任务类型运行下面其中一个命令。生成结果会在`outputs/sampling/`目录下看到。

``` python
python3 test_vectorization.py --input muten.png

python3 test_rough_sketch_simplification.py --input rocket.png

python3 test_photograph_to_line.py --input 1390.png
```

**注意!!!** 我们的方法从一个随机挑选的初始位置启动绘制，所以每跑一次测试理论上都会得到一个不同的结果（有可能效果不错，但也可能效果不是很好）。因此，建议做多几次测试来挑选看上去最好的结果。也可以通过设置 `--sample`参数来定义跑一次测试代码同时输出（不同结果）的数量：

``` python
python3 test_vectorization.py --input muten.png --sample 10

python3 test_rough_sketch_simplification.py --input rocket.png --sample 10

python3 test_photograph_to_line.py --input 1390.png --sample 10
```

**如何复现论文展示的结果？** 可以从[这里](https://drive.google.com/drive/folders/1-hi2cl8joZ6oMOp4yvk_hObJGAK6ELHB?usp=sharing)下载论文展示的结果。这些是我们通过若干次测试得到不同输出后挑选的最好的结果。显然，若要复现这些结果，需要使用相同的初始位置启动绘制。

### 其他工具

#### a) 可视化

我们的矢量输出均使用`npz` 文件包存储。运行以下的命令可以得到渲染后的结果以及绘制顺序。可以在`npz` 文件包相同的目录下找到这些可视化结果。
``` python
python3 tools/visualize_drawing.py --file path/to/the/result.npz 
```

#### b) GIF制作

若要看到动态的绘制过程，可以运行以下命令来得到 `gif`。结果在`npz` 文件包相同的目录下。
``` python
python3 tools/gif_making.py --file path/to/the/result.npz 
```


#### c) 转化为SVG

`npz` 文件包中的矢量结果均按照论文里面的公式(1)格式存储。可以运行以下命令行，来将其转化为 `svg` 文件格式。结果在`npz` 文件包相同的目录下。

``` python
python3 tools/svg_conversion.py --file path/to/the/result.npz 
```
  - 转化过程以两种模式实现（设置`--svg_type`参数）：
    - `single` (默认模式): 每个笔划（一根单独的曲线）构成SVG文件中的一个path路径
    - `cluster`: 每个连续曲线（多个笔划）构成SVG文件中的一个path路径

**重要注意事项**

在SVG文件格式中，一个path上的所有线段均只有同一个线宽（*stroke-width*）。然而在我们论文里面，定义一个连续曲线上所有的笔划可以有不同的线宽。同时，对于一个单独的笔划（贝塞尔曲线），定义其线宽从一个端点到另一个端点线性递增或者递减。

因此，上述两个转化方法得到的SVG结果理论上都无法保证跟论文里面的结果在视觉上完全一致。（*假如你在论文里面使用这里转化后的SVG结果进行视觉上的对比，请提及此问题。*）


<br>

## 重新训练

### 训练准备

在[这里](https://drive.google.com/drive/folders/1-hi2cl8joZ6oMOp4yvk_hObJGAK6ELHB?usp=sharing)下载模型：
  - `pretrain_neural_renderer` (40 MB): 预训练好的神经网络渲染器
  - `pretrain_perceptual_model` (691 MB): 预训练好的perceptual model，用于算 raster loss

在[这里](https://drive.google.com/drive/folders/1-hi2cl8joZ6oMOp4yvk_hObJGAK6ELHB?usp=sharing)下载训练数据集：
  - `QuickDraw-clean` (14 MB): 用于线稿矢量化。来自 [QuickDraw](https://github.com/googlecreativelab/quickdraw-dataset)数据集。
  - `QuickDraw-rough` (361 MB): 用于粗糙草图简化。利用[Sketch Simplification](https://github.com/bobbens/sketch_simplification#pencil-drawing-generation)里面的铅笔画图像生成方法合成。
  - `CelebAMask-faces` (370 MB): 用于自然图像到矢量草图转换。使用[CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ)数据集进行处理后得到。

然后，按照如下结构放置数据集：
```
datasets/
    QuickDraw-clean/
    QuickDraw-rough/
    CelebAMask-faces/
outputs/
    snapshot/
        pretrain_neural_renderer/
        pretrain_perceptual_model/
```

### 训练方法

建议使用多GPU进行训练。每个任务，我们均使用2个GPU（每个11 GB）来训练。

``` python
python3 train_vectorization.py

python3 train_rough_photograph.py --data rough

python3 train_rough_photograph.py --data face
```

<br>

## 引用

若使用此代码和模型，请引用本工作，谢谢！

```
@article{mo2021virtualsketching,
  title   = {General Virtual Sketching Framework for Vector Line Art},
  author  = {Mo, Haoran and Simo-Serra, Edgar and Gao, Chengying and Zou, Changqing and Wang, Ruomei},
  journal = {ACM Transactions on Graphics (Proceedings of ACM SIGGRAPH 2021)},
  year    = {2021},
  volume  = {40},
  number  = {4},
  pages   = {51:1--51:14}
}
```

