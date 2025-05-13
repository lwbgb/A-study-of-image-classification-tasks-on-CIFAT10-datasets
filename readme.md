## Codes

|        程序        | 说明                                                         |
| :----------------: | :----------------------------------------------------------- |
|  `base_model.py`   | 构建的模型基类，包含模型的构建和模型相关参数和方法           |
| `train_options.py` | 设置模型训练相关参数                                         |
|     `train.py`     | 模型训练程序                                                 |
|     `test.py`      | 模型测试程序                                                 |
| `test_options.py`  | 设置模型测试相关参数                                         |
|   `visulizer.py`   | 包含一些模型信息打印和绘制的方法                             |
|  `resnet_plot.py`  | ResNet 模型准确率随训练轮数变化图绘制                        |
|   `vit_plot.py`    | ViT 模型准确率随训练轮数变化图绘制                           |
|     `temp.py`      | 用于绘制预处理后的图像与原图像的对比，案例图 `bee.jpg`，对比情况请见论文 |

> 本实验框架采用 Pytorch，IDE 软件采用 Pycharm，图像绘制使用 TensorBoard
>
> CIFAR10 数据集链接：*[https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html).* 
>
> 整个项目的架构参考 CycleGAN 的源码实现，链接：*https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/tree/master*

---

## Pictures

|          文件           | 说明                             |
| :---------------------: | :------------------------------- |
| `trian_loss_resnet.svg` | ResNet 训练过程中的 loss         |
|  `trian_loss_vit.svg`   | ViT 训练过程中的 loss            |
|  `resnet_accuracy.svg`  | 训练过程中 ResNet 准确率变化情况 |
|   `VIT_accuracy.svg`    | 训练过程中 ViT 准确率变化情况    |
|    `resnet_loss.svg`    | 训练过程中 ResNet loss 变化情况  |
|     `VIT_loss.svg`      | 训练过程中 ViT loss 变化情况     |

> 以上图片都由 TensorBoard 绘制并保存
