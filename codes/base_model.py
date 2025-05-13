# @version : 1.0
# @author  : 李文彬
# @name    : base_model.py
# @time    : 2024/12/9 13:30

import os

import torch
from torch.nn import Module, Linear, CrossEntropyLoss, Sequential
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.models import vit_b_16
from torchvision.transforms import v2

from test_options import TestOptions
from train_options import TrainOptions


def create_model():
    model = BaseModel()
    print("model [%s] was created" % model.name)
    return model.to(model.device)


class BaseModel(Module, TrainOptions, TestOptions):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        TrainOptions.__init__(self)
        TestOptions.__init__(self)

        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device(
            'cpu')
        print(f"device:{self.device}")
        self.checkpoints_dir = "checkpoints"
        self.name = "VIT_precessed"
        self.save_dir = os.path.join(self.checkpoints_dir, self.name)  # 将所有检查点保存到 save_dir
        if self.preprocess != 'scale_width':  # 使用 [scale_width]，输入图像可能具有不同的大小，这会损害 cudnn.benchmark 的性能。
            torch.backends.cudnn.benchmark = True
        self.losses = []  # 指定要绘制和保存的训练损失。
        self.loss_function = "linear"
        self.model_name = ""  # 定义我们训练中使用的网络。
        self.visual_names = []  # 指定要显示和保存的图像。
        # 定义和初始化优化器。您可以为每个网络定义一个优化器。如果两个网络同时更新，
        # 则可以使用 itertools.chain 对它们进行分组。有关示例，请参阅 cycle_gan_model.py。
        self.optimizer = None
        self.image_path = ""
        self.metric = 0  # 用于学习率策略 'plateau'

        # VIT
        self.module = vit_b_16(progress=True)
        self.module.heads = Sequential(Linear(768, 10, bias=True))
        # self.module.heads.add_module("add_linear", Linear(in_features=1000, out_features=10, bias=True))

        # ResNet
        # self.module = resnet18(progress=True)
        # self.module.fc = Linear(in_features=512, out_features=10, bias=True)

        # print(self.module)

        # CIFAR10 模型
        # self.module = Sequential(OrderedDict([
        #     ("conv1", Conv2d(3, 32, 5, padding=2)),
        #     ("max_pool1", MaxPool2d(2)),
        #     ("conv2", Conv2d(32, 32, 5, padding=2)),
        #     ("max_pool2", MaxPool2d(2)),
        #     ("conv3", Conv2d(32, 64, 5, padding=2)),
        #     ("max_pool3", MaxPool2d(2)),
        #     ("flatten", Flatten()),
        #     ("dense1", Linear(1024, 64)),
        #     ("dense2", Linear(64, 10)),
        # ]))

    def forward(self, x):
        return self.module(x)

    # 初始化模型训练参数
    def setup(self):
        self.n_epochs = 20
        self.batch_size = 64
        self.print_freq = 10 * self.batch_size
        self.save_latest_freq = 100 * self.batch_size
        self.save_epoch_freq = 1
        self.model_name = "CIFAR10Model"

    # 预处理数据集
    def create_dataset(self, is_loader):
        # 构建转化器
        # 包含数据预处理
        transformer = v2.Compose(
            [v2.Resize((224, 224)), v2.RandomHorizontalFlip(p=0.5),
             v2.ToImage(), v2.ToDtype(torch.float32, True),
             v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        # 不包含数据预处理
        # transformer = v2.Compose(
        #     [v2.Resize((224, 224)),
        #      v2.ToImage(), v2.ToDtype(torch.float32, True)])

        # 读取数据
        self.data_root = "Dataset"
        train_dataset = datasets.CIFAR10(os.path.join(self.data_root, "train"), train=True, transform=transformer,
                                         download=True)
        test_dataset = datasets.CIFAR10(os.path.join(self.data_root, "test"), train=False, transform=transformer,
                                        download=True)
        self.train_size = len(train_dataset)
        self.test_size = len(test_dataset)
        print('The number of training images = %d' % self.train_size)
        print('The number of testing images = %d' % self.test_size)

        if not is_loader:
            return train_dataset, test_dataset
        else:
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=not self.serial_batches)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=not self.serial_batches)
            return train_loader, test_loader

    # 更新学习率
    def update_learning_rate(self, lr):
        self.lr = lr

    # 预处理数据
    def set_input(self, data):
        images, targets = data
        return images.to(self.device), targets.to(self.device)

    # 计算损失函数，优化模型
    def optimize_parameters(self, targets, outputs):
        self.loss_function = CrossEntropyLoss().to(self.device)
        self.optimizer = Adam(self.parameters(), self.lr)
        self.optimizer.zero_grad()

        # 计算损失
        loss = self.loss_function(outputs, targets)
        self.losses.append(loss.item())

        # 优化器优化权重
        loss.backward()
        self.optimizer.step()

    # 获取当前阶段的 loss 值
    def get_current_losses(self, s, e):
        errors_ret = self.losses[s:e:1]
        return errors_ret

    # 根据预测结果和标签计算 loss
    def get_losses(self, yhat, y):
        self.loss_function = CrossEntropyLoss().to(self.device)
        loss = self.loss_function(yhat, y)
        self.losses.append(loss.item())
        return loss

    # 保存模型
    def save_networks(self, epoch):
        if isinstance(self.model_name, str):
            save_filename = '%s_net_%s.pth' % (epoch, self.model_name)
            save_path = os.path.join(self.save_dir, save_filename)
            os.makedirs(self.save_dir, exist_ok=True)
            net = self

            if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                torch.save(net.module.cpu().state_dict(), save_path)
                net.to(self.device)
            else:
                torch.save(net.cpu().state_dict(), save_path)

    # 加载模型
    def load_networks(self, epoch):
        if isinstance(self.model_name, str):
            load_filename = '%s_net_%s.pth' % (epoch, self.model_name)
            load_path = os.path.join(self.save_dir, load_filename)
            net = self
            print('loading the model from %s' % load_path)
            # if you are using PyTorch newer than 0.4 (e.g., built from
            # GitHub source), you can remove str() on self.device
            state_dict = torch.load(load_path, map_location=self.device, weights_only=True)
            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata

            net.module.load_state_dict(state_dict)

    # 测试时的计算函数
    def test(self, inputs):
        with torch.no_grad():
            return self.forward(inputs)


if __name__ == '__main__':
    model = create_model()
    # print(model)
    batch = torch.ones((64, 3, 224, 224)).to(model.device)
    yhat = model.forward(batch)
    print(yhat.shape)
    print()
