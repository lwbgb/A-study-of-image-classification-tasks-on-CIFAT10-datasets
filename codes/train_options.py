# @version : 1.0
# @author  : 李文彬
# @name    : train_options.py
# @time    : 2024/12/9 22:22

# 训练参数
class TrainOptions:

    def __init__(self):
        super().__init__()

        # 基本参数
        self.data_root = "Dataset"
        self.name = "Image_Classification"  # '实验的名称，它决定将样本和模型的存储位置
        self.gpu_ids = '0'  # gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU
        self.checkpoints_dir = "checkpoints"
        # 模型参数
        self.model = None
        # 数据集参数
        self.dataset_mode = "unaligned"
        self.serial_batches =  False
        self.num_threads = 4
        self.train_size = 0
        self.test_size = 0
        self.batch_size = 1
        self.load_size = 286
        self.crop_size = 256
        self.max_dataset_size =  float("inf")
        self.preprocess = "none"
        # 其他参数
        self.epoch = "latest"
        self.load_iter = 0

        # 显示参数
        self.display_freq = 400
        self.print_freq = 100
        self.display_id = 1  # Web 显示器的窗口 ID
        # 网络保存和加载参数
        self.save_latest_freq = 1000
        self.save_epoch_freq = 5
        self.save_by_iter = False
        self.epoch_count = 1  # 起始 epoch 计数，我们通过以下方式保存模型 <epoch_count>, <epoch_count>+<save_latest_freq>, ...
        self.phase = "train"
        # 训练参数
        self.n_epochs = 100  # 具有初始学习率的 epoch 数
        self.n_epochs_decay = 0  # 将学习率线性衰减为零的纪元数
        self.beta1 = 0.5
        self.lr = 1e-3
        self.pool_size = 50
        self.lr_policy = "linear"
        self.lr_decay_iters = 50

        self.isTrain = True



