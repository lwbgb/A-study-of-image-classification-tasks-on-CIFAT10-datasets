# @version : 1.0
# @author  : 李文彬
# @name    : train.py
# @time    : 2024/12/9 13:32

import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from base_model import *
from visualizer import Visualizer

# 创建模型
# opt = TrainOptions()
model = create_model()
model.setup()  # 初始化模型
model.epoch_count = 3
model.load_networks(2)

# 读取数据
# train_dateset, test_dataset = model.create_dataset(is_loader=False)
train_loader, test_loader = model.create_dataset(is_loader=True)

# 训练模型
total_iters = 0  # 总迭代次数
visualizer = Visualizer()
model.train()  # 标记模型开始训练
model.n_epochs = 150
# writer = SummaryWriter("./logs/train")
for epoch in range(model.epoch_count, model.n_epochs + model.n_epochs_decay + 1):  # epoch 从 1 开始
    epoch_start_time = time.time()  # timer for entire epoch
    iter_data_time = time.time()  # timer for data loading per iteration
    epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
    visualizer.reset()  # reset the visualizer: make sure it saves the results to HTML at least once every epoch
    model.update_learning_rate(1e-4)  # update learning rates in the beginning of every epoch.

    t_data = -1  # 默认为负数，如果输出时间为负数表示第一次迭代
    for i, data in enumerate(train_loader):  # i 表示训练数据被分成的 batch 个数
        iter_start_time = time.time()  # timer for computation per iteration
        if total_iters % model.print_freq == 0:
            t_data = iter_start_time - iter_data_time  # 进入当前 epoch 所经历的时间

        total_iters += model.batch_size  # 总迭代次数
        epoch_iter += model.batch_size  # 当前 epoch 迭代次数

        images, targets = model.set_input(data)  # unpack data from dataset and apply preprocessing
        outputs = model.forward(images)  # 模型计算输出结果
        model.optimize_parameters(targets, outputs)  # calculate loss functions, get gradients, update network weights

        # 展示
        # if total_iters % model.display_freq == 0:  # display images on visdom and save images to a HTML file
        #     save_result = total_iters % model.update_html_freq == 0
        #     model.compute_visuals()
        #     visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

        # 打印
        if total_iters % model.print_freq == 0:  # print training losses and save logging information to the disk
            losses = model.get_current_losses(int((total_iters - model.print_freq) / model.batch_size),
                                              int(total_iters / model.batch_size))  # 截取单个 print_freq 区间的 losses
            t_comp = (time.time() - iter_start_time) / model.batch_size  # 每次迭代所用的时间
            visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data, is_save=False)
            # writer.add_scalar("trian_loss", torch.Tensor(np.mean(losses).reshape((1, 1))), total_iters / model.batch_size)
            if model.display_id > 0:
                visualizer.plot_current_losses(epoch, float(epoch_iter) / model.train_size, losses)

        # 根据迭代次数保存模型
        if total_iters % model.save_latest_freq == 0:  # cache our latest model every <save_latest_freq> iterations
            print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
            save_suffix = 'iter_%d' % total_iters if model.save_by_iter else 'latest'
            model.save_networks(save_suffix)

    # 根据 epoch 保存模型
    iter_data_time = time.time()
    if epoch % model.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
        model.save_networks('latest')
        model.save_networks(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' % (
    epoch, model.n_epochs + model.n_epochs_decay, time.time() - epoch_start_time))

# writer.close()
