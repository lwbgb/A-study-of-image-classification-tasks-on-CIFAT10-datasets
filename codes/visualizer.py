# @version : 1.0
# @author  : 李文彬
# @name    : visualizer.py
# @time    : 2024/12/9 22:04

import os

import numpy as np

from train_options import TrainOptions


# 打印和绘图类
class Visualizer(TrainOptions):

    def __init__(self):
        super().__init__()
        self.current_epoch = 0
        self.saved = False
        self.name = "VIT"

        self.log_name = os.path.join(self.checkpoints_dir, self.name, 'loss_log.txt')

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    def display_current_results(self):
        pass

    def plot_current_losses(self, epoch, counter_ratio, losses):
        pass

    def print_current_losses(self, epoch, iters, losses, t_comp, t_data, is_save):
        mean_loss = np.mean(losses)  # 平均 loss
        total_loss = sum(losses)  # 总的 loss
        message = f"训练轮数: {epoch}, 训练样本数量: {iters}, 时间间隔: {t_data:.2f}, 平均每个样本训练用时: {t_comp:.2e}, mean_loss: {mean_loss:.3f}, total_loss: {total_loss:.3f}"
        print(message)

        if is_save:
            with open(self.log_name, "a") as log_file:
                log_file.write('%s\n' % message)  # save the message
