# @version : 1.0
# @author  : 李文彬
# @name    : resnet_plot.py
# @time    : 2024/12/11 22:30

from torch.utils.tensorboard import SummaryWriter

from base_model import *

if __name__ == '__main__':
    model = create_model()
    model.setup()

    train_loader, test_loader = model.create_dataset(is_loader=True)

    writer = SummaryWriter("./logs/test/resnet")
    for epoch in range(5, 35):
        model.load_networks(epoch)
        model.num_test = 100  # 控制训练的样本数量
        total_test_loss = 0
        total_test_acc = 0
        for i, data in enumerate(test_loader):
            if i >= model.num_test:
                break
            images, targets = model.set_input(data)
            writer.add_graph(model, input_to_model=images)  # 绘制框架图
            y_pred = model.test(images)  # 预测结果
            # 计算损失和准确率
            results = y_pred.argmax(1)
            acc = (results == targets).sum() / len(targets)
            loss = model.get_losses(y_pred, targets)
            total_test_acc += acc
            total_test_loss += loss

        acc_mean = (total_test_acc / model.num_test)
        loss_mean = (total_test_loss / model.num_test)
        print(f"total_test_acc: {acc_mean:.3f}")
        print(f"total_test_loss: {loss_mean:.3f}")
        writer.add_scalar("ResNet_accuracy", acc_mean, epoch)
        writer.add_scalar("ResNet_loss", loss_mean, epoch)

    writer.close()
