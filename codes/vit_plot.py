# @version : 1.0
# @author  : 李文彬
# @name    : vit_plot.py
# @time    : 2024/12/11 23:11

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from base_model import *

if __name__ == '__main__':
    model = create_model()
    model.setup()

    train_loader, test_loader = model.create_dataset(is_loader=True)

    arr = np.array([i for i in range(1, 6)] + [i for i in range(10, 31, 5)] + [i for i in range(40, 151, 10)])

    writer = SummaryWriter("./logs/test/vit")

    model.module.eval()
    for epoch in range(1, 46):
        model.load_networks(epoch)
        model.num_test = 100
        total_test_loss = 0
        total_test_acc = 0
        for i, data in enumerate(test_loader):
            if i >= model.num_test:
                break
            images, targets = model.set_input(data)
            y_pred = model.test(images)
            results = y_pred.argmax(1)
            acc = (results == targets).sum() / len(targets)
            loss = model.get_losses(y_pred, targets)
            total_test_acc += acc
            total_test_loss += loss

        acc_mean = (total_test_acc / model.num_test)
        loss_mean = (total_test_loss / model.num_test)
        print(f"total_test_acc: {acc_mean:.3f}")
        print(f"total_test_loss: {loss_mean:.3f}")
        writer.add_scalar("VIT_accuracy", acc_mean, epoch)
        writer.add_scalar("VIT_loss", loss_mean, epoch)

    writer.close()
