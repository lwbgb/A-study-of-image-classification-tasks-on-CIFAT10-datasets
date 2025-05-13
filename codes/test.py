# @version : 1.0
# @author  : 李文彬
# @name    : test.py
# @time    : 2024/12/10 12:04

from torch.utils.tensorboard import SummaryWriter
from base_model import *

if __name__ == '__main__':
    model = create_model()
    model.setup()
    model.load_networks(4)
    # print(model)

    train_loader, test_loader = model.create_dataset(is_loader=True)

    model.num_test = 100
    # model.eval()
    total_test_loss = 0
    # shutil.rmtree("./logs/test")  # 清除日志
    writer = SummaryWriter("./logs/test")
    total_acc = 0
    for i, data in enumerate(test_loader):
        if i >= model.num_test:
            break
        images, targets = model.set_input(data)
        # writer.add_graph(model, input_to_model=images)
        y_pred = model.test(images)
        results = y_pred.argmax(1)
        acc = (results == targets).sum() / len(targets)
        total_acc += acc
        writer.add_scalar("test_acc", acc, i)
        print(f"iter: {i}, acc: {acc:.3f}")

        loss = model.get_losses(y_pred, targets)
        writer.add_scalar("test_loss", loss, i)
        print(f"iter: {i}, loss: {loss:.3f}")

    print(f"total_test_acc: {(total_acc / model.num_test):.3f}")
    writer.close()
