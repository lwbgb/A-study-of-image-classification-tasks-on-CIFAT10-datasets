# @version : 1.0
# @author  : 李文彬
# @name    : temp.py
# @time    : 2024/12/12 01:00
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

from base_model import *

if __name__ == '__main__':
    model = create_model()
    model.setup()

    writer = SummaryWriter("./logs/test/temp")

    # 读取图片案例并转化为向量
    img = Image.open("./bee.jpg")
    img_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    img = img_tensor(img)

    writer.add_image("origin", img, 1)

    # 对图片进行预处理
    transformer = v2.Compose(
        [v2.Resize((224, 224)), v2.RandomHorizontalFlip(p=0.5),
         v2.ToImage(), v2.ToDtype(torch.float32, True),
         v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    img_transformed = transformer(img)

    writer.add_image("transformed", img_transformed, 1)

    writer.close()
