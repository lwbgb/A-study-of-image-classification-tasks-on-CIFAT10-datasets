# @version : 1.0
# @author  : 李文彬
# @name    : test_options.py
# @time    : 2024/12/10 20:44

# 测试参数
class TestOptions:

    def __init__(self):
        super().__init__()

        self.results_dir = "./results/"
        self.aspect_ratio = 1.0
        self.phase = "test"
        self.eval = True  # use eval mode during test time.
        self.num_test = 50  # how many test images to run
        self.isTrain = False

