import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from IPython.display import display, clear_output
import pandas as pd
import time
import json

from itertools import product
from collections import namedtuple
from collections import OrderedDict


class RunManager:

    # 构造函数
    def __init__(self):
        self.epoch_count = 0
        self.epoch_loss = 0
        self.epoch_num_correct = 0
        self.epoch_start_time = None

        self.run_params = None
        self.run_count = 0
        self.run_data = []
        self.run_start_time = None

        self.network = None
        self.loader = None
        self.tb = None

    # 每一次run（对应不同的超参）
    def begin_run(self, run, network, loader):
        self.run_start_time = time.time()
        self.run_count += 1
        self.run_params = run  # run参数来自我们定义的RunBuilder类中

        self.network = network
        self.loader = loader
        self.tb = SummaryWriter(comment=f'-{run}')

        images, labels = next(iter(self.loader))
        grid = torchvision.utils.make_grid(images)

        self.tb.add_image('images', grid)
        # 若指定了device则用指定的device，若未指定，则用‘cpu’作为默认值
        self.tb.add_graph(self.network, images.to(getattr(run, 'device', 'cpu')))

    # 一个run结束后关闭tensorboard且epoch清零
    def end_run(self):
        self.tb.close()
        self.epoch_count = 0

    # 周期开始时的参数初始化
    def begin_epoch(self):
        self.epoch_start_time = time.time()
        self.epoch_loss = 0
        # self.epoch_count = 0
        self.epoch_count += 1
        self.epoch_num_correct = 0

    # 周期结束后重新计算参数，并通过并在tensorboard中添加数据保存到disk
    def end_epoch(self):
        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time

        loss = self.epoch_loss / len(self.loader.dataset)
        accuracy = self.epoch_num_correct / len(self.loader.dataset)

        self.tb.add_scalar('Loss', loss, self.epoch_count)
        self.tb.add_scalar('Accuracy', accuracy, self.epoch_count)

        for name, param in self.network.named_parameters():
            self.tb.add_histogram(name, param, self.epoch_count)
            self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch_count)

        # 将训练过程中的这些参数全部保存下来
        results = OrderedDict()
        results["run"] = self.run_count
        results["epoch"] = self.epoch_count
        results["accuracy"] = accuracy
        results["epoch duration"] = epoch_duration
        results["run duration"] = run_duration
        for k, v in self.run_params._asdict().items():
            results[k] = v
        self.run_data.append(results)
        # 以pandas数据格式来保存，实现格式化输出
        df = pd.DataFrame.from_dict(self.run_data, orient='columns')
        clear_output(wait=True)
        display(df)

    # 追踪损失
    def track_loss(self, loss):
        self.epoch_loss += loss.item() * self.loader.batch_size

    # 最终正确个数
    def track_num_correct(self, preds, labels):
        self.epoch_num_correct += self._get_num_correct(preds, labels)

    @staticmethod
    def _get_num_correct(preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()

    # Save方法将run_data保存为json与csv格式
    def save(self, fileName):
        pd.DataFrame.from_dict(
            self.run_data,
            orient='columns'
        ).to_csv(f'{fileName}.csv')

        with open(f'{fileName}.json', 'w', encoding='utf-8') as f:
            json.dump(self.run_data, f, ensure_ascii=False, indent=4)

