from collections import OrderedDict
from collections import namedtuple
from itertools import product


# 在CNN.py文件中使用该方法
class RunBuilder():
    @staticmethod
    def get_runs(params):
        # 定义一个namedtuple类型Run，并包含params中的lr和batch_size属性
        Run = namedtuple('Run', params.keys())

        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs


params = OrderedDict(
    lr=[.01, .001],
    batch_size=[1000, 10000]
)
print(params.keys(), params.values())
Run = namedtuple('Run', params.keys())
print(Run)
print(RunBuilder.get_runs(params)[0].lr)