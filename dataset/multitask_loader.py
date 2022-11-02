# taken from https://github.com/AaronGrainer/pytorch-nlp-multitask/blob/master/trainer/task.py
import numpy as np


class StrIgnoreDevice(str):
    """
    This is a hack. The Trainer is going call .to(device) on every input
    value, but we need to pass in an additional `task_name` string.
    This prevents it from throwing an error
    """

    def to(self, device):
        return self


class DataLoaderWithTaskname:
    """
    Wrapper around a DataLoader to also yield a task name
    """

    def __init__(self, task_name, data_loader):
        self.task_name = task_name
        self.data_loader = data_loader

        self.batch_size = data_loader.batch_size
        self.dataset = data_loader.dataset

    def __len__(self):
        return len(self.data_loader)

    def __iter__(self):
        for batch in self.data_loader:
            yield [StrIgnoreDevice(self.task_name)] + batch


class MultitaskDataloader:
    """
    Data loader that combines and samples from multiple single-task
    data loaders.
    """

    def __init__(self, dataloader_dict, sample_ratios=None):
        self.dataloader_dict = dataloader_dict
        self.sample_ratios = [x/sum(sample_ratios) for x in sample_ratios]
        self.num_batches_dict = {
            task_name: len(dataloader)
            for task_name, dataloader in self.dataloader_dict.items()
        }
        self.max_num_batches = max(self.num_batches_dict.values())
        self.task_name_list = list(self.dataloader_dict)
        self.dataset = [None] * sum(
            len(dataloader.dataset) for dataloader in self.dataloader_dict.values()
        )

        self.iters = {task_name: iter(dataloader) for task_name, dataloader in self.dataloader_dict.items()}

    def __len__(self):
        return sum(self.num_batches_dict.values())

    def __iter__(self):
        """
        Sample randomly a task with equal probability. When the dataloader reaches its end, reset it."""

        for _ in range(len(self)):
            task_name = np.random.choice(self.task_name_list, p=self.sample_ratios)
            try:
                yield next(self.iters[task_name])
            except StopIteration:
                self.iters[task_name] = iter(self.dataloader_dict[task_name])
                yield next(self.iters[task_name])
