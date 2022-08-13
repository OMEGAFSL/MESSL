import numpy as np
import torch
import pandas as pd
from PIL import Image
import os
from transform_cfg import transforms_options


# 遥感图像数据读取函数
def data_loader(path):
    return Image.open(path).convert('RGB').resize((84, 84))


class CPUDataset():
    def __init__(self, data, targets, batch_size, transforms=[], use_hd=False):
        self.data = data
        if torch.is_tensor(data):
            self.length = data.shape[0]
        else:
            self.length = len(self.data)
        self.targets = targets
        assert (self.length == targets.shape[0])
        self.batch_size = batch_size
        self.transforms = transforms
        self.use_hd = use_hd

    def __getitem__(self, idx):
        if self.use_hd:
            elt = data_loader(self.data[idx])
        else:
            elt = self.data[idx]
        return self.transforms(elt), self.targets[idx]

    def __len__(self):
        return self.length


class EpisodicCPUDataset():
    def __init__(self, args, data, num_classes, transforms=[], use_hd=False):
        self.data = data
        if torch.is_tensor(data):
            self.length = data.shape[0]
        else:
            self.length = len(self.data)
        self.episode_size = (args.episode_size // args.n_ways) * args.n_ways
        self.episodes_per_epoch = args.episodes_per_epoch
        self.n_ways = args.n_ways

        self.transforms = transforms
        self.use_hd = use_hd
        self.num_classes = num_classes
        self.targets = []
        self.indices = []
        self.corrected_length = args.episodes_per_epoch * self.episode_size
        episodes = args.episodes_per_epoch
        for i in range(episodes):
            classes = np.random.permutation(np.arange(self.num_classes))[:args.n_ways]
            for c in range(args.n_ways):
                class_indices = np.random.permutation(np.arange(self.length // self.num_classes))[
                                :self.episode_size // args.n_ways]
                self.indices += list(class_indices + classes[c] * (self.length // self.num_classes))
                self.targets += [c] * (self.episode_size // args.n_ways)
        self.indices = np.array(self.indices)
        self.targets = np.array(self.targets)

    def generate_next_episode(self, idx):
        if idx >= self.episodes_per_epoch:
            idx = 0
        classes = np.random.permutation(np.arange(self.num_classes))[:self.n_ways]
        n_samples = (self.episode_size // self.n_ways)
        for c in range(self.n_ways):
            class_indices = np.random.permutation(np.arange(self.length // self.num_classes))[
                            :self.episode_size // self.n_ways]
            self.indices[idx * self.episode_size + c * n_samples: idx * self.episode_size + (c + 1) * n_samples] = (
                        class_indices + classes[c] * (self.length // self.num_classes))

    def __getitem__(self, idx):
        if idx % self.episode_size == 0:
            self.generate_next_episode((idx // self.episode_size) + 1)
        if self.use_hd:
            elt = data_loader(self.data[self.indices[idx]])
        else:
            elt = self.data[self.indices[idx]]
        return self.transforms(elt), self.targets[idx]

    def __len__(self):
        return self.corrected_length


class Dataset():
    def __init__(self, data, targets, batch_size, transforms=[], shuffle=True, device='gpu'):
        if torch.is_tensor(data):
            self.length = data.shape[0]
            self.data = data.to(device)
        else:
            self.length = len(self.data)
        self.targets = targets.to(device)
        assert (self.length == targets.shape[0])
        self.batch_size = batch_size
        self.transforms = transforms
        self.permutation = torch.arange(self.length)
        self.n_batches = self.length // self.batch_size + (0 if self.length % self.batch_size == 0 else 1)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            self.permutation = torch.randperm(self.length)
        for i in range(self.n_batches):
            if torch.is_tensor(self.data):
                yield self.transforms(self.data[self.permutation[i * self.batch_size: (i + 1) * self.batch_size]]), \
                      self.targets[self.permutation[i * self.batch_size: (i + 1) * self.batch_size]]
            else:
                yield torch.stack([self.transforms(self.data[x]) for x in
                                   self.permutation[i * self.batch_size: (i + 1) * self.batch_size]]), self.targets[
                          self.permutation[i * self.batch_size: (i + 1) * self.batch_size]]

    def __len__(self):
        return self.n_batches


class EpisodicDataset():
    def __init__(self, args, data, num_classes, transforms=[], use_hd=False):
        if torch.is_tensor(data):
            self.length = data.shape[0]
            self.data = data.to(args.dataset_device)
        else:
            self.data = data
            self.length = len(self.data)
        self.episode_size = args.batch_size
        self.transforms = transforms
        self.num_classes = num_classes
        self.n_batches = args.episodes_per_epoch
        self.use_hd = use_hd
        self.device = args.dataset_device
        self.n_ways = args.n_ways

    def __iter__(self):
        for i in range(self.n_batches):
            classes = np.random.permutation(np.arange(self.num_classes))[:self.n_ways]
            indices = []
            for c in range(self.n_ways):
                class_indices = np.random.permutation(np.arange(self.length // self.num_classes))[
                                :self.episode_size // self.n_ways]
                indices += list(class_indices + classes[c] * (self.length // self.num_classes))
            targets = torch.repeat_interleave(torch.arange(self.n_ways), self.episode_size // self.n_ways).to(
                self.device)
            if torch.is_tensor(self.data):
                yield self.transforms(self.data[indices]), targets
            else:
                if self.use_hd:
                    yield torch.stack(
                        [self.transforms(data_loader(self.data[x]).to(self.device)) for x in indices]), targets
                else:
                    yield torch.stack([self.transforms(self.data[x].to(self.device)) for x in indices]), targets

    def __len__(self):
        return self.n_batches


def iterator(args, data, target, transforms, forcecpu=False, shuffle=True, use_hd=False):
    if args.dataset_device == "cpu" or forcecpu:
        dataset = CPUDataset(data, target, args.batch_size, transforms, use_hd=use_hd)
        return torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                           shuffle=shuffle)  # , num_workers = min(8, os.cpu_count()))
    else:
        return Dataset(data, target, args.batch_size, transforms,
                       shuffle=shuffle, device=args.dataset_device)


def episodic_iterator(args, data, num_classes, transforms, forcecpu=False, use_hd=False):
    if args.dataset_device == "cpu" or forcecpu:
        dataset = EpisodicCPUDataset(args, data, num_classes, transforms, use_hd=use_hd)
        return torch.utils.data.DataLoader(dataset,
                                           batch_size=(args.batch_size // args.n_ways) * args.n_ways,
                                           shuffle=False)  # , num_workers = min(8, os.cpu_count()))
    else:
        return EpisodicDataset(args, data, num_classes, transforms, use_hd=use_hd)


# 遥感数据集类别信息，train，val，test，num_per
cls_list = {
    'NWPU45': (25, 10, 10, 700),
    'WHURS19': (9, 5, 5, 50),  # 这个数据集是每类至少50张
    'UCM': (10, 5, 6, 100)
}


def RSDataset(args, data_name, train_transforms, all_transforms, use_hd=True):
    datasets = {}
    for subset in ["train", "val", "test"]:
        if data_name == "WHURS19":  # WHURS19使用归一化数量后的数据集
            f_csv = pd.read_csv(os.path.join(args.dataset_path, data_name, '{}_count50.csv'.format(subset)))
        else:
            f_csv = pd.read_csv(os.path.join(args.dataset_path, data_name, '{}.csv'.format(subset)))
        imgs = list(f_csv.loc[0])[1::]  # 得到数据路径
        labels = list(f_csv.loc[1])[1::]  # 得到数据标签
        labels = list(map(int, labels))  # 标签转换为数字

        if not use_hd:
            loader = lambda x: data_loader(x)
            imgs = list(map(loader, imgs))  # 读取数据
        datasets[subset] = [imgs, torch.LongTensor(labels)]

    train_loader = iterator(args, datasets["train"][0], datasets["train"][1], transforms=train_transforms,
                            forcecpu=True, use_hd=use_hd)
    train_clean = iterator(args, datasets["train"][0], datasets["train"][1], transforms=all_transforms, forcecpu=True,
                           shuffle=False, use_hd=use_hd)
    val_loader = iterator(args, datasets["val"][0], datasets["val"][1], transforms=all_transforms, forcecpu=True,
                          shuffle=False, use_hd=use_hd)
    test_loader = iterator(args, datasets["test"][0], datasets["test"][1], transforms=all_transforms, forcecpu=True,
                           shuffle=False, use_hd=use_hd)
    return (train_loader, train_clean, val_loader, test_loader), [3, 84, 84], cls_list[data_name], True, False


def get_dataset(args):
    if args.dataset == "NWPU45":
        return RSDataset(args, args.dataset, transforms_options['N'][0], transforms_options['N'][1])
    elif args.dataset == "UCM":
        return RSDataset(args, args.dataset, transforms_options['U'][0], transforms_options['U'][1])
    elif args.dataset == "WHURS19":
        return RSDataset(args, args.dataset, transforms_options['W'][0], transforms_options['W'][1])
    else:
        print("Unknown dataset!")


