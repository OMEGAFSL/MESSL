import torch
import torch.nn as nn
import scipy.stats as st
import numpy as np
import os


# 若path文件夹不存在，创建
def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
        print('creat {} successful'.format(path))


# 显示时间
def format_time(duration):
    duration = int(duration)
    s = duration % 60
    m = (duration // 60) % 60
    h = (duration // 3600)
    return "{:d}h{:02d}m{:02d}s".format(h, m, s)


# 计算准确率和置信度
def stats(scores, name):
    if len(scores) == 1:
        low, up = 0., 1.
    elif len(scores) < 30:
        low, up = st.t.interval(0.95, df=len(scores) - 1, loc=np.mean(scores), scale=st.sem(scores))
    else:
        low, up = st.norm.interval(0.95, loc=np.mean(scores), scale=st.sem(scores))
    if name == "":
        return np.mean(scores), up - np.mean(scores)
    else:
        print("{:s} {:.2f} (± {:.2f}) "
              "(conf: [{:.2f}, {:.2f}]) "
              "(worst: {:.2f}, best: {:.2f})".format(name, 100 * np.mean(scores),
                                                     100 * np.std(scores),
                                                     100 * low, 100 * up,
                                                     100 * np.min(scores),
                                                     100 * np.max(scores)))
class ncm_output(nn.Module):
    def __init__(self, indim, outdim):
        super(ncm_output, self).__init__()
        self.linear = nn.Linear(indim, outdim)

    def forward(self, x):
        return -1 * torch.norm(
            x.reshape(x.shape[0], 1, -1) - self.linear.weight.transpose(0, 1).reshape(1, -1, x.shape[1]), dim=2).pow(
            2) - self.linear.bias


def linear(indim, outdim, ncm_loss):
    if ncm_loss:
        return ncm_output(indim, outdim)
    else:
        return nn.Linear(indim, outdim)


def criterion_episodic(features, targets, n_shots, n_ways, temperature):
    feat = features.reshape(n_ways, -1, features.shape[1])
    feat = preprocess(feat, feat)
    means = torch.mean(feat[:, :n_shots], dim=1)
    dists = torch.norm(feat[:, n_shots:].unsqueeze(2) - means.unsqueeze(0).unsqueeze(0), dim=3, p=2).reshape(-1,
                                                                                                             n_ways).pow(
        2)
    return torch.nn.CrossEntropyLoss()(-1 * dists / temperature, targets.reshape(n_ways, -1)[:, n_shots:].reshape(-1))


def sphering(features):
    return features / torch.norm(features, p=2, dim=2, keepdim=True)


def centering(train_features, features):
    return features - train_features.reshape(-1, train_features.shape[2]).mean(dim=0).unsqueeze(0).unsqueeze(0)


def preprocess(train_features, features, preprocessing, elements_train=None):
    if elements_train != None and "M" in preprocessing:
        train_features = torch.cat(
            [train_features[l, torch.arange(elements_train[l]), :] for l in range(len(elements_train))],
            axis=0).unsqueeze(1)

    for i in range(len(preprocessing)):
        if preprocessing[i] == 'R':
            with torch.no_grad():
                train_features = torch.relu(train_features)
            features = torch.relu(features)
        if preprocessing[i] == 'P':
            with torch.no_grad():
                train_features = torch.pow(train_features, 0.5)
            features = torch.pow(features, 0.5)
        if preprocessing[i] == 'E':
            with torch.no_grad():
                train_features = sphering(train_features)
            features = sphering(features)
        if preprocessing[i] == 'M':
            features = centering(train_features, features)
            with torch.no_grad():
                train_features = centering(train_features, train_features)
    return features


def postprocess(runs, postprocessing):
    # runs shape: [100, 5, 16, 640]
    for i in range(len(postprocessing)):
        if postprocessing[i] == 'R':
            runs = torch.relu(runs)
        if postprocessing[i] == 'P':
            runs = torch.pow(runs, 0.5)
        if postprocessing[i] == 'E':
            runs = runs / torch.norm(runs, p=2, dim=3, keepdim=True)
        if postprocessing[i] == 'M':
            runs = runs - runs.reshape(runs.shape[0], -1, runs.shape[-1]).mean(dim=1, keepdim=True).unsqueeze(1)
    return runs


class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.cls = num_classes

    def forward(self, pred, target):
        assert 0 <= self.smoothing < 1
        pred = pred.log_softmax(dim=-1)

        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), 1 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))

