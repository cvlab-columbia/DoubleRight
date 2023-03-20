
import os
import sys
import glob
import json
import socket
from shutil import copyfile
import tqdm
import shutil


def getDictImageNetClasses(path_imagenet_classes_name='data/imagenet_classes.txt'):
    '''
    Returns dictionary of classname --> classid. Eg - {n02119789: 'kit_fox'}
    '''

    count = 0
    dict_imagenet_classname2id = {}
    with open(path_imagenet_classes_name) as f:
        line = f.readline()
        while line:
            split_name = line.strip().split()
            cat_name = split_name[2]
            id = split_name[0]
            # if cat_name in dict_imagenet_classname2id.keys():
                # print(cat_name)
            dict_imagenet_classname2id[id] = cat_name.lower()
            count += 1
            # print(cat_name, id)
            line = f.readline()
    # print("Total categories categories", count)
    return dict_imagenet_classname2id


import shutil
import os
import pickle
import torch
import numpy as np
from torchvision.datasets import CIFAR10
from typing import Any, Callable, Optional, Tuple
from PIL import Image


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()


def refine_classname(class_names):
    for i, class_name in enumerate(class_names):
        class_names[i] = class_name.lower().replace('_', ' ').replace('-', ' ').replace('/', ' ')
    return class_names


def save_checkpoint(state, args, is_best=False, filename='checkpoint.pth.tar'):
    savefile = os.path.join(args.model_folder, filename)
    bestfile = os.path.join(args.model_folder, 'model_best.pth.tar')
    torch.save(state, savefile)
    if is_best:
        shutil.copyfile(savefile, bestfile)
        print('saved best file')


def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr

    return _lr_adjuster


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def load_imagenet_folder2name(path):
    dict_imagenet_folder2name = {}
    with open(path) as f:
        line = f.readline()
        while line:
            split_name = line.strip().split()
            cat_name = split_name[2]
            id = split_name[0]
            dict_imagenet_folder2name[id] = cat_name
            line = f.readline()
    # print(dict_imagenet_folder2name)
    return dict_imagenet_folder2name


def load_imagenet_names(path='data/imagenet_classes.txt'):
    all_names= [ ]
    with open(path) as f:
        line = f.readline()
        while line:
            split_name = line.strip().split()
            cat_name = split_name[2]
            all_names.append(cat_name)
            line = f.readline()

    return all_names

class CLASS_SPLIT_CIFAR100(CIFAR10):
    base_folder = "cifar-100-python"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list = [
        ["train", "16019d7e3df5f24257cddd939b257f8d"],
    ]

    test_list = [
        ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
    ]
    meta = {
        "filename": "meta",
        "key": "fine_label_names",
        "md5": "7973b15100ade9c7d40fb424638fde48",
    }

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            train_class_count=50,
            load_train_classes=True,
            mix_all_data=False,
            random_seed=0,  # TODO: random class splitting?
    ) -> None:

        super(CIFAR10, self).__init__(root, transform=transform, target_transform=target_transform)
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        if mix_all_data:
            downloaded_list = self.train_list + self.test_list
        else:
            downloaded_list = self.train_list if train else self.test_list

        self.data: Any = []
        self.targets = []
        self._load_meta()

        class_idx = list(range(len(self.classes)))
        train_classes = set(class_idx[:train_class_count])
        test_classes = set(class_idx[train_class_count:])
        if load_train_classes:
            desired_classes = train_classes
        else:
            desired_classes = test_classes

        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
            labels = entry["fine_labels"]
            imgs = entry["data"]
            for img, label in zip(imgs, labels):
                if label in desired_classes:
                    self.data.append(img)
                    self.targets.append(label)

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC


def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.
    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.
    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    return y[labels]


def get_attributes(des):
    cat = []
    ans = []
    for each in des.keys():
        # ans.append(each)
        cat.append(each)
        for com in des[each]:
            ans.append(com)
    return ans, cat













