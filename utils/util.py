import shutil
import os
import json
from collections import OrderedDict

import cv2
import numpy as np
import sys
import platform
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets

sys.path.append('../')

from PIL import Image


# source: source file path
# target:target file path
def copy(source, target):
    if not os.path.exists(source):
        raise RuntimeError('source file does not exists!')
    if os.path.exists(target):
        raise RuntimeError('target file has existed!')
    shutil.copyfile(source, target)


def move(source, target):
    if not os.path.exists(source):
        raise RuntimeError('source file does not exists!')
    if os.path.exists(target):
        raise RuntimeError('target file has existed!')
    shutil.move(source, target)


# center_crop image
def center_crop(path, new_width, new_height):
    image = Image
    width, height = image.size

    # resize to (224,224) directly if the new height or new width is larger(i.e. enlarge not crop)
    if width < new_width or height < new_height:
        print(path)
        return image.resize((new_width, new_height))

    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2

    return image.crop((left, top, right, bottom))


# del all file
def clear(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.mkdir(path)


# write json to txt file
def write(dic, path):
    with open(path, 'w+') as f:
        f.write(json.dumps(dic))


# read from txt file and transfer to json
def read(path):
    with open(path, 'r') as f:
        result = json.loads(f.read())
    return result


def save_list(lst, path):
    f = open(path, 'w')
    for i in lst:
        f.write((str)(i))
        f.write('\n')
    f.close()


def set_prefix(prefix, name):
    if not os.path.isdir(prefix):
        os.mkdir(prefix)
    if platform.system() == 'Windows':
        name = name.split('\\')[-1]
    else:
        name = name.split('/')[-1]
    shutil.copy(name, os.path.join(prefix, name))


def to_variable(x, has_gpu, requires_grad=False):
    if has_gpu:
        x = Variable(x.cuda(), requires_grad=requires_grad)
    else:
        x = Variable(x, requires_grad=requires_grad)
    return x


def get_parent_diectory(name, num):
    """
    return the parent directory
    :param name: __file__
    :param num: parent num
    :return: path
    """
    root = os.path.dirname(name)
    for i in range(num):
        root = os.path.dirname(root)
    return root


def read_single_image(path, mean=None, std=None):
    # image.shape=(h, w, c)
    image = cv2.imread(path)
    # BGR -> RGB    hwc [0,255]=>[0, 1]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transforms.ToTensor()(image)
    if mean is not None and std is not None:
        for t, m, s in zip(image, mean, std):
            t.sub_(m).div_(s)

    return Variable(image.unsqueeze(0))


def write_list(lst, path):
    if not isinstance(lst, list):
        raise TypeError('parameter lst must be list.')
    with open(path, 'w') as file:
        file.write(str(lst))


def read_list(path):
    with open(path, 'r') as file:
        lst = eval(file.readline())
    return lst


def to_np(x):
    return x.data.cpu().numpy()


def add_prefix(prefix, path):
    return os.path.join(prefix, path)


def weight_to_cpu(path, is_load_on_cpu=True):
    if is_load_on_cpu:
        weights = torch.load(path, map_location=lambda storage, loc: storage)
    else:
        weights = torch.load(path)
    new_state_dict = OrderedDict()
    for k, v in weights.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v

    return new_state_dict


def merge_dict(dic1, dic2):
    merge = dic1.copy()
    merge.update(dic2)
    return merge


def to_image_type(x):
    x = torch.squeeze(x)
    x = to_np(x)

    x = np.transpose(x, (1, 2, 0))
    x = np.clip(x * 255, 0, 255).astype(np.uint8)
    return x


def rgb2gray(rgb):

    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


def get_mean_and_std(path, transform, channels=3):
    dataset = datasets.ImageFolder(root=path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    mean = torch.zeros(channels)
    std = torch.zeros(channels)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(channels):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    mean, std = mean.numpy().tolist(), std.numpy().tolist()
    return [round(x, 4) for x in mean], [round(y, 4) for y in std]


if __name__ == '__main__':
    # tensor = Variable(torch.randn((1, 3, 224, 224)))
    # to_image_type(tensor)
    # data_dir = '../data/diabetic_without_boundry/train'
    # data_dir = '../data/mnist/train'
    # data_dir = '../data/xray_all/train'
    data_dir = '../data/data_augu/train'
    print(get_mean_and_std(path=data_dir,
                           channels=3,
                           transform=transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor()])))
