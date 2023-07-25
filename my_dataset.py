from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np


# 随机擦除
def random_erase(img, n_holes, length, rate):  # 输入img为PIL图片格式的图片
    if np.random.rand(1)[0] < rate:
        img = np.array(img)
        h = img.shape[0]  # 图片的高
        w = img.shape[1]  # 图片的宽

        n_holes = np.random.randint(n_holes)
        mask = np.ones((h, w), np.float32)  # 32*32w*h的全1矩阵

        for n in range(n_holes):  # n_holes=2,length=4 选择2个区域；每个区域的边长为4
            y = np.random.randint(h)  # 0~31随机选择一个数 y=4
            x = np.random.randint(w)  # 0~31随机选择一个数 x=24

            y1 = np.clip(y - length // 2, 0, h)  # 2,0,32 ->2
            y2 = np.clip(y + length // 2, 0, h)  # 6,0,32 ->6
            x1 = np.clip(x - length // 2, 0, w)  # 24-2,0,32 ->22
            x2 = np.clip(x + length // 2, 0, w)  # 24+2,0,32 ->26

            mask[y1: y2, x1: x2] = 0.  # 将这一小块区域去除
        img[:, :, 0] = img[:, :, 0] * mask
        img[:, :, 1] = img[:, :, 1] * mask
        img[:, :, 2] = img[:, :, 2] * mask
        return Image.fromarray(img)
    else:
        return img


def random_mixup(img, label, mixup_img, mixup_label):  # 输入img和mixup为IMG格式的图片，label和mixup_label为int类型
    img = np.array(img)
    mixup_img = np.array(mixup_img)
    label_onehot = np.zeros(25)
    label_onehot[label] = 1
    mixup_label_onehot = np.zeros(25)
    mixup_label_onehot[mixup_label] = 1

    alpha = 1
    lam = np.random.beta(alpha, alpha)  # 混合比例

    img_new = lam * img + (1 - lam) * mixup_img
    label_new = lam * label_onehot + (1 - lam) * mixup_label_onehot

    return Image.fromarray(np.uint8(img_new)), torch.tensor(np.float32(label_new))


def rand_bbox(size, lam):
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3 or len(size) == 2:
        W = size[0]
        H = size[1]
    else:
        raise Exception

    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix(img ,label, cutmix_img, cutmix_label):
    #int转化为one_hot
    label_onehot = np.zeros(25)
    label_onehot[label] = 1
    cutmix_label_onehot = np.zeros(25)
    cutmix_label_onehot[cutmix_label] = 1

    alpha = 1
    lam = np.random.beta(alpha,alpha)

    bbx1, bby1, bbx2, bby2 = rand_bbox(img.size, lam)

    img_new = img.copy()
    img_new.paste(cutmix_img.crop((bbx1, bby1, bbx2, bby2)),(bbx1, bby1, bbx2, bby2))

    # 计算 1 - bbox占整张图像面积的比例
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img_new.size[0] * img_new.size[1]))
    label_new = lam*label_onehot + (1-lam)*cutmix_label_onehot

    return img_new,torch.tensor(np.float32(label_new))

def label_smooth(label_onehot, epsilon=0.1):
    n_classes = label_onehot.size(0)
    smooth_label_onehot = (1 - epsilon) * label_onehot + epsilon / n_classes
    return smooth_label_onehot

def add_gaussian_noise(img_tensor, mean=0.0, std=1.0, p=0.2):
    noise = torch.randn(img_tensor.size()) * std + mean
    noisy_img_tensor = img_tensor + noise
    return noisy_img_tensor if torch.rand(1) < p else img_tensor


class MyDataSet2(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, label: list, transforms=None, model="train"):
        self.images_path = images_path
        self.label = label
        self.transforms = transforms
        self.model = model
    def __getitem__(self, index):
        img = Image.open(self.images_path[index]).convert('RGB')

        #将label转换为one_hot编码
        label_onehot = np.zeros(25)
        label_onehot[self.label[index]] = 1
        label_onehot = torch.tensor(np.float32(label_onehot))

        if self.model == 'train':
            # 随机擦除 150代表150个正方形，3代表每个正方形边长为3像素，0.2代表20%的概率
            img = random_erase(img,150,3,0.2)
            # mixup,0.2概率
            if np.random.rand(1)[0]<0.2:
                mixup_idx = np.random.randint(0, len(self.images_path) - 1)
                mixup_img = Image.open(self.images_path[mixup_idx]).convert('RGB')
                mixup_label = self.label[mixup_idx]
                img, label_onehot = random_mixup(img, self.label[index], mixup_img, mixup_label)
            #cutmix,0.2概率
            if np.random.rand(1)[0]<0.2:
                cutmix_idx = np.random.randint(0, len(self.images_path) - 1)
                cutmix_img = Image.open(self.images_path[cutmix_idx]).convert('RGB')
                cutmix_label = self.label[cutmix_idx]
                img, label_onehot = cutmix(img, self.label[index], cutmix_img, cutmix_label)
        if self.transforms is not None:
            img = self.transforms(img)

        label_onehot = label_smooth(label_onehot)
        return img, label_onehot
    def __len__(self):
        return len(self.images_path)

class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, label: list, transforms=None):
        self.images_path = images_path
        self.label = label
        self.transforms = transforms

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item]).convert('RGB')
        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.label[item]

        if self.transforms is not None:
            img = self.transforms(img)

        return img, label