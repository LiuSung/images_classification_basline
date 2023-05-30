# -*- coding: utf-8 -*-
import glob
import pandas as pd
import numpy as np

import time
from PIL import Image

# from efficientnet_pytorch import EfficientNet
# model = EfficientNet.from_pretrained('efficientnet-b4')

import torch

torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import os
import shutil

class QRDataset(Dataset):
    def __init__(self, train_jpg, transform=None):
        self.train_jpg = train_jpg
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        start_time = time.time()
        img = Image.open(self.train_jpg[index]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, torch.from_numpy(np.array(0))

    def __len__(self):
        return len(self.train_jpg)


class SVHN_Model2(nn.Module):
    def __init__(self):
        super(SVHN_Model2, self).__init__()

        # resnet18
        model_conv = models.resnet50(pretrained=True)
        fc_infeatures = model_conv.fc.in_features
        model_conv.avgpool = nn.AdaptiveAvgPool2d(1)
        model_conv = nn.Sequential(*list(model_conv.children())[:-1])  # 去除最后一个fc layer
        self.cnn = model_conv

        self.hd_fc1 = nn.Linear(fc_infeatures, 128)
        self.dropout_1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128, 9)

    def forward(self, img):
        feat = self.cnn(img)
        feat = feat.view(feat.shape[0], -1)
        feat1 = self.hd_fc1(feat)
        feat1 = self.dropout_1(feat1)
        c1 = self.fc1(feat1)
        return c1


def predict(test_loader, model, tta=1):
    # switch to evaluate mode
    model.eval()

    test_pred_tta = None
    for _ in range(tta):
        test_pred = []
        with torch.no_grad():
            for i, (input, target) in enumerate(test_loader):
                input = input.cuda()
                output = model(input)
                output = output.data.cpu().numpy()

                test_pred.append(output)
        test_pred = np.vstack(test_pred)

        if test_pred_tta is None:
            test_pred_tta = test_pred
        else:
            test_pred_tta += test_pred

    return test_pred_tta


test_jpg = glob.glob('test/test/*.jpg')
test_jpg = np.array(test_jpg)

if __name__ == '__main__':
    ## 预测
    test_pred = None
    for model_path in ['resnet50_fold_dict3.pt']:

        test_loader = torch.utils.data.DataLoader(
            QRDataset(test_jpg,
                    transforms.Compose([
                        transforms.Resize((384, 384)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
                    ), batch_size=10, shuffle=False, num_workers=10, pin_memory=True
        )

        model = SVHN_Model2().cuda()
        model.load_state_dict(torch.load(model_path))
        if test_pred is None:
            test_pred = predict(test_loader, model)
        else:
            test_pred += predict(test_loader, model)

    image_name = pd.Series(glob.glob('test/test/*.jpg')).apply(lambda x:x.split('/')[2]).tolist()
    ## 置信度
    probabilities = torch.softmax(torch.from_numpy(test_pred), dim=1)
    proba_values = torch.max(probabilities, 1).values.tolist()
    proba_label = torch.max(probabilities, 1).indices.tolist()

    ##创建伪标签保存目录
    directory = "fasle_images"
    if not os.path.exists(directory):
        os.makedirs(directory)
        print("Directory created successfully!")
    else:
        print("Directory already exists!")
    ##置信度小于0.8的copy到fasle_images目录中，大于0.8的直接copy到训练数据中
    for i in range(len(image_name)):
        if proba_values[i] < 0.8:
            shutil.copy('test/test/'+image_name[i],'fasle_images/'+image_name[i])
        else:
            shutil.copy('test/test/'+image_name[i],'data/train/d'+str(proba_label[i]+1)+'/d'+str(proba_label[i]+1)+'-'+image_name[i])

