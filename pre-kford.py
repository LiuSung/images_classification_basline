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

test_pred = None
for model_path in ['resnet50_fold_dict0.pt', 'resnet50_fold_dict1.pt', 'resnet50_fold_dict2.pt',
                   'resnet50_fold_dict3.pt', 'resnet50_fold_dict4.pt', 'resnet50_fold_dict5.pt',
                   'resnet50_fold_dict6.pt', 'resnet50_fold_dict7.pt', 'resnet50_fold_dict8.pt',
                   'resnet50_fold_dict9.pt']:

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

test_csv = pd.DataFrame()
test_csv['uuid'] = pd.Series(glob.glob('test/test/*.jpg')).apply(lambda x:x.split('/')[2]).tolist()
test_csv['label'] = np.argmax(test_pred, 1)
test_csv['label'] = test_csv['label'].map({0:'d1', 1:'d2', 2:'d3', 3:'d4', 4:'d5', 5:'d6', 6:'d7', 7:'d8',8:'d9'})
test_csv.to_csv('pre_submit.csv', index=None)