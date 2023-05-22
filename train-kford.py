import glob
import numpy as np
from PIL import Image
from sklearn.model_selection import  KFold
# from efficientnet_pytorch import EfficientNet
import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset

## 定义模型
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

## 训练方法
def train(train_loader, model, criterion, optimizer, epoch):
    train_loss = 0.0
    model.train()
    for i, (input, target) in enumerate(train_loader):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        # compute output
        output = model(input)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.data.item()
    train_avg_loss = train_loss/len(train_loader)
    return train_avg_loss

##预测方法
def validate(val_loader, model, criterion):
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    with torch.no_grad():
        for i,(input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()
            output = model(input)

            loss = criterion(output, target)
            val_loss += loss.data.item()

            _, predicted = torch.max(output.data, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)

    val_avg_loss = val_loss/len(val_loader)
    accuracy = correct / total
    return val_avg_loss, accuracy

## 定义dataset类以及标签对齐方法
def imgs_label(train_path):
    if 'd1' in train_path:
        return 0
    elif 'd2' in train_path:
        return 1
    elif 'd3' in train_path:
        return 2
    elif 'd4' in train_path:
        return 3
    elif 'd5' in train_path:
        return 4
    elif 'd6' in train_path:
        return 5
    elif 'd7' in train_path:
        return 6
    elif 'd8' in train_path:
        return 7
    else:
        return 8
class QRDataset(Dataset):
    def __init__(self, train_jpg, transform=None):
        self.train_jpg = train_jpg
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        img = Image.open(self.train_jpg[index]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img,torch.from_numpy(np.array(imgs_label(self.train_jpg[index])))

    def __len__(self):
        return len(self.train_jpg)

## 加载数据
train_jpg = np.array(glob.glob('data/train/*/*.jpg'))

## 十折交叉验证
skf = KFold(n_splits=10,random_state=233,shuffle=True)
for flod_idx, (train_idx, val_idx) in enumerate(skf.split(train_jpg)):
    train_loader = torch.utils.data.DataLoader(
        QRDataset(train_jpg[train_idx],
                transforms.Compose([
                            # transforms.RandomGrayscale(),
                            transforms.Resize((384, 384)),
                            transforms.RandomAffine(10),
                            # transforms.ColorJitter(hue=.05, saturation=.05),
                            # transforms.RandomCrop((450, 450)),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomVerticalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        ), batch_size=32, shuffle=True, num_workers=20, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        QRDataset(train_jpg[val_idx],
                transforms.Compose([
                            transforms.Resize((384, 384)),
                            # transforms.Resize((124, 124)),
                            # transforms.RandomCrop((450, 450)),
                            # transforms.RandomCrop((88, 88)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        ), batch_size=32, shuffle=False, num_workers=10, pin_memory=True
    )
    model = SVHN_Model2().cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), 0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.85)
    best_acc = 0.0

    for epoch in range(10):
        train_avg_loss = train(train_loader, model, criterion, optimizer, epoch)
        scheduler.step()
        val_avg_loss,val_acc = validate(val_loader, model, criterion)

        print('Ford',flod_idx+1,' Epoch:', epoch+1,' train_avg_loss:',train_avg_loss,' val_avg_loss:',val_avg_loss,' val_acc:',val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model, './resnet50_fold{0}.pkl'.format(flod_idx))#将最优模型保存
            torch.save(model.state_dict(), './resnet50_fold_dict{0}.pt'.format(flod_idx))
