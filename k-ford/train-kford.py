import glob
import numpy as np
from PIL import Image
from sklearn.model_selection import KFold
import argparse
# from efficientnet_pytorch import EfficientNet
import torch
import sys
from tqdm import tqdm
torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

import torchvision.models as models
import timm
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset


## 定义模型
class SVHN_Model2(nn.Module):
    def __init__(self):
        super(SVHN_Model2, self).__init__()

        # resnet18
        # model_conv = timm.create_model('resnext101_32x8d', pretrained=True)
        model_conv = models.resnet50(pretrained=True)
        fc_infeatures = model_conv.fc.in_features
        model_conv.avgpool = nn.AdaptiveAvgPool2d(1)
        model_conv = nn.Sequential(*list(model_conv.children())[:-1])  # 去除最后一个fc layer
        self.cnn = model_conv

        self.hd_fc1 = nn.Linear(fc_infeatures, 128)
        self.dropout_1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128, 25)

    def forward(self, img):
        feat = self.cnn(img)
        feat = feat.view(feat.shape[0], -1)
        feat1 = self.hd_fc1(feat)
        feat1 = self.dropout_1(feat1)
        c1 = self.fc1(feat1)
        return c1


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
        return img, torch.from_numpy(np.array(int(self.train_jpg[index].split('/')[2])))

    def __len__(self):
        return len(self.train_jpg)

def main(args):
    ## 十折交叉验证
    train_jpg = np.array(glob.glob(args.data_path+'/*/*.jpg'))
    skf = KFold(n_splits=5, random_state=334, shuffle=True)
    for flod_idx, (train_idx, val_idx) in enumerate(skf.split(train_jpg)):
        train_loader = torch.utils.data.DataLoader(
            QRDataset(train_jpg[train_idx],
                  transforms.Compose([
                      # transforms.RandomGrayscale(),
                      transforms.Resize((224, 224)),
                      # transforms.RandomAffine(10),
                      # transforms.ColorJitter(hue=.05, saturation=.05),
                      # transforms.RandomCrop((450, 450)),
                      # transforms.RandomHorizontalFlip(),
                      # transforms.RandomVerticalFlip(),
                      transforms.ToTensor(),
                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                  ])
                  ), batch_size=16, shuffle=True, num_workers=10, pin_memory=True
    )
        val_loader = torch.utils.data.DataLoader(
            QRDataset(train_jpg[val_idx],
                  transforms.Compose([
                      transforms.Resize((224, 224)),
                      # transforms.Resize((124, 124)),
                      # transforms.RandomCrop((450, 450)),
                      # transforms.RandomCrop((88, 88)),
                      transforms.ToTensor(),
                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                  ])
                  ), batch_size=16, shuffle=False, num_workers=10, pin_memory=True
        )
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        epochs = args.epochs
        model = timm.create_model('convnext_small.fb_in22k', pretrained=True, num_classes=25).to(device)
        loss_function = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.SGD(model.parameters(), 0.01)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.85)
        best_acc=0.0

        train_steps = len(train_loader)
        val_steps = len(val_loader)


        for epoch in range(epochs):
            # train
            model.train()
            running_loss = 0.0
            train_bar = tqdm(train_loader, file=sys.stdout)
            for step, data in enumerate(train_bar):
                images, labels = data
                optimizer.zero_grad()
                logits = model(images.to(device))
                loss = loss_function(logits, labels.to(device))
                loss.backward()
                optimizer.step()

            # print statistics
                running_loss += loss.item()

                train_bar.desc = "fold{} train epoch[{}/{}] loss:{:.3f}".format(flod_idx+1 ,epoch + 1,
                                                                     epochs,
                                                                     loss)
            scheduler.step()
        # validate
            model.eval()
            acc = 0.0  # accumulate accurate number / epoch
            val_loss_t = 0.0
            with torch.no_grad():
                val_bar = tqdm(val_loader, file=sys.stdout)
                for val_data in val_bar:
                    val_images, val_labels = val_data
                    outputs = model(val_images.to(device))
                    val_loss = loss_function(outputs, val_labels.to(device))
                    val_loss_t += val_loss.item()

                    predict_y = torch.max(outputs, dim=1)[1]
                    acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                    val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)

            val_accurate = acc / len(train_jpg[train_idx])

            print('[fold %d] [epoch %d] train_loss: %.3f val_loss: %.3f val_accuracy: %.3f' %
                (flod_idx,epoch + 1, running_loss / train_steps, val_loss_t / val_steps, val_accurate))
            if best_acc < val_accurate:
                torch.save(model.state_dict(), './weights/kfold_model-{}.pth'.format(flod_idx+1))

if __name__ == '__main__':

        parser = argparse.ArgumentParser()

        parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
        parser.add_argument('--data-path', type=str,
                            default="data-nong/train")
        parser.add_argument('--img_size', type=int, default=224)
        parser.add_argument('--batch-size', type=int, default=8)
        parser.add_argument('--lr', type=float, default=0.01)
        parser.add_argument('--epochs', type=int, default=8)
        opt = parser.parse_args()

        main(opt)

