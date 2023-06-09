import os
import argparse
import sys
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torchvision import transforms
import torchvision.models as models
import timm
from my_dataset import MyDataSet
from utils import read_split_data
from pprint import pprint

class SVHN_Model2(nn.Module):
    def __init__(self):
        super(SVHN_Model2, self).__init__()


        model_conv = timm.create_model('resnext101_32x8d', pretrained=True)
        # model_conv = models.resnext101_32x8d(pretrained=True)
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


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    img_size = args.img_size
    data_transform = {
        "train": transforms.Compose([transforms.Resize((img_size, img_size)),
                                     transforms.RandomAffine(10),
                                     transforms.ColorJitter(hue=.05, saturation=.05),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomVerticalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize((img_size, img_size)),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)
    net = SVHN_Model2()
    net.to(device)

    ##定义超参数
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.85)
    epochs = args.epochs

    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
        scheduler.step()
        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)

        val_accurate = acc / len(val_dataset)
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))
        torch.save(net.state_dict(), './weights/best_model-{}.pth'.format(epoch))
        # if val_accurate > best_acc:
        #     best_acc = val_accurate
        #     torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    model_names = timm.list_models(pretrained=True)
    pprint(model_names)

    parser = argparse.ArgumentParser()

    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--data-path', type=str,
                        default="data/train")
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=20)
    opt = parser.parse_args()

    main(opt)
