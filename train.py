import os
import argparse
import sys
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torchvision import transforms
import timm
from my_dataset import MyDataSet
from utils import read_split_data
import numpy as np
from sklearn.model_selection import KFold
import glob


def train(net,optimizer,loss_function,train_loader,device,epochs,epoch):
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
    return running_loss

def val_smooth(net,loss_function,val_loader,device,epochs,epoch):
    net.eval()
    acc = 0.0  # accumulate accurate number / epoch
    val_loss_t = 0.0
    with torch.no_grad():
        val_bar = tqdm(val_loader, file=sys.stdout)
        for val_data in val_bar:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            val_loss = loss_function(outputs, val_labels.to(device))
            val_loss_t += val_loss.item()

            predict_y = torch.tensor([output.argmax() for output in outputs])
            val_labels = torch.tensor([val_label.argmax().to(device) for val_label in val_labels])

            acc += torch.eq(predict_y, val_labels).sum().item()

            val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                       epochs)
    return acc,val_loss_t

def val(net,loss_function,val_loader,device,epochs,epoch):
    net.eval()
    acc = 0.0  # accumulate accurate number / epoch
    val_loss_t = 0.0
    with torch.no_grad():
        val_bar = tqdm(val_loader, file=sys.stdout)
        for val_data in val_bar:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            val_loss = loss_function(outputs, val_labels.to(device))
            val_loss_t += val_loss.item()

            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

            val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                       epochs)
    return acc,val_loss_t

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    img_size = args.img_size
    data_transform = {
        "train": transforms.Compose([transforms.Resize((img_size, img_size)),
                                     transforms.RandomHorizontalFlip(p=0.5),
                                     transforms.RandomVerticalFlip(p=0.5),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize((img_size, img_size)),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              label=train_images_label,
                              transforms=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            label=val_images_label,
                            transforms=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw)
    net = timm.create_model('convnext_small_in22k', pretrained=True, num_classes=12).to(device)
    net.to(device)

    ##定义超参数
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.85)
    epochs = args.epochs

    train_steps = len(train_loader)
    val_steps = len(val_loader)
    best_acc=0.0
    for epoch in range(epochs):
        # train
        running_loss = train(net, optimizer, loss_function, train_loader, device, epochs, epoch)
        scheduler.step()
        # validate
        acc, val_loss_t = val(net, loss_function, val_loader, device, epochs, epoch)

        val_accurate = acc / len(val_dataset)
        print('[epoch %d] train_loss: %.3f val_loss: %.3f val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_loss_t / val_steps, val_accurate))
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), './weights/best_model-{}.pth'.format(epoch))

    print('Finished Training')


def kfoldmain(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")
    train_jpg = np.array(glob.glob(args.data_path + 'data/*/*/*.jpg'))
    skf = KFold(n_splits=args.number, random_state=334, shuffle=True)
    for flod_idx, (train_idx, val_idx) in enumerate(skf.split(train_jpg)):
        img_size = args.img_size
        data_transform = {
            "train": transforms.Compose([transforms.Resize((img_size, img_size)),
                                         transforms.RandomHorizontalFlip(p=0.5),
                                         transforms.RandomVerticalFlip(p=0.5),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
            "val": transforms.Compose([transforms.Resize((img_size, img_size)),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        }
        train_images_path = train_jpg[train_idx]
        train_images_label = [node.split('/')[2] for node in train_images_path]

        val_images_path = train_jpg[val_idx]
        val_images_label = [node.split('/')[2] for node in val_images_path]
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
        # net = SVHN_Model2()
        # model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=10)
        # checkpoint = torch.load('resnet18_checkpoint.pth')  ##模型权重
        # net.load_state_dict(checkpoint)
        net = timm.create_model('convnext_small_in22k', pretrained=True, num_classes=12).to(device)
        net.to(device)

        ##定义超参数
        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(), args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.85)
        epochs = args.epochs

        train_steps = len(train_loader)
        val_steps = len(val_loader)
        best_acc = 0.0
        for epoch in range(epochs):
            # train
            net.train()
            running_loss = train(net, optimizer, loss_function, train_loader, device, epochs, epoch)
            scheduler.step()
            # validate
            acc, val_loss_t = val(net, loss_function, val_loader, device, epochs, epoch)
            val_accurate = acc / len(val_dataset)
            print('[epoch %d] train_loss: %.3f val_loss: %.3f val_accuracy: %.3f' %
                  (epoch + 1, running_loss / train_steps, val_loss_t / val_steps, val_accurate))

            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(net.state_dict(), './weights/fold-{}.pth'.format(flod_idx))

        print('Finished Training')


if __name__ == '__main__':
    kFold = False
    if kFold is False:
        parser = argparse.ArgumentParser()
        parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
        parser.add_argument('--data-path', type=str,
                            default="data/train")
        parser.add_argument('--img_size', type=int, default=224)
        parser.add_argument('--batch-size', type=int, default=8)
        parser.add_argument('--lr', type=float, default=0.008)
        parser.add_argument('--epochs', type=int, default=20)
        opt = parser.parse_args()

        main(opt)
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
        parser.add_argument('--number', type=int,
                            default=5)
        parser.add_argument('--img_size', type=int, default=256)
        parser.add_argument('--batch-size', type=int, default=16)
        parser.add_argument('--lr', type=float, default=0.01)
        parser.add_argument('--epochs', type=int, default=20)
        opt = parser.parse_args()

