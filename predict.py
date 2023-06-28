import torch.utils.data.distributed
import torchvision.transforms as transforms

from torch.autograd import Variable
from PIL import Image
import pandas as pd
import glob
import torch.nn as nn
from torchvision import models
import timm
from tqdm import tqdm

path_pre_img = pd.Series(glob.glob('test/test/*'))
path_pre_img = path_pre_img.apply(lambda x:x.split('/')[2]).tolist()

transform_test = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class SVHN_Model2(nn.Module):
    def __init__(self):
        super(SVHN_Model2, self).__init__()

        # resnet18
        model_conv = models.resnext101_32x8d(pretrained=True)
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

model = SVHN_Model2().cuda()
model.load_state_dict(torch.load("resnet50_fold0.pt"))

classes=['d1','d2','d3','d4','d5','d6','d7','d8','d9']
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.eval()
model.to(DEVICE)
path='test/test/'
pre_data =[]
pre_label=[]
for i in tqdm(range(len(path_pre_img))):
    img=Image.open(path+path_pre_img[i])
    img=transform_test(img)
    img.unsqueeze_(0)
    img = Variable(img).to(DEVICE)
    out=model(img)
    # Predict
    _, pred = torch.max(out.data, 1)
    pre_data.append(path_pre_img[i])
    pre_label.append(classes[pred.data.item()])

sub = pd.DataFrame()
sub['uuid'] = pre_data
sub['label'] = pre_label
sub.to_csv('./pre_submit.csv',index=False)