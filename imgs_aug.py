import os
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

## 常规数据增广
def normal(src):
    # 定义增广方法,需要根据任务以及图像适当做调整
    trains_list = [
        transforms.Compose([
            transforms.RandomHorizontalFlip(p=1),
            transforms.ToTensor(),
    ]),
        transforms.Compose([
            transforms.RandomVerticalFlip(p=1),
                transforms.ToTensor(),
    ]),
        transforms.Compose([
            transforms.RandomRotation(20),
            transforms.ToTensor(),
    ]),
    #     transforms.Compose([
    #         transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  # 随机调整亮度，对比度，饱和度和色相
    #         transforms.ToTensor(),
    # ]),
    #     transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.RandomErasing(p=1, scale=(0.02, 0.1), ratio=(0.3, 0.3), value=(0, 0, 0)),
    # ]),
    #     transforms.Compose([
    #         transforms.Resize((1024, 1024)),
    #         transforms.RandomCrop(size=512, padding=0, pad_if_needed=False, fill=0, padding_mode='constant'),
    #         transforms.ToTensor(),
    # ])
    ]
    all_items = os.listdir(src) #data/train/

    for h in tqdm(range(len(all_items))):
        data_dir = src + all_items[h]
        imgs_names = glob.glob(data_dir + '/*.jpg')
        count = 0
        for i in range(len(imgs_names)):
            image = Image.open(imgs_names[i]).convert('RGB')
            for trans in trains_list:
                aug_image = trans(image)
                count += 1
                filename = f'{all_items[h]}_image_{count}.jpg'
                save_path = os.path.join("data/train/"+all_items[h], filename)
                aug_image_pil = transforms.ToPILImage()(aug_image)
                aug_image_pil.save(save_path)


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

def gossimage(src):
    all_items = os.listdir(src)  # data/train/
    for i in tqdm(range(len(all_items))):
        data_dir = src + all_items[i]
        imgs_names = glob.glob(data_dir + '/*.jpg')
        count = 0
        for j in range(len(imgs_names)):
            img = cv2.imread(imgs_names[i],cv2.COLOR_BGR2GRAY)
            # 添加高斯噪声
            mean = 0  # 均值
            std = 20  # 标准差，控制噪声强度
            noisy = img + np.random.normal(mean, std, img.shape)
            noisy = cv2.normalize(noisy, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            count += 1
            cv2.imwrite("data/train/"+all_items[i]+"/goss"+str(count)+".jpg", noisy)



if __name__ == '__main__':
    normal("data-nong/train/")
    gossimage("data-nong/train/")

    ## random_erase
    all_items = os.listdir('data-nong/train')
    for i in tqdm(range(len(all_items))):
        data_dir = 'data-nong/train/' + all_items[i]
        imgs_names = glob.glob(data_dir + '/*.jpg')
        count = 0
        for j in range(len(imgs_names)):
            img = Image.open(imgs_names[j]).convert('RGB')
            img2 = random_erase(img, 150, 3, 1)
            count += 1
            img2.save("data/train/" + all_items[i] + "/erase" + str(count) + ".jpg")
