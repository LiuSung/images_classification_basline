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

## 灰度图应用直方图均衡化
def equalized(src):
    all_items = os.listdir(src)  # data/train/
    for i in tqdm(range(len(all_items))):
        data_dir = src + all_items[i]
        imgs_names = glob.glob(data_dir + '/*.jpg')
        count = 0
        for j in range(len(imgs_names)):
            img = cv2.imread(imgs_names[i])
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 应用直方图均衡化
            equalized = cv2.equalizeHist(gray)
            count += 1
            cv2.imwrite("data/train/"+all_items[i]+"/equalized"+str(count)+".jpg", equalized)
    # # 显示原始图像和增强后的图像
    # fig, ax = plt.subplots(1,2 , figsize=(10, 5))
    # ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # ax[0].set_title('Original Image')
    # ax[0].axis('off')


    # ax[1].imshow(equalized, cmap='gray')
    # ax[1].set_title('denoised Image')
    # ax[1].axis('off')

    # plt.show()
## 高斯噪声
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
    # equalized("data-nong/train/")
    # gossimage("data-nong/train/")