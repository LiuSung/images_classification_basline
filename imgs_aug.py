import os
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import glob


# 定义增广方法,需要根据任务以及图像适当做调整
transform_HRandom = transforms.Compose([
    transforms.RandomHorizontalFlip(p=1),
    transforms.ToTensor(),
])
transform_VRandom = transforms.Compose([
    transforms.RandomVerticalFlip(p=1),
    transforms.ToTensor(),
])
transform_xuanzhuan = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.ToTensor(),
])
transform_cropRandom = transforms.Compose([
    transforms.Resize((1024,1024)),
    transforms.RandomCrop(size=512,padding=0, pad_if_needed=False, fill=0, padding_mode='constant'),
    transforms.ToTensor(),
])
transform_Eras = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomErasing(p=1,scale=(0.02, 0.1), ratio=(0.3, 0.3),value=(0,0,0)),
])

all_items = os.listdir('data/train')

for h in tqdm(range(len(all_items))):
    data_dir = 'data/train/' + all_items[h]
    imgs_names = glob.glob(data_dir + '/*.jpg')
    count = 0
    for i in range(len(imgs_names)):
        image = Image.open(imgs_names[i]).convert('RGB')
        aug_image = transform_HRandom(image)
        count += 1
        filename = f'{all_items[h]}_image_{count}.jpg'
        save_path = os.path.join(data_dir, filename)
        aug_image_pil = transforms.ToPILImage()(aug_image)
        aug_image_pil.save(save_path)

        aug_image1 = transform_VRandom(image)
        count += 1
        filename = f'{all_items[h]}_image_{count}.jpg'
        save_path = os.path.join(data_dir, filename)
        aug_image_pil = transforms.ToPILImage()(aug_image1)
        aug_image_pil.save(save_path)

        aug_image2 = transform_xuanzhuan(image)
        count += 1
        filename = f'{all_items[h]}_image_{count}.jpg'
        save_path = os.path.join(data_dir, filename)
        aug_image_pil = transforms.ToPILImage()(aug_image2)
        aug_image_pil.save(save_path)

        aug_image3 = transform_Eras(image)
        count += 1
        filename = f'{all_items[h]}_image_{count}.jpg'
        save_path = os.path.join(data_dir, filename)
        aug_image_pil = transforms.ToPILImage()(aug_image3)
        aug_image_pil.save(save_path)

        aug_image4 = transform_cropRandom(image)
        count += 1
        filename = f'{all_items[h]}_image_{count}.jpg'
        save_path = os.path.join(data_dir, filename)
        aug_image_pil = transforms.ToPILImage()(aug_image4)
        aug_image_pil.save(save_path)