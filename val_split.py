import random
import os
import shutil

train_dir = 'data-nong/train'
val_dir = 'data-nong/val'

## 获取类别个数
classes = os.listdir(train_dir)
class_num = len(os.listdir(train_dir))
class_count = {}
for c in classes:
    class_path = os.path.join(train_dir, c)
    class_files = os.listdir(class_path)
    class_count[c] = len(class_files)
print(class_count)

# 从每个类别中随机抽取一部分图片作为验证集
os.mkdir(val_dir)
val_percent = 0.2
for c, n in class_count.items():
    val_num = int(n * val_percent)
    if val_num == 0:
        continue
    class_path = os.path.join(train_dir, c)
    files = os.listdir(class_path)
    random.shuffle(files)
    for f in files[:val_num]:
        src_path = os.path.join(class_path, f)
        dst_path = os.path.join(val_dir, c, f)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.move(src_path, dst_path)

# 显示各类别的数量和验证集数量
print('Total number of classes:', class_num)
for c, n in class_count.items():
    print('Class {}: {} images, {} in validation set'.format(c, n, int(n*val_percent)))