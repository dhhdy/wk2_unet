import torch
import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import PIL
import glob
from PIL import Image
import random
import numpy as np

class Data_Loader(Dataset):
    #初始化读取图片
    def __init__(self, data_path):
        self.data_path = data_path
        self.train_img_path = glob.glob(os.path.join(data_path, 'images/*.tif'))
        self.train_label_path = glob.glob(os.path.join(data_path, '1st_manual/*.gif'))
    #数据增强
    def change(self, image, rd, is_train):
        trans = []
        if rd > 0.75:
            trans.append(transforms.RandomHorizontalFlip(1))
        elif rd > 0.5:
            trans.append(transforms.RandomVerticalFlip(1))
        elif rd > 0.25:
            trans.extend([
                transforms.RandomHorizontalFlip(1),
                transforms.RandomVerticalFlip(1),
            ])
        trans.append(transforms.ToTensor())
        if is_train:
            trans.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        tr = transforms.Compose(trans)
        return tr(image)
    def __getitem__(self, index):
        img_path = self.train_img_path[index]
        label_path = self.train_label_path[index]
        img = Image.open(img_path).convert('RGB')
        lbl = Image.open(label_path).convert('L')
        rd = random.random()
        img = self.change(img, rd, True)
        lbl = self.change(lbl, rd, False)
        # label二值化
        lbl[lbl > 0] = 1
        lbl[lbl <= 0] = 0
        return img, lbl

    def __len__(self):
        return len(self.train_img_path)

# if __name__ == '__main__':
#     data = Data_Loader('.')
#     print(data.__len__())
#     img = Image.open(data.train_label_path[0])
#     img.show()
#     #print(os.path.abspath(data.train_img_path))