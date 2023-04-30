import glob
import numpy as np
import torch
import os
from  Unet import UNet
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import PIL
import glob
from PIL import Image
import random

if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(device)
    # 加载网络，图片单通道，分类为1。
    net = UNet(n_channels=3, n_classes=1)
    # 将网络拷贝到device中
    net.to(device=device)
    # 加载模型参数
    net.load_state_dict(torch.load('best_model.pth', map_location=device))
    # 测试模式
    net.eval()
    # 读取所有图片路径
    data_path ='.'
    tests_path = glob.glob(os.path.join(data_path,'DIRVE/test/images/*.tif'))
    print(len(tests_path))
    # 遍历所有图片
    index = 0
    T = []
    T.append(transforms.ToTensor())
    T.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    tran = transforms.Compose(T)

    with torch.no_grad():
        for test_path in tests_path:
                index += 1
                # 保存结果地址
                save_path = '.'
                save_res_path = os.path.join(save_path, 'DIRVE/result')
                save_res_path = os.path.join(save_res_path, f'{index}'+'_res.png')
                # img -> tensor
                img = Image.open(test_path).convert('RGB')
                img = tran(img)
                # N C H W
                print(img.shape)
                img = img.reshape(1, 3, img.shape[1], img.shape[2])
                img = img.to(device=device, dtype=torch.float32)

                # 预测 !!
                pred = net(img)

                # tensor -> numpy
                # print(pred.data.cpu()[0][0].shape, pred.data.cpu()[0][0]) [1, 1, 512, 512] -> [512, 512]
                # print(type(pred))
                # 将数据拿到cpu上，cuda -> cpu
                pred = np.array(pred.data.cpu()[0])[0]
                # print(type(pred))
                pred[pred > 0] = 255
                pred[pred <= 0] = 0

                # 像素需要整形保存
                pred = pred.astype(np.uint8)
                pred = Image.fromarray(pred)
                pred.save(save_res_path)