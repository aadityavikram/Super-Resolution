# making a data loader same as that of mnist in torchvision.datasets

import os
import torch
import random
from PIL import Image
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader

device = 'cuda'
imagenet_mean = torch.FloatTensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(2)
imagenet_std = torch.FloatTensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(2)
imagenet_mean_cuda = torch.FloatTensor([0.485, 0.456, 0.406]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
imagenet_std_cuda = torch.FloatTensor([0.229, 0.224, 0.225]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)


class LoadDataResNet(Dataset):
    def __init__(self, root, crop_size, scaling_factor):
        self.crop_size = int(crop_size)
        self.scaling_factor = int(scaling_factor)
        self.files = []
        for img in os.listdir(root):
            self.files.append(os.path.join(root, img))

    def __getitem__(self, item):
        img = Image.open(self.files[item], mode='r')
        img = img.convert('RGB')

        left = random.randint(1, img.width - self.crop_size)
        top = random.randint(1, img.height - self.crop_size)
        right = left + self.crop_size
        bottom = top + self.crop_size
        hr_img = img.crop((left, top, right, bottom))

        # Downsize this crop to obtain a low-resolution version of it
        lr_width = int(hr_img.width / self.scaling_factor)
        lr_height = int(hr_img.height / self.scaling_factor)
        lr_img = hr_img.resize((lr_width, lr_height), Image.BICUBIC)

        # Normalize the images after converting into tensors
        transform = ToTensor()
        lr_img = transform(lr_img)
        if lr_img.ndimension() == 3:
            lr_img = (lr_img - imagenet_mean) / imagenet_std
        elif lr_img.ndimension() == 4:
            lr_img = (lr_img - imagenet_mean_cuda) / imagenet_std_cuda

        hr_img = transform(hr_img)
        hr_img = 2. * hr_img - 1.

        return lr_img, hr_img

    def __len__(self):
        return len(self.files)


def load_data_resnet(root='', crop_size=96, scaling_factor=4, batch_size=16):
    data = LoadDataResNet(root, crop_size=crop_size, scaling_factor=scaling_factor)
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    return loader
