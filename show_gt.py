import numpy as np
import torch
import cv2
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader

dir_img = 'data/imgs/'
dir_mask = 'data/masks/'

def show_gt():
    batch_size = 1
    img_scale = 1
    dataset = BasicDataset(dir_img, dir_mask, img_scale)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    for batch in train_loader:
        img = batch['image'].numpy()[0].transpose(1,2,0)
        mask = batch['mask'].numpy()[0].transpose(1,2,0)
        cv2.imshow("img",img)
        cv2.imshow("mask",mask)
        cv2.waitKey()

if __name__=="__main__":
    show_gt()