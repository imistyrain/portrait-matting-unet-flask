from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import imgaug as ia
from imgaug import augmenters as iaa

class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        #self.ids = self.ids[:20]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)
        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def preprocess_all(self, img, mask, scale):
        w, h = img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = img.resize((newW, newH))
        pil_mask = mask.resize((newW, newH))
        img_nd = np.array(pil_img)
        mask_nd = np.array(pil_mask)
        seq = iaa.Sequential([
            iaa.Sometimes(0.5,iaa.Crop(px=(0,16))),
            iaa.Affine(rotate=(-90,90)),
            iaa.Sometimes(0.5,iaa.Fliplr(0.5)),
            iaa.Sometimes(0.5,iaa.GaussianBlur((0, 0.5)),
            iaa.Sometimes(0.5,iaa.AdditiveGaussianNoise(loc=0,scale=(0.0,0.05*255),per_channel=0.5)),
            random_state=True)
        ])
        seg_map = ia.SegmentationMapsOnImage(mask_nd, shape = img_nd.shape)
        image_aug, seg_aug = seq(image=img_nd, segmentation_maps = seg_map)
        seg_map = seg_aug.get_arr()
        img_trans = image_aug.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255
        seg_map = np.expand_dims(seg_map, axis=2)
        seg_trans = seg_map.transpose((2, 0, 1))
        if seg_trans.max() > 1:
            seg_trans = seg_trans / 255

        return img_trans, seg_trans


    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + '*')
        img_file = glob(self.imgs_dir + idx + '*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'
        # img = self.preprocess(img, self.scale)
        # mask = self.preprocess(mask, self.scale)
        img, mask = self.preprocess_all(img, mask, self.scale)

        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}
