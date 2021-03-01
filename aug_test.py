import cv2
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa

seq = iaa.Sequential([
    iaa.Sometimes(0.5,iaa.Crop(px=(0,16))),
    iaa.Affine(rotate=(-90,90)),
    iaa.Sometimes(0.5,iaa.Fliplr(0.5)),
    iaa.Sometimes(0.5,iaa.GaussianBlur((0, 0.5)),
    iaa.Sometimes(0.5,iaa.AdditiveGaussianNoise(loc=0,scale=(0.0,0.05*255),per_channel=0.5)),
    random_state=True)
])

data_dir = "data/"
index = 1
filename = "{:05}".format(index)+".png"
maskpath = data_dir+"/masks/{:05}_matte".format(index)+".png"
image = cv2.imread(data_dir+"/imgs/"+filename)
mask = cv2.imread(maskpath)
mask = mask > 0
while True:
    seg_map = ia.SegmentationMapsOnImage(mask, shape = image.shape)
    image_aug, seg_aug = seq(image=image, segmentation_maps = seg_map)
    cells = [image,
        seg_aug.draw_on_image(image_aug)[0],
        seg_aug.draw(size=image.shape[:2])[0]
    ]
    grid_image = ia.draw_grid(cells,cols=3)
    cv2.imshow("grid",grid_image)
    cv2.waitKey()