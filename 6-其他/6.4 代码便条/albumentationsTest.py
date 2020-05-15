import numpy as np
import cv2
from matplotlib import pyplot as plt

from albumentations import (
    # HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue, MultiplicativeNoise,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, Flip, OneOf, Compose, JpegCompression, CLAHE
)

def img_loader(path):
    try:
        with open(path, 'rb') as f:
            img = cv2.imread(path)
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
    except IOError:
        print('Cannot load image ' + path)


def augment_and_show(aug, image):
    image = aug(image=image)['image']
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.show()


def blur_aug(p=.5):
    return Compose([
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.2),
    ])


def strong_aug_pixel(p=.5):
    return Compose([
        # RandomRotate90(),
        # Flip(),
        # Transpose(),
        OneOf([
            MultiplicativeNoise(multiplier=[0.5, 1.5], per_channel=True, p=0.2),
            JpegCompression(quality_lower=39, quality_upper=80, p=0.2)
        ], p=0.2),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.2),
        OneOf([
            MotionBlur(p=.2),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        # ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        # OneOf([
        #     OpticalDistortion(p=0.3),
        #     GridDistortion(p=.1),
        #     IAAPiecewiseAffine(p=0.3),
        # ], p=0.2),
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
            RandomBrightnessContrast(),            
        ], p=0.3),
        HueSaturationValue(p=0.3),
    ], p=p)


if __name__ == '__main__':
    aug = strong_aug()
    imgPath = 'D:/Dataset/randomSelected/000.mp4_0.jpg'
    img = img_loader(imgPath)
    augment_and_show(aug, img)

    