from random import uniform
import numpy as np
import cv2
import matplotlib.pyplot as plt


def linear_deform(warped, scale=0.5, shake_h=0.2, random=True):
    """缩放+高度抖动

    params:
        warped {np.ndarray} -- float mask of areas for transfer.
        scale {float}  -- random minimum scale
            1.0 for keep original scale, 0.0 for one pixel
        shake_h {float} -- random minimum shake for height.
            1.0 for no shake, 0.01 for shake from bottom
    return:
        deformed {np.ndarray} -- float mask.
    """
    if shake_h == 0.0:
        shake_h = 0.001
    h, w, _ = warped.shape
    deformed = np.zeros_like(warped)
    # cv2.imwrite('warped.jpg', warped*255)
    scaleRandom, shakeRandom = scale, shake_h
    if random:
        # randPair = np.random.rand(2)
        # scaleRandom = 1-randPair[0]*scale  # [scale, 1]
        # shakeRandom = randPair[1]*shake_h  # [0， shake_h]
        
        scaleRandom = uniform(min(1, scale), 1.0)
        shakeRandom = uniform(min(1, shake_h), 1.0)
    # print(scaleRandom, shakeRandom)
    hScale, wScale = int(h*scaleRandom), int(w*scaleRandom)
    warped = cv2.resize(warped, (wScale, hScale))
    hPlus = int((1-shakeRandom)*(h-hScale)//2)
    hNew, wNew = int((h-hScale)//2), int((w-wScale)//2)
    hNew += hPlus
    deformed[hNew: hNew+hScale, wNew: wNew+wScale, :] += warped
    # cv2.imwrite('deformed.jpg', deformed*255)
    return deformed


if __name__ == '__main__':
    img = cv2.imread('D:/xray.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    while True:
        deformed = linear_deform(img, scale=1, shake_h=1, random=True)
        plt.imshow(deformed)
        plt.show()