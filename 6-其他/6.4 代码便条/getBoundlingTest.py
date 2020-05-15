import numpy as np
import cv2


def get_bounding(mask):
    print(type(mask), mask.shape, mask.dtype)
    cv2.GaussianBlur(mask, (3, 3), 3)
    # bounding = np.zeros((mask.shape[1], mask.shape[0], 3))
    bounding = 4* mask * (1-mask)
    # for i in range(mask.shape[1]):
    #     for j in range(mask.shape[0]):
    #         bounding[i, j] = mask[i, j] * (1 - mask[i, j]) * 4 # 处理每个像素点
    return bounding


if __name__ == '__main__':
    mask = cv2.imread('D:/mask.jpg')
    mask = mask / 255.
    bounding = get_bounding(mask)
    cv2.imwrite('D:/mask_xray.jpg', bounding*255)