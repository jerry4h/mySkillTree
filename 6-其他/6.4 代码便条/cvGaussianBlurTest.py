
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt

def convex_hull(size, points, fillColor=(255,)*3):
    mask = np.zeros(size, dtype=np.uint8) # mask has the same depth as input image
    points = cv2.convexHull(np.array(points))
    corners = np.expand_dims(points, axis=0).astype(np.int32)
    cv2.fillPoly(mask, corners, fillColor)
    return mask


def get_bounding(mask):

    bounding = np.zeros((mask.shape[1], mask.shape[0], 3))
    for i in range(mask.shape[1]):
        for j in range(mask.shape[0]):
            bounding[i, j] = mask[i, j] * (1 - mask[i, j]) * 4 # 处理每个像素点
    return bounding


parser = argparse.ArgumentParser()
parser.add_argument('--ksize', type=int, default=1)
parser.add_argument('--sigma', type=float, default=1)
args = parser.parse_args()

points = [
    [42, 109], [46, 128], [50, 150], [54, 169], [61, 192], [76, 207], [88, 215],
    [107, 222], [137, 226], [163, 215], [179, 207], [194, 196], [205, 177], [213, 158],
    [213, 139], [216, 116], [216, 94], [54, 78], [61, 67], [76, 67], [88, 67],
    [99, 71], [145, 67], [156, 63], [167, 63], [182, 67], [194, 71], [126, 90],
    [126, 105], [126, 120], [126, 131], [114, 143], [118, 146], [129, 146],
    [137, 143], [141, 143], [73, 97], [80, 90], [92, 90], [103, 94], [92, 97],
    [80, 101], [148, 94], [156, 86], [167, 86], [179, 90], [167, 94], [156, 94],
    [103, 177], [110, 169], [122, 162], [129, 165], [137, 162], [148, 165], [156, 173],
    [148, 180], [141, 184], [129, 188], [122, 188], [114, 184], [107, 177], [122, 173],
    [129, 173], [137, 173], [156, 173], [137, 173], [129, 177], [122, 177]
]

hullMask = convex_hull((256, 256, 3), points) / 255.
blured = cv2.GaussianBlur(hullMask, (args.ksize, args.ksize), sigmaX=args.sigma, sigmaY=args.sigma)
plt.subplot(1, 2, 1)
plt.imshow(blured)
plt.subplot(1, 2, 2)
xray = get_bounding(blured)
plt.imshow(xray)
plt.show()



def normalize(img):
    # import pdb
    # pdb.set_trace()
    img = (img / img.max())
    return img

def plot(i, j, ksize, sigma):
    hullMask = convex_hull((256, 256, 3), points) / 255.
    blured = cv2.GaussianBlur(hullMask, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
    xray = get_bounding(blured)
    xray = normalize(xray)
    # plt.plot([1, 1, 1])
    plt.subplot(6, 5, (6-1)*i+j+1)
    plt.imshow(xray)
    plt.title = '{}, {}'.format(ksize, sigma)
    
ksizes = [1, 7, 15, 31, 63, 127]  # 31-63
sigmas = [1, 3, 7, 15, 30]  # 7-15

for i in range(len(ksizes)):
    for j in range(len(sigmas)):
        ksize, sigma = ksizes[i], sigmas[j]
        plot(i, j, ksize, sigma)

plt.show()
