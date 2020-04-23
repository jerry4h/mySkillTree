import os
import os.path as osp
import numpy as np
from tqdm import tqdm
import cv2
import pdb
from shutil import copyfile


def check(i, suffix):
    if str(i).endswith(suffix):
        return True
    return False

def tool(root, dataOut, labelOut):
    for name in os.listdir(root):
        if check(name, suffix='_label.jpg'):
            copyfile(osp.join(root, name), labelOut)
        else:
            copyfile(osp.join(root, name), dataOut)


if __name__ == '__main__':
    root = '/mnt/hjr/celebrityBlended'
    trainOut = '/mnt/hjr/BlendedTrain/data'
    validOut = '/mnt/hjr/BlendedTrain/label'
    tool(root, trainOut, validOut)
