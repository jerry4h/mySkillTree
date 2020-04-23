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

def tool(root, trainOut, validOut, suffix='_label.jpg'):
    ids = [i.rstrip(suffix) for i in os.listdir(root) if check(i, suffix)]
    ids.sort()
    trainNum = len(ids) * 3 // 4
    with open(trainOut, 'w') as f:
        for name in ids[:trainNum]:
            f.write('{}\n'.format(name))
    with open(validOut, 'w') as f:
        for name in ids[trainNum:]:
            f.write('{}\n'.format(name))

            
if __name__ == '__main__':
    root = '/mnt/hjr/celebrityBlended'
    trainOut = 'train.txt'
    validOut = 'valid.txt'
    tool(root, trainOut, validOut)
