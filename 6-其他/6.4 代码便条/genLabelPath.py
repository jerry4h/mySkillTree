import os
import os.path as osp
from random import sample
import numpy as np
from tqdm import tqdm
import cv2
import pdb
import glob
from shutil import copyfile


def check(i):
    if str(i).endswith('.jpg') and not str(i).endswith('_label.jpg'):
        return True
    return False

def processDir(root, f, dir, label, sampleNum=-1):
    lines = []
    for imgName in glob.glob(dir):
        relPath = imgName
        imgPath = osp.join(root, relPath)
            
        if check(imgPath):  
            line = '%d %s\n' %(label, relPath)
            lines.append(line)
    
    if sampleNum != -1 and sampleNum < len(lines):
        lines = sample(lines, sampleNum)
    for line in lines:
        f.write(line)

def generate(root, outTxt, posDir, negDir):
    SAMPLENUM = 1000
    with open(outTxt, 'w') as f:
        processDir(root, f, posDir, label=1, sampleNum=SAMPLENUM)
        processDir(root, f, negDir, label=0, sampleNum=SAMPLENUM)


if __name__ == '__main__':
    root = '/mnt/hjr/FF++c23'
    outTxt = 'FF++C23Test.txt'
    posDir = './*e*/images_test_256/*/[0-9].jpg'
    negDir = './original/images_test_256/*/[0-9].jpg'
    generate(root, outTxt, posDir, negDir)
