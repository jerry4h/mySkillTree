# -*- coding: gbk -*-

import os
import os.path as osp
import numpy as np
from random import sample
from tqdm import tqdm
import cv2
import pdb
from shutil import copyfile

class fileTools:
    def __init__(self):
        pass

    def Select(self, inPath, outPath, checkFunc, workFunc, selectNum=1):
        ''' Universal Function for Select something from Path.
        inPath: directory like this:  /ids/something
        outPath: directory of root.
        checkFunc: func obj for satisfaction check
        workFunc: func obj for work
        selectNum: max select number, default=1
        '''
        ids = os.listdir(inPath)
        for idName in tqdm(ids):
            idPath = osp.join(inPath, idName)
            names = os.listdir(idPath)
            k = min(len(names), selectNum)
            names = sample(names, k)
            # names.sort()
            # satisfied = 0
            for name in names:
                imgPath = osp.join(idPath, name)
                status = checkFunc(imgPath)
                if status:
                    workFunc(imgPath, outPath)
                    # satisfied += status
                # if satisfied == selectNum:
                #     break
                

def isColorImage(imgPath, suffix='.jpg'):
    ''' 判断是否是彩色图
    方法比较low，比较channel像素数值之和是否相等，后期再改改吧。
    '''
    if not str(imgPath).endswith(suffix) or\
         not osp.isfile(imgPath):
        return False
    try:
        with open(imgPath, 'rb') as f:
            img = cv2.imread(imgPath)
            if img is None or len(img.shape) == 2:
                return False
    except IOError:
        print('Cannot load image ', imgPath)
        return False
    reshaped = img.reshape(-1,3)
    channel_0 = reshaped[:,0].sum(-1)
    channel_1 = reshaped[:,1].sum(-1)
    return  channel_0 != channel_1

def copyFile(filePath, outRoot):
    fileName = osp.split(filePath)[-1]
    dstPath = osp.join(outRoot, fileName)
    copyfile(filePath, dstPath)
    
def copyFile2(filePath, outRoot):
    # import pdb
    # pdb.set_trace()
    temp, name = osp.split(filePath)
    _, idx = osp.split(temp)
    fileName = '_'.join([idx, name])
    dstPath = osp.join(outRoot, fileName)
    copyfile(filePath, dstPath)

            
if __name__ == '__main__':
    ftools = fileTools()
    inPath = './generator'
    outPath = './randomSelected' 
    ftools.Select(inPath=inPath, outPath=outPath,
                  checkFunc=isColorImage, workFunc=copyFile2,selectNum=1)
