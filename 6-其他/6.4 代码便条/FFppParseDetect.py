
import os.path as osp
from tqdm import tqdm
import json
import argparse
import pdb
from shutil import copyfile

def get_args():
    parser = argparse.ArgumentParser('FaceForensics++ Parse for Train/Val/Test set.')
    parser.add_argument('--mode', type=str, help='train/val/test set split',\
        default='train', choices=['train', 'val', 'test'])
    parser.add_argument('--method', type=str, help='method of originals/manipulation.',\
        default='originals', choices=['original', 'Face2Face', 'Deepfakes', 'FaceSwap'])
    parser.add_argument('--root', type=str, help='root path for dataset',\
        default='D:/Dataset')

    args = parser.parse_args()
    return args


def get_ids(jsonPath):
    with open(jsonPath, 'r') as f:
        splits = json.load(f)
    return splits

def get_video_names(splits, method='original'):
    names = []
    suffix = '.mp4'
    for pair in splits:
        if method == 'original':
            name1 = str(pair[0])
            name2 = str(pair[1])
        else:
            name1 = '_'.join(pair)
            name2 = '_'.join(pair[::-1])
        names.append(name1 + suffix)
        names.append(name2 + suffix)
    return names

def start(root, method, names, func):
    methodRoot = osp.join(root, method, 'videos')
    dstPath = osp.join(root, method, 'videos_test')
    for name in tqdm(names):
        videoName = osp.join(methodRoot, name)
        status = func(videoName, dstPath)
        if not status:
            print("Error: func for {} and {}".format(videoName, dstPath))
            continue

def fileCheck(videoName, dstPath):
    if osp.isfile(videoName) and videoName.endswith('.mp4'):
        return True
    return False

def fileCopy(videoPath, dstPath):
    if not osp.isfile(videoPath) or not videoPath.endswith('.mp4'):
        return False
    videoName = osp.split(videoPath)[-1]
    dstPath = osp.join(dstPath, videoName)
    copyfile(videoPath, dstPath)
    return True


if __name__ == '__main__':
    args = get_args()
    jsonPath = osp.join(args.root, args.mode + '.json')
    splits = get_ids(jsonPath)
    names = get_video_names(splits, args.method)
    start(args.root, args.method, names, fileCopy)
