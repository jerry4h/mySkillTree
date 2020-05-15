

import glob
import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
from tqdm import tqdm

from ffppAlignTrans import Alignmenter
alignmentor = Alignmenter()

ALIGN = False
IMAGE_SIZE = 256
DETECT_MARGIN = 100


def parse_lamdmark(lmPath):
    """
    Make sure returned landmark is not None
    """
    with open(lmPath, 'r') as f:
        id_lm = json.load(f)

    # filter None landmarks
    tobe_filterd_keys = []
    
    for key in id_lm:
        if id_lm[key] is None:
            tobe_filterd_keys.append(key)
    for key in tobe_filterd_keys:
        del id_lm[key]
    
    return id_lm


def choose_idx(id_lm, num_samples=100):
    """
    id_lm: each id have an landmark.
    """
    # id_lm = parse_lamdmark(lmPath)
    keys = list(id_lm.keys())
    stride = len(id_lm) // num_samples
    if stride == 0:
        indexes = keys
    else:
        indexes = []
        for i in range(num_samples):
        # while len(indexes) < num_samples:
            key = keys[i*stride]
            if id_lm[key] is not None:
                indexes.append(key)
            else:
                print('lm Skipped:', key, id_lm[key])
    # import pdb; pdb.set_trace()
    return indexes


def parse_face(frame, landmark):
    face = alignmentor(frame, landmark, image_size=IMAGE_SIZE, margin=DETECT_MARGIN, align=ALIGN)
    return face


def parse_video(videoPath, id_lm, indexes):
    reader = cv2.VideoCapture(videoPath)
    frame_num = 0
    faces = []
    while reader.isOpened():
        success, image = reader.read()
        if not success:
            break
        id_key = '{:04d}'.format(frame_num)
        if id_key in indexes:
            # frame_name = '{:04d}.jpg'.format(frame_num)

            face = parse_face(image, id_lm[id_key])
            faces.append(face)
        frame_num += 1
    reader.release()
    return faces


def normalization(x):
    return (x - x.min()) / (x.max() - x.min())

def inner_margin(mask, margin=20):
    mask_ = np.zeros_like(mask)
    w, h = mask.shape[1], mask.shape[0]
    w_, h_ = w-margin*2, h-margin*2
    resized = cv2.resize(mask, (w_,h_), cv2.INTER_LINEAR)
    mask_[margin:h-margin, margin:w-margin] = resized
    # mask = cv2.GaussianBlur(mask, ksize=(ksize,ksize), sigmaX=3)
    # ret, thresh = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    return mask_
    

def polly_to_xray(polly, ksize=21, sigma=3):
    polly = polly / 255.
    blured = cv2.GaussianBlur(polly, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
    xray = 4 * blured * (1 - blured)
    return xray


def residual_to_convex_polly(residual, m=20):
    """
    residual: np.ndarray, float [0., 255.], shape=[256, 256, 3]
    m: margin, xray margin size.
    """
    # residual = normalization(residual)
    # threshold
    # 寮卞寲杈圭紭
    # import pdb; pdb.set_trace()
    # residual_cliped = np.clip(residual-clip_min, a_min=0, a_max=255)

    gray = cv2.cvtColor(np.uint8(residual), cv2.COLOR_BGR2GRAY)

    ksize = m
    if ksize % 2 == 0:
        ksize = m+1
    gray_blured = cv2.GaussianBlur(gray, ksize=(ksize,ksize), sigmaX=3)

    ret, thresh = cv2.threshold(gray_blured, 0., 255, cv2.THRESH_BINARY)  # TODO: adaptive threshold
    contours, hierarchy = cv2.findContours(thresh, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None
    cnt = contours[0]
    for contour in contours:
        if contour.shape[0] > cnt.shape[0]:
            cnt = contour
    
    hull = cv2.convexHull(cnt)
    corners = np.expand_dims(hull, axis=0).astype(np.int32)
    mask = np.zeros_like(residual, dtype=np.uint8)
    cv2.fillPoly(mask, corners, (255,)*3)

    mask = inner_margin(mask, margin=m//2)
    polly = mask
    return polly


def get_xray(original_face, fake_face, m=20):
    # fake_face, original_face = fake_face / 255., original_face / 255.
    fake_face, original_face = fake_face * 1., original_face * 1.
    residual = fake_face - original_face
    residual = np.abs(residual)  # .mean(2, keepdims=True)
    polly = residual_to_convex_polly(residual, m=m)
    if polly is None:
        return None
    ksize = (m//2)*2 + 1
    xray = polly_to_xray(polly, ksize=ksize, sigma=ksize//3)
    
    # residual = np.clip(residual*50, a_min=0, a_max=1.0)
    # residual[residual > 0] = 1.0

    # residual = np.concatenate([residual]*3, axis=2)
    return xray


def main_landmark(lmPath, root, original_name, fake_names, num_samples=100):
    """
    lmPath: /root/landmarks/000.json
    root: /root/
    original_name: 'original'
    fake_names: ['Face2Face', 'Deepfakes', 'FaceSwap']
    """
    id_lm = parse_lamdmark(lmPath)
    _, lmName = osp.split(lmPath)
    indexes = choose_idx(id_lm, num_samples)
    video_index, _ = osp.splitext(lmName)
    video_path = osp.join(root, original_name, 'videos', video_index + '.mp4')

    original_faces = parse_video(video_path, id_lm, indexes)
    
    save_root = osp.join(root, original_name, 'calculated_xray', video_index)
    os.makedirs(save_root, exist_ok=True)
    for original_face, frame_index in zip(original_faces, indexes):
        original_face_path = osp.join(save_root, frame_index + '.png')
        status = cv2.imwrite(original_face_path, original_face)
        assert status, 'Error writing: {}'.format(original_face)


    for fake_name in fake_names:
        # Face2Face and FaceSwap video have smaller length than Deepfakes
        video_path_pattern = osp.join(root, fake_name, 'videos', video_index + '*.mp4')
        video_pathes = glob.glob(video_path_pattern)
        assert len(video_pathes) == 1, 'video_pathes: {}, video_path_pattern: {}'.format(video_pathes, video_path_pattern)
        video_path = video_pathes[0]
        fake_faces = parse_video(video_path, id_lm, indexes)
        print(fake_name, 'faces:', len(fake_faces))

        save_root = osp.join(root, fake_name, 'calculated_xray', video_index)
        os.makedirs(save_root, exist_ok=True)
        for original_face, fake_face, frame_index in zip(original_faces, fake_faces, indexes):
            xray_path = osp.join(save_root, frame_index + '_label.png')
            fake_face_path = osp.join(save_root, frame_index + '.png')
            xray = get_xray(original_face, fake_face, m=20)
            if xray is None:
                # import pdb; pdb.set_trace()
                print('Xray is None: ', fake_face_path)
                continue
                # import pdb; pdb.set_trace()
            # plt.imshow(xray); plt.show()
            
            status = 0
            status += cv2.imwrite(fake_face_path, fake_face)
            status += cv2.imwrite(xray_path, xray*255)
            assert status == 2, 'Error Writing: {}, {}'.format(xray_path, fake_face_path)
            # save path: /root/[fake_name]/aligned_xray/[video_index]/[frame_index].jpg


def main(root):
    """
    root: /root/[original, Face2Face, Deepfakes, FaceSwap]/videos/000.mp4
          /root/landmarks/000.json
          /root/[original, Face2Face, Deepfakes, FaceSwap]/aligned_xray/000/0000.jpg
    """
    landmark_paths = glob.glob(osp.join(root, 'landmarks', '*.json'))
    original_name = 'original_sequences'
    fake_names = [
        'manipulated_sequences/Deepfakes',
        'manipulated_sequences/Face2Face',
        'manipulated_sequences/FaceSwap'
    ]
    num_samples = 50
    start_num = 128
    
    print(original_name)
    print(fake_names)
    print(num_samples)
    for landmark_path in tqdm(landmark_paths[start_num:]):
        print(landmark_path)
        main_landmark(landmark_path, root, original_name, fake_names, num_samples)


if __name__ == '__main__':
    ROOT_PATH = '/mnt/hjr/FF++/'  # 'D:/Dataset/FF++'
    main(ROOT_PATH)
    # import pdb
    # pdb.set_trace()
    