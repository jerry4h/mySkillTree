import numpy as np
import os
import glob
import time
import cv2
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm
# See github.com/timesler/facenet-pytorch:
from facenet_pytorch import MTCNN

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'Running on device: {device}')

# Load face detector
mtcnn = MTCNN(image_size=256, margin=80, min_face_size=40, keep_all=True, factor=0.5, post_process=False, device=device).eval()


class DetectionPipeline:
    """Pipeline class for detecting faces in the frames of a video file."""
    
    def __init__(self, detector, n_frames=None, batch_size=32, resize=None):
        """Constructor for DetectionPipeline class.
        
        Keyword Arguments:
            n_frames {int} -- Total number of frames to load. These will be evenly spaced
                throughout the video. If not specified (i.e., None), all frames will be loaded.
                (default: {None})
            batch_size {int} -- Batch size to use with MTCNN face detector. (default: {32})
            resize {float} -- Fraction by which to resize frames from original prior to face
                detection. A value less than 1 results in downsampling and a value greater than
                1 result in upsampling. (default: {None})
        """
        self.detector = detector
        self.n_frames = n_frames
        self.batch_size = batch_size
        self.resize = resize
    
    def __call__(self, filename):
        """Load frames from an MP4 video and detect faces.

        Arguments:
            filename {str} -- Path to video.
        """
        # Create video reader and find length
        v_cap = cv2.VideoCapture(filename)
        v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Pick 'n_frames' evenly spaced frames to sample
        if self.n_frames is None:
            sample = np.arange(0, v_len)
        else:
            sample = np.linspace(0, v_len - 1, self.n_frames).astype(int)

        # Loop through frames
        faces = []
        frames = []
        for j in range(v_len):
            success = v_cap.grab()
            if j in sample:
                # Load frame
                success, frame = v_cap.retrieve()
                if not success:
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                
                # Resize frame to desired size
                if self.resize is not None:
                    frame = frame.resize([int(d * self.resize) for d in frame.size])
                frames.append(frame)

                # When batch is full, detect faces and reset frame list
                if len(frames) % self.batch_size == 0 or j == sample[-1]:
                    faces.extend(self.detector(frames))
                    frames = []

        v_cap.release()

        return faces    

# Define face detection pipeline
detection_pipeline = DetectionPipeline(detector=mtcnn, n_frames=100, batch_size=32, resize=None)
# Get all test videos
filenames = glob.glob('D:/Dataset/*.mp4')
outRoot = 'D:/Dataset/ai_pictures'

X = []
start = time.time()
n_processed = 0
with torch.no_grad():
    for i, filename in tqdm(enumerate(filenames), total=len(filenames)):
        try:
            # Load frames and find faces
            faces = detection_pipeline(filename)
            videoName = os.path.split(filename)[-1]
            faceOutPath = os.path.join(outRoot, videoName)
            # import pdb
            # pdb.set_trace()
            os.makedirs(faceOutPath + "/", exist_ok=True)

            for i in range(len(faces)):
                face = faces[i].cpu().clone()
                face = face.squeeze(0) / 255
                face = torchvision.transforms.ToPILImage()(face)
                face.save(os.path.join(faceOutPath, '{}.jpg'.format(i)))
            # plt.imshow(face)
            # plt.show()
            # plt.pause(0.001)
            # Calculate embeddings
#             X.append(process_faces(faces, resnet))

        except KeyboardInterrupt:
            print('\nStopped.')
            break

        except Exception as e:
            print(e)
            X.append(None)
        
        n_processed += len(faces)
        print(f'Frames per second (load+detect+embed): {n_processed / (time.time() - start):6.3}\r', end='')