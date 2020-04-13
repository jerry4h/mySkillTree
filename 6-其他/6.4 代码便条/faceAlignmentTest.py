import face_alignment
from skimage import io
import pdb

# cuda for CUDA
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,
 device='cpu', flip_input=False, face_detector='dlib')
imgPath = 'D:\\BaiduNetdiskDownload\\webface_align_112.tar\\webface_align_112\\0000145\\020.jpg'
input = io.imread(imgPath)
preds = fa.get_landmarks(input)
pdb.set_trace()
