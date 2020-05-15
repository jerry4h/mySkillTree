import numpy as np
from numpy.linalg import inv, norm, lstsq
from numpy.linalg import matrix_rank as rank
import cv2


BIAS = -15
# reference facial points, a list of coordinates (x,y)
REFERENCE_FACIAL_POINTS = [        # default reference facial points for crop_size = (112, 112); should adjust REFERENCE_FACIAL_POINTS accordingly for other crop_size
    [30.29459953,  51.69630051+BIAS], 
    [65.53179932,  51.50139999+BIAS],
    [48.02519989,  71.73660278+BIAS],
    [33.54930115,  92.3655014+BIAS],
    [62.72990036,  92.20410156+BIAS]
]

DEFAULT_CROP_SIZE = (96, 112)



def tformfwd(trans, uv):
    """
    Function:
    ----------
        apply affine transform 'trans' to uv
    Parameters:
    ----------
        @trans: 3x3 np.array
            transform matrix
        @uv: Kx2 np.array
            each row is a pair of coordinates (x, y)
    Returns:
    ----------
        @xy: Kx2 np.array
            each row is a pair of transformed coordinates (x, y)
    """
    uv = np.hstack((
        uv, np.ones((uv.shape[0], 1))
    ))
    xy = np.dot(uv, trans)
    xy = xy[:, 0:-1]
    return xy


def tforminv(trans, uv):
    """
    Function:
    ----------
        apply the inverse of affine transform 'trans' to uv
    Parameters:
    ----------
        @trans: 3x3 np.array
            transform matrix
        @uv: Kx2 np.array
            each row is a pair of coordinates (x, y)
    Returns:
    ----------
        @xy: Kx2 np.array
            each row is a pair of inverse-transformed coordinates (x, y)
    """
    Tinv = inv(trans)
    xy = tformfwd(Tinv, uv)
    return xy


def findNonreflectiveSimilarity(uv, xy, options=None):

    options = {'K': 2}

    K = options['K']
    M = xy.shape[0]
    x = xy[:, 0].reshape((-1, 1))  # use reshape to keep a column vector
    y = xy[:, 1].reshape((-1, 1))  # use reshape to keep a column vector
    # print('--->x, y:\n', x, y

    tmp1 = np.hstack((x, y, np.ones((M, 1)), np.zeros((M, 1))))
    tmp2 = np.hstack((y, -x, np.zeros((M, 1)), np.ones((M, 1))))
    X = np.vstack((tmp1, tmp2))
    # print('--->X.shape: ', X.shape
    # print('X:\n', X

    u = uv[:, 0].reshape((-1, 1))  # use reshape to keep a column vector
    v = uv[:, 1].reshape((-1, 1))  # use reshape to keep a column vector
    U = np.vstack((u, v))
    # print('--->U.shape: ', U.shape
    # print('U:\n', U

    # We know that X * r = U
    if rank(X) >= 2 * K:
        r, _, _, _ = lstsq(X, U)
        r = np.squeeze(r)
    else:
        raise Exception("cp2tform: two Unique Points Req")

    # print('--->r:\n', r

    sc = r[0]
    ss = r[1]
    tx = r[2]
    ty = r[3]

    Tinv = np.array([
        [sc, -ss, 0],
        [ss,  sc, 0],
        [tx,  ty, 1]
    ])

    # print('--->Tinv:\n', Tinv

    T = inv(Tinv)
    # print('--->T:\n', T

    T[:, 2] = np.array([0, 0, 1])

    return T, Tinv


def findSimilarity(uv, xy, options=None):

    options = {'K': 2}

#    uv = np.array(uv)
#    xy = np.array(xy)

    # Solve for trans1
    trans1, trans1_inv = findNonreflectiveSimilarity(uv, xy, options)

    # Solve for trans2

    # manually reflect the xy data across the Y-axis
    xyR = xy
    xyR[:, 0] = -1 * xyR[:, 0]

    trans2r, trans2r_inv = findNonreflectiveSimilarity(uv, xyR, options)

    # manually reflect the tform to undo the reflection done on xyR
    TreflectY = np.array([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    trans2 = np.dot(trans2r, TreflectY)

    # Figure out if trans1 or trans2 is better
    xy1 = tformfwd(trans1, uv)
    norm1 = norm(xy1 - xy)

    xy2 = tformfwd(trans2, uv)
    norm2 = norm(xy2 - xy)

    if norm1 <= norm2:
        return trans1, trans1_inv
    else:
        trans2_inv = inv(trans2)
        return trans2, trans2_inv


def get_similarity_transform(src_pts, dst_pts, reflective = True):
    """
    Function:
    ----------
        Find Similarity Transform Matrix 'trans':
            u = src_pts[:, 0]
            v = src_pts[:, 1]
            x = dst_pts[:, 0]
            y = dst_pts[:, 1]
            [x, y, 1] = [u, v, 1] * trans
    Parameters:
    ----------
        @src_pts: Kx2 np.array
            source points, each row is a pair of coordinates (x, y)
        @dst_pts: Kx2 np.array
            destination points, each row is a pair of transformed
            coordinates (x, y)
        @reflective: True or False
            if True:
                use reflective similarity transform
            else:
                use non-reflective similarity transform
    Returns:
    ----------
       @trans: 3x3 np.array
            transform matrix from uv to xy
        trans_inv: 3x3 np.array
            inverse of trans, transform matrix from xy to uv
    """

    if reflective:
        trans, trans_inv = findSimilarity(src_pts, dst_pts)
    else:
        raise NotImplementedError(reflective)

    return trans, trans_inv


def cvt_tform_mat_for_cv2(trans):
    """
    Function:
    ----------
        Convert Transform Matrix 'trans' into 'cv2_trans' which could be
        directly used by cv2.warpAffine():
            u = src_pts[:, 0]
            v = src_pts[:, 1]
            x = dst_pts[:, 0]
            y = dst_pts[:, 1]
            [x, y].T = cv_trans * [u, v, 1].T
    Parameters:
    ----------
        @trans: 3x3 np.array
            transform matrix from uv to xy
    Returns:
    ----------
        @cv2_trans: 2x3 np.array
            transform matrix from src_pts to dst_pts, could be directly used
            for cv2.warpAffine()
    """
    cv2_trans = trans[:, 0:2].T

    return cv2_trans


def get_similarity_transform_for_cv2(src_pts, dst_pts, reflective = True):
    """
    Function:
    ----------
        Find Similarity Transform Matrix 'cv2_trans' which could be
        directly used by cv2.warpAffine():
            u = src_pts[:, 0]
            v = src_pts[:, 1]
            x = dst_pts[:, 0]
            y = dst_pts[:, 1]
            [x, y].T = cv_trans * [u, v, 1].T
    Parameters:
    ----------
        @src_pts: Kx2 np.array
            source points, each row is a pair of coordinates (x, y)
        @dst_pts: Kx2 np.array
            destination points, each row is a pair of transformed
            coordinates (x, y)
        reflective: True or False
            if True:
                use reflective similarity transform
            else:
                use non-reflective similarity transform
    Returns:
    ----------
        @cv2_trans: 2x3 np.array
            transform matrix from src_pts to dst_pts, could be directly used
            for cv2.warpAffine()
    """
    trans, trans_inv = get_similarity_transform(src_pts, dst_pts, reflective)
    cv2_trans = cvt_tform_mat_for_cv2(trans)

    return cv2_trans


def get_reference_facial_points(output_size = None,
                                inner_padding_factor = 0.0,
                                outer_padding=(0, 0),
                                default_square = False):
    """
    Function:
    ----------
        get reference 5 key points according to crop settings:
        0. Set default crop_size:
            if default_square: 
                crop_size = (112, 112)
            else: 
                crop_size = (96, 112)
        1. Pad the crop_size by inner_padding_factor in each side;
        2. Resize crop_size into (output_size - outer_padding*2),
            pad into output_size with outer_padding;
        3. Output reference_5point;
    Parameters:
    ----------
        @output_size: (w, h) or None
            size of aligned face image
        @inner_padding_factor: (w_factor, h_factor)
            padding factor for inner (w, h)
        @outer_padding: (w_pad, h_pad)
            each row is a pair of coordinates (x, y)
        @default_square: True or False
            if True:
                default crop_size = (112, 112)
            else:
                default crop_size = (96, 112);
        !!! make sure, if output_size is not None:
                (output_size - outer_padding) 
                = some_scale * (default crop_size * (1.0 + inner_padding_factor))
    Returns:
    ----------
        @reference_5point: 5x2 np.array
            each row is a pair of transformed coordinates (x, y)
    """
    #print('\n===> get_reference_facial_points():')

    #print('---> Params:')
    #print('            output_size: ', output_size)
    #print('            inner_padding_factor: ', inner_padding_factor)
    #print('            outer_padding:', outer_padding)
    #print('            default_square: ', default_square)
    tmp_5pts = np.array(REFERENCE_FACIAL_POINTS)
    
    tmp_crop_size = np.array(DEFAULT_CROP_SIZE)

    # 0) make the inner region a square
    if default_square:
        size_diff = max(tmp_crop_size) - tmp_crop_size
        tmp_5pts += size_diff / 2
        tmp_crop_size += size_diff

    #print('---> default:')
    #print('              crop_size = ', tmp_crop_size)
    #print('              reference_5pts = ', tmp_5pts)

    if (output_size and
            output_size[0] == tmp_crop_size[0] and
            output_size[1] == tmp_crop_size[1]):
        #print('output_size == DEFAULT_CROP_SIZE {}: return default reference points'.format(tmp_crop_size))
        return tmp_5pts

    if (inner_padding_factor == 0 and
            outer_padding == (0, 0)):
        if output_size is None:
            #print('No paddings to do: return default reference points')
            return tmp_5pts
        else:
            raise Exception(
                'No paddings to do, output_size must be None or {}'.format(tmp_crop_size))

    # check output size
    if not (0 <= inner_padding_factor <= 1.0):
        raise Exception('Not (0 <= inner_padding_factor <= 1.0)')

    if ((inner_padding_factor > 0 or outer_padding[0] > 0 or outer_padding[1] > 0)
            and output_size is None):
        output_size = tmp_crop_size * \
            (1 + inner_padding_factor * 2).astype(np.int32)
        output_size += np.array(outer_padding)
        #print('              deduced from paddings, output_size = ', output_size)

    if not (outer_padding[0] < output_size[0]
            and outer_padding[1] < output_size[1]):
        raise Exception('Not (outer_padding[0] < output_size[0]'
                                'and outer_padding[1] < output_size[1])')

    # 1) pad the inner region according inner_padding_factor
    #print('---> STEP1: pad the inner region according inner_padding_factor')
    if inner_padding_factor > 0:
        size_diff = tmp_crop_size * inner_padding_factor * 2
        tmp_5pts += size_diff / 2
        tmp_crop_size += np.round(size_diff).astype(np.int32)

    #print('              crop_size = ', tmp_crop_size)
    #print('              reference_5pts = ', tmp_5pts)

    # 2) resize the padded inner region
    #print('---> STEP2: resize the padded inner region')
    size_bf_outer_pad = np.array(output_size) - np.array(outer_padding) * 2
    #print('              crop_size = ', tmp_crop_size)
    #print('              size_bf_outer_pad = ', size_bf_outer_pad)

    if size_bf_outer_pad[0] * tmp_crop_size[1] != size_bf_outer_pad[1] * tmp_crop_size[0]:
        raise Exception('Must have (output_size - outer_padding)'
                                '= some_scale * (crop_size * (1.0 + inner_padding_factor)')

    scale_factor = size_bf_outer_pad[0].astype(np.float32) / tmp_crop_size[0]
    #print('              resize scale_factor = ', scale_factor)
    tmp_5pts = tmp_5pts * scale_factor
#    size_diff = tmp_crop_size * (scale_factor - min(scale_factor))
#    tmp_5pts = tmp_5pts + size_diff / 2
    tmp_crop_size = size_bf_outer_pad
    #print('              crop_size = ', tmp_crop_size)
    #print('              reference_5pts = ', tmp_5pts)

    # 3) add outer_padding to make output_size
    reference_5point = tmp_5pts + np.array(outer_padding)
    tmp_crop_size = output_size
    #print('---> STEP3: add outer_padding to make output_size')
    #print('              crop_size = ', tmp_crop_size)
    #print('              reference_5pts = ', tmp_5pts)

    #print('===> end get_reference_facial_points\n')

    return reference_5point


def warp_and_crop_face(src_img,
                       facial_pts,
                       reference_pts = None,
                       crop_size=(96, 112),
                       align_type = 'similarity'):
    """
    Function:
    ----------
        apply affine transform 'trans' to uv
    Parameters:
    ----------
        @src_img: 3x3 np.array
            input image
        @facial_pts: could be
            1)a list of K coordinates (x,y)
        or
            2) Kx2 or 2xK np.array
            each row or col is a pair of coordinates (x, y)
        @reference_pts: could be
            1) a list of K coordinates (x,y)
        or
            2) Kx2 or 2xK np.array
            each row or col is a pair of coordinates (x, y)
        or
            3) None
            if None, use default reference facial points
        @crop_size: (w, h)
            output face image size
        @align_type: transform type, could be one of
            1) 'similarity': use similarity transform
            2) 'cv2_affine': use the first 3 points to do affine transform,
                    by calling cv2.getAffineTransform()
            3) 'affine': use all points to do affine transform
    Returns:
    ----------
        @face_img: output face image with size (w, h) = @crop_size
    """

    if reference_pts is None:
        if crop_size[0] == 96 and crop_size[1] == 112:
            reference_pts = REFERENCE_FACIAL_POINTS
        else:
            default_square = False
            inner_padding_factor = 0
            outer_padding = (0, 0)
            output_size = crop_size

            reference_pts = get_reference_facial_points(output_size,
                                                        inner_padding_factor,
                                                        outer_padding,
                                                        default_square)

    ref_pts = np.float32(reference_pts)
    ref_pts_shp = ref_pts.shape
    if max(ref_pts_shp) < 3 or min(ref_pts_shp) != 2:
        raise Exception(
            'reference_pts.shape must be (K,2) or (2,K) and K>2')

    if ref_pts_shp[0] == 2:
        ref_pts = ref_pts.T

    src_pts = np.float32(facial_pts)
    src_pts_shp = src_pts.shape
    if max(src_pts_shp) < 3 or min(src_pts_shp) != 2:
        raise Exception(
            'facial_pts.shape must be (K,2) or (2,K) and K>2')

    if src_pts_shp[0] == 2:
        src_pts = src_pts.T

#    #print('--->src_pts:\n', src_pts
#    #print('--->ref_pts\n', ref_pts

    if src_pts.shape != ref_pts.shape:
        raise Exception(
            'facial_pts and reference_pts must have the same shape')
    if align_type == 'similarity':
        tfm = get_similarity_transform_for_cv2(src_pts, ref_pts)
#            #print(('get_similarity_transform_for_cv2() returns tfm=\n' + str(tfm))
    else:
        raise NotImplementedError(align_type)

#    #print('--->Transform matrix: '
#    #print(('type(tfm):' + str(type(tfm)))
#    #print(('tfm.dtype:' + str(tfm.dtype))
#    #print( tfm

    face_img = cv2.warpAffine(src_img, tfm, (crop_size[0], crop_size[1]))

    return face_img


def get_size(img):
    if isinstance(img, np.ndarray):
        return img.shape[1::-1]
    else:
        return img.size


def crop_resize(img, box, image_size):
    if isinstance(img, np.ndarray):
        out = cv2.resize(
            img[box[1]:box[3], box[0]:box[2]],
            (image_size, image_size),
            interpolation=cv2.INTER_AREA
        ).copy()
    else:
        raise NotImplementedError('img type: {}'.format(type(img)))
        # out = img.crop(box).copy().resize((image_size, image_size), Image.BILINEAR)
    return out


def resize_box(box, image_size=160):
    """Resize box to fixed scale rate.

    Arguments:
        box {numpy.ndarray} -- left, up, right, bottom
        image_size {int} -- image_size
    """
    if not isinstance(image_size, int):
        raise NotImplementedError
    centers = [(box[0]+box[2])/2., (box[1]+box[3])/2.]
    l = (box[2] + box[3] - box[0] - box[1])/2.
    box[0] = centers[0] - l/2.
    box[1] = centers[1] - l/2.
    box[2] = centers[0] + l/2.
    box[3] = centers[1] + l/2.
    return box
    
def landmark_to_bbox(point, scale_face_to_bbox=2.0):
    """
    point：[[x0,y0], ..., [x4, y4]]
    0   1
      2
    3   4
    scale_face_to_bbox: bbox length/landmark_length scale
    return: bbox [left, up, right, bottom]
    """

    center_x, center_y = point[2]
    width = max(point[4][0], point[1][0]) - min(point[0][0], point[3][0])
    height = max(point[4][1], point[3][1]) - min(point[1][1], point[0][1])
    
    l = max(width, height)
    l_half = scale_face_to_bbox * l / 2.
    
    left, right = center_x - l_half, center_x + l_half
    up, bottom = center_y - l_half, center_y + l_half
    bbox = [left, up, right, bottom]
    # import pdb
    # pdb.set_trace()
    return bbox



class Alignmenter:
    def __init__(self, crop_size=112, margin=0):
        self.params = {
            'reference': get_reference_facial_points(default_square=True),
            'params': [112, 0]
        }
        self.exam_params(crop_size, margin)
    
    def exam_params(self, crop_size=112, margin=0):
        if not self.params['params'] == [crop_size, margin]:
            self.params['params'] = [crop_size, margin]
            self.params['reference'] = get_reference_facial_points(
                output_size = None,
                inner_padding_factor = np.array([0.15*margin/crop_size]),
                outer_padding=(0, 0),
                default_square = True
            ) * crop_size/112.
        return

    def detect(self, img, box, crop_size=160, margin=0, adapt_size=False):
        """Extract face + margin from PIL Image given bounding box.
    
        Arguments:
            img {PIL.Image} -- A PIL Image.
            box {numpy.ndarray} -- Four-element bounding box.
            crop_size {int} -- Output image size in pixels. The image will be square.
            margin {int} -- Margin to add to bounding box, in terms of pixels in the final image. 
                Note that the application of the margin differs slightly from the davidsandberg/facenet
                repo, which applies the margin to the original image before resizing, making the margin
                dependent on the original image size.
            save_path {str} -- Save path for extracted face image. (default: {None})
            adapt_size {bool} -- whether to resize box to image_size, avoiding stretch with different
                scale for height/ weight dimension 
    
        Returns:
            torch.tensor -- tensor representing the extracted face.
        """
        if adapt_size:
            box = resize_box(box)
    
        margin = [
            margin * (box[2] - box[0]) / (crop_size - margin),
            margin * (box[3] - box[1]) / (crop_size - margin),
        ]
        raw_image_size = get_size(img)
        box = [
            int(max(box[0] - margin[0] / 2, 0)),
            int(max(box[1] - margin[1] / 2, 0)),
            int(min(box[2] + margin[0] / 2, raw_image_size[0])),
            int(min(box[3] + margin[1] / 2, raw_image_size[1])),
        ]

        face = crop_resize(img, box, crop_size)

        return face

    def align(self, img, points, crop_size=112, margin=0):
        self.exam_params(crop_size, margin)

        facial5points = [[points[0][j], points[0][j + 5]] for j in range(5)]
        # print('facial5points\n', facial5points)
        warped_face = warp_and_crop_face(
            np.array(img),
            facial5points,
            self.params['reference'],
            crop_size=(crop_size, crop_size)
        )
        # print('warped_face\n', warped_face)
        return warped_face

    def __call__(self, img, point, image_size=160, margin=0, align=True):
        
        """Extract face + margin from PIL Image given bounding box.
    
        Arguments:
            img {PIL.Image} -- A PIL Image.
            box {numpy.ndarray} -- Four-element bounding box.
            image_size {int} -- Output image size in pixels. The image will be square.
            margin {int} -- Margin to add to bounding box, in terms of pixels in the final image. 
                Note that the application of the margin differs slightly from the davidsandberg/facenet
                repo, which applies the margin to the original image before resizing, making the margin
                dependent on the original image size.
            save_path {str} -- Save path for extracted face image. (default: {None})
            adapt_size {bool} -- whether to resize box to image_size, avoiding stretch with different
                scale for height/ weight dimension 
    
        Returns:
            torch.tensor -- tensor representing the extracted face.
        """

        if align:
            point_flattened = np.transpose(point, (1,0)).reshape(1, -1)
            face = self.align(img, point_flattened, crop_size=image_size, margin=margin)
        else:
            bbox = landmark_to_bbox(point)
            face = self.detect(img, bbox, crop_size=image_size, margin=margin)

        return face
