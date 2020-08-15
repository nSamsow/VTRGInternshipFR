import warnings
import cv2
import numpy as np
from numba import jit
from tensorflow.keras import backend as K

V1_LABELS_PATH = 'https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_labels_v1.npy'
V2_LABELS_PATH = 'https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_labels_v2.npy'
VGG16_WEIGHTS_PATH = 'https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_tf_vgg16.h5'
VGG16_WEIGHTS_PATH_NO_TOP = 'https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_tf_notop_vgg16.h5'
RESNET50_WEIGHTS_PATH = 'https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_tf_resnet50.h5'
RESNET50_WEIGHTS_PATH_NO_TOP = 'https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_tf_notop_resnet50.h5'
SENET50_WEIGHTS_PATH = 'https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_tf_senet50.h5'
SENET50_WEIGHTS_PATH_NO_TOP = 'https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_tf_notop_senet50.h5'
VGGFACE_DIR = 'models/vggface'


@jit(nopython=True)
def cosine_dissimilarity(u, v):
    assert (u.shape[0] == v.shape[0])
    uv = np.dot(u, v)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u != 0 and norm_v != 0:
        cos_theta = np.subtract(1., np.divide(uv, np.multiply(norm_u, norm_v)))
    else:
        cos_theta = 0.
    return cos_theta


@jit(nopython=True)
def l2_distance(u, v):
    assert (u.shape[0] == v.shape[0])
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    diff = np.subtract(np.divide(u, norm_u), np.divide(v, norm_v))
    return np.sum(np.square(diff))


def is_match(known_emb, new_emb, thresh, metric='cosine'):
    if metric == 'cosine':
        distance = cosine_dissimilarity(known_emb, new_emb)
    elif metric == 'l2_distance':
        distance = l2_distance(known_emb, new_emb)
    else:
        raise Exception('Unknown distance')

    if distance <= thresh:
        match = 1
    else:
        match = 0
    return match, distance


def preprocess_input(x, version, data_format=None):
    x_temp = np.copy(x)
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}

    if version == 1:
        if data_format == 'channels_first':
            x_temp = x_temp[:, ::-1, ...]
            x_temp[:, 0, :, :] -= 93.5940
            x_temp[:, 1, :, :] -= 104.7624
            x_temp[:, 2, :, :] -= 129.1863
        else:
            x_temp = x_temp[..., ::-1]
            x_temp[..., 0] -= 93.5940
            x_temp[..., 1] -= 104.7624
            x_temp[..., 2] -= 129.1863

    elif version == 2:
        if data_format == 'channels_first':
            x_temp = x_temp[:, ::-1, ...]
            x_temp[:, 0, :, :] -= 91.4953
            x_temp[:, 1, :, :] -= 103.8827
            x_temp[:, 2, :, :] -= 131.0912
        else:
            x_temp = x_temp[..., ::-1]
            x_temp[..., 0] -= 91.4953
            x_temp[..., 1] -= 103.8827
            x_temp[..., 2] -= 131.0912
    else:
        raise NotImplementedError

    return x_temp


def generate_bbox(x, y, w, h):
    r = max(w, h) / 2
    nx = int(x + (w / 2) - r)
    ny = int(y + (h / 2) - r)
    nr = int(r * 2)
    return r, nx, ny, nr


def find_most_sim_face(face_names, known_emb, emb, metric, threshold):
    name = None
    score = np.inf
    for i in range(len(face_names)):
        match, distance = is_match(known_emb[i], emb, threshold, metric)
        if match:
            if distance < score:
                score = distance
                name = face_names[i]
    return name, score


def face_n_text_box(raw_img, x, y, w, h, text, color, font=cv2.FONT_HERSHEY_DUPLEX, font_scale=0.5, y_margin=0.25):
    (text_w, text_h) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
    box_coor = ((x - 1, y), (int(x + text_w + 2), int(y - text_h * (1 + y_margin * 2))))
    # face box
    cv2.rectangle(raw_img, (x, y), ((x + w), (y + h)), color, 2)
    # text box
    cv2.rectangle(raw_img, box_coor[0], box_coor[1], color, cv2.FILLED)
    cv2.putText(raw_img, text, (x, int(y - text_h * y_margin)), font, font_scale, (0, 0, 0), 1)


def unrecognized_box_n_text(raw_img, x, y, w, h):
    color = (127, 127, 255)
    text = 'Unknown'
    face_n_text_box(raw_img, x, y, w, h, text, color)
    # (text_w, text_h) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
    # box_coords = ((x - 1, y), (int((x + text_w + 2)), int((y - text_h * (1 + y_margin * 2)))))
    # # face box
    # cv2.rectangle(raw_img, (x, y), ((x + w), (y + h)), color, 2)
    # # text box
    # cv2.rectangle(raw_img, box_coords[0], box_coords[1], color, cv2.FILLED)
    # cv2.putText(raw_img, text, (x, int(y - text_h * y_margin)), font, font_scale, (0, 0, 0), 1)


def recognized_box_n_text(raw_img, name, score, x, y, w, h):
    color = (127, 255, 127)
    text = '%s-%.2f' % (name, score)
    face_n_text_box(raw_img, x, y, w, h, text, color)
    # (text_w, text_h) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
    # box_coords = ((x - 1, y), (int((x + text_w + 2)), int((y - text_h * (1 + y_margin * 2)))))
    # # face box
    # cv2.rectangle(raw_img, (x, y), ((x + w), (y + h)), color, 2)
    # # text box
    # cv2.rectangle(raw_img, box_coords[0], box_coords[1], color, cv2.FILLED)
    # cv2.putText(raw_img, text, (x, int(y - text_h * y_margin)), font, font_scale, (0, 0, 0), 1)


def show_proc_time(raw_img, proc_time, detection_time, match_time):
    font, font_scale, color = cv2.FONT_HERSHEY_DUPLEX, 0.5, (178, 255, 102)
    proc_time_text = 'proc. time: {} ms, detection time: {} ms, match time: {} ms'.format(
        round(sum(proc_time) / len(proc_time) * 1000),
        round(sum(detection_time) / len(detection_time) * 1000, 1),
        round(sum(match_time) / len(match_time) * 1000)
    )
    (proc_time_text_w, proc_time_text_h) = cv2.getTextSize(proc_time_text, font, fontScale=font_scale, thickness=1)[0]
    box_coor = ((0, 0), (int(proc_time_text_w + 5), int(proc_time_text_h + 5)))
    cv2.rectangle(raw_img, box_coor[0], box_coor[1], color, cv2.FILLED)
    cv2.putText(raw_img, proc_time_text, (int(0 + 5), int(0 + proc_time_text_h)), font, font_scale, (0, 0, 0), 1)


def _obtain_input_shape(input_shape,
                        default_size,
                        min_size,
                        data_format,
                        require_flatten,
                        weights=None):
    """Internal utility to compute/validate a model's tensor shape.
    # Arguments
        input_shape: Either None (will return the default network input shape),
            or a user-provided shape to be validated.
        default_size: Default input width/height for the model.
        min_size: Minimum input width/height accepted by the model.
        data_format: Image data format to use.
        require_flatten: Whether the model is expected to
            be linked to a classifier via a Flatten layer.
        weights: One of `None` (random initialization)
            or 'imagenet' (pre-training on ImageNet).
            If weights='imagenet' input channels must be equal to 3.
    # Returns
        An integer shape tuple (may include None entries).
    # Raises
        ValueError: In case of invalid argument values.
    """
    if weights != 'imagenet' and input_shape and len(input_shape) == 3:
        if data_format == 'channels_first':
            if input_shape[0] not in {1, 3}:
                warnings.warn(
                    'This model usually expects 1 or 3 input channels. '
                    'However, it was passed an input_shape with {input_shape}'
                    ' input channels.'.format(input_shape=input_shape[0]))
            default_shape = (input_shape[0], default_size, default_size)
        else:
            if input_shape[-1] not in {1, 3}:
                warnings.warn(
                    'This model usually expects 1 or 3 input channels. '
                    'However, it was passed an input_shape with {n_input_channels}'
                    ' input channels.'.format(n_input_channels=input_shape[-1]))
            default_shape = (default_size, default_size, input_shape[-1])
    else:
        if data_format == 'channels_first':
            default_shape = (3, default_size, default_size)
        else:
            default_shape = (default_size, default_size, 3)
    if weights == 'imagenet' and require_flatten:
        if input_shape is not None:
            if input_shape != default_shape:
                raise ValueError('When setting `include_top=True` '
                                 'and loading `imagenet` weights, '
                                 '`input_shape` should be {default_shape}.'.format(default_shape=default_shape))
        return default_shape
    if input_shape:
        if data_format == 'channels_first':
            if input_shape is not None:
                if len(input_shape) != 3:
                    raise ValueError(
                        '`input_shape` must be a tuple of three integers.')
                if input_shape[0] != 3 and weights == 'imagenet':
                    raise ValueError('The input must have 3 channels; got '
                                     '`input_shape={input_shape}`'.format(input_shape=input_shape))
                if ((input_shape[1] is not None and input_shape[1] < min_size) or
                    (input_shape[2] is not None and input_shape[2] < min_size)):
                    raise ValueError('Input size must be at least {min_size}x{min_size};'
                                     ' got `input_shape={input_shape}`'.format(min_size=min_size,
                                                                               input_shape=input_shape))
        else:
            if input_shape is not None:
                if len(input_shape) != 3:
                    raise ValueError(
                        '`input_shape` must be a tuple of three integers.')
                if input_shape[-1] != 3 and weights == 'imagenet':
                    raise ValueError('The input must have 3 channels; got '
                                     '`input_shape={input_shape}`'.format(input_shape=input_shape))
                if ((input_shape[0] is not None and input_shape[0] < min_size) or
                    (input_shape[1] is not None and input_shape[1] < min_size)):
                    raise ValueError('Input size must be at least {min_size}x{min_size};'
                                     ' got `input_shape={input_shape}`'.format(min_size=min_size,
                                                                               input_shape=input_shape))
    else:
        if require_flatten:
            input_shape = default_shape
        else:
            if data_format == 'channels_first':
                input_shape = (3, None, None)
            else:
                input_shape = (None, None, 3)
    if require_flatten:
        if None in input_shape:
            raise ValueError('If `include_top` is True, '
                             'you should specify a static `input_shape`. '
                             'Got `input_shape={input_shape}`'.format(input_shape=input_shape))
    return input_shape
