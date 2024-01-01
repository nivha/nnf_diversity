import numpy as np
from functools import partial

from .resize_right import resize
from .image import img_read


def round2even(x):
    if x % 2 != 0:
        return x + 1
    else:
        return x


def get_out_shapes(T, H, W, st, sh, sw):
    T_ = round2even(np.ceil(T * st).astype(int))
    H_ = round2even(np.ceil(H * sh).astype(int))
    W_ = round2even(np.ceil(W * sw).astype(int))
    return (T_, H_, W_)


def get_frame_resizer(first_frame_path, max_size=None, target_shape=None):
    first_frame = img_read(first_frame_path, device='cpu')
    _, C, H, W = first_frame.shape

    if max_size is not None:
        scale1 = min(max_size / H, 1)
        H_ = round2even(np.ceil(H * scale1).astype(int))
        W_ = round2even(np.ceil(W * scale1).astype(int))
        return partial(resize, scale_factors=scale1, out_shape=[H_, W_])
    elif target_shape is not None:
        scale1 = min(max(target_shape) / H, 1)
        return partial(resize, scale_factors=scale1, out_shape=target_shape)