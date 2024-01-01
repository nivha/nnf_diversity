__author__ = "Niv Haim (niv.haim@weizmann.ac.il)"
import os

import numpy as np
import torch
from matplotlib import animation
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from .image import img_read, img_255_to_m11
from common_utils import scale_utils


def read_frames(frames_dir, start_frame, end_frame, frame_resizer=None, device='cuda', verbose=True, ext='png'):
    frames = []
    for fi in tqdm(range(start_frame, end_frame + 1), disable=not verbose):
        frame_path = os.path.join(frames_dir, f'{fi}.{ext}')

        x = img_read(frame_path, device=device)
        if frame_resizer is not None:
            x = frame_resizer(x)

        frames.append(x[:, :3, :, :])

    return frames


def read_original_video(frames_dir, start_frame, end_frame, max_size=None, device='cuda', verbose=True, ext='png'):
    # read frames to video
    first_frame_path = os.path.join(frames_dir, f'{start_frame}.{ext}')
    resizer = None
    if max_size is not None:
        resizer = scale_utils.get_frame_resizer(first_frame_path, max_size=max_size, target_shape=None) if max_size is not None else None
    frames = read_frames(frames_dir, start_frame, end_frame, resizer, device=device, verbose=verbose, ext=ext)
    frames = [img_255_to_m11(f) for f in frames]
    orig_vid = torch.cat(frames, dim=0).permute(1, 0, 2, 3).unsqueeze(0)
    return orig_vid


###############################################################################
#                   HTML Video (for notebooks)                                #
###############################################################################
def html_vid(vid, interval=100):
    """
        vid: THWC (where C=3..)
        Use in jupyter:
        anim = html_vid(q_vid)
        from IPython.display import HTML
        HTML(anim.to_html5_video())
    """
    video = vid.detach().cpu().numpy()
    video = (video + 1) / 2
    video = np.clip(video, 0, 1)
    fig = plt.figure()
    im = plt.imshow(video[0, :, :, :])
    plt.axis('off')
    plt.close()  # this is required to not display the generated image

    def init():
        im.set_data(video[0, :, :, :])

    def animate(i):
        im.set_data(video[i, :, :, :])
        return im

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=video.shape[0], interval=interval)
    return anim