import torch
import numpy as np


def img_read(img_path, device):
    import skimage.io
    x = skimage.io.imread(img_path)
    if len(x.shape) == 2:
        x = np.expand_dims(x, -1)
    gt = np.expand_dims(x, 0)
    gt = torch.tensor(gt).contiguous().permute(0,3,1,2).detach().to(device).float()

    return gt

def img_255_to_m11(x):
    return x.div(255).mul(2).add(-1)


def tensor2npimg(x, vmin=-1, vmax=1, normmaxmin=False, to_numpy=True):
    """tensor in [-1,1] (1x3xHxW) --> numpy image ready to plt.imshow"""
    if normmaxmin:
        vmin = x.min().item()
        vmax = x.max().item()
    final = x[0].add(-vmin).div(vmax-vmin).mul(255).add(0.5).clamp(0, 255)

    if to_numpy:
        final = final.permute(1, 2, 0)
        # if input has 1-channel, pass grayscale to numpy
        if final.shape[-1] == 1:
            final = final[:,:,0]
        return final.to('cpu', torch.uint8).numpy()
    else:
        return final.to('cpu', torch.uint8)
