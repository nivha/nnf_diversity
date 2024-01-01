__author__ = "Niv Haim (niv.haim@weizmann.ac.il)"

import torch
import zlib

from common_utils.resize_right import resize
from nnf_solvers import compute_nnf_patchmatch


def get_out_shape(scale, shapes):
    """shapes is [T,H,W] or [H,W]"""
    assert len(shapes) in [2,3]
    out_shapes_float = scale * torch.tensor(shapes)
    int_idxs = (out_shapes_float - out_shapes_float.round()).abs() < 1e-5
    out_shapes = out_shapes_float.ceil()
    out_shapes[int_idxs] = out_shapes_float[int_idxs].round()
    out_shapes = out_shapes.long().tolist()
    return out_shapes


def resize_by_target_size(vid, target_shape):
    ndim = vid.dim()
    if ndim == 4:  # image
        n, c, h, w = vid.shape
        scale = target_shape / min(h, w)
        H_, W_ = get_out_shape(scale, [h, w])
        return resize(vid, scale_factors=scale, out_shape=[H_, W_])
    elif ndim == 5:  # video
        n, c, t, h, w = vid.shape
        scale = target_shape / min(h, w)
        T_, H_, W_ = get_out_shape(scale, [t, h, w])
        return resize(vid, scale_factors=scale, out_shape=[T_, H_, W_])
    else:
        raise Exception('unknown ndim in resize by target')


def get_nnf(q, k, ks=(3,3,3), target_shape=None, nnf_solver='faiss', verbose=True, patchmatch_steps=None):
    """ Compute nnf from q to k """
    # Downscale videos to work in a coarse scale
    if target_shape is None:
        a, b = q, k
    else:
        a = resize_by_target_size(q, target_shape)
        b = resize_by_target_size(k, target_shape)

    # Compute NNF
    if nnf_solver == 'faiss2d':
        from nnf_solvers import compute_nnf_faiss_2d
        nnf, dist = compute_nnf_faiss_2d(a, b, ks=ks)
    elif nnf_solver == 'faiss3d':
        from nnf_solvers import compute_nnf_faiss_3d
        nnf, dist = compute_nnf_faiss_3d(a, b, ks=ks)
    elif nnf_solver == 'patchmatch':
        nnf, dist = compute_nnf_patchmatch(a, b, ks=ks, steps=patchmatch_steps)
    else:
        raise Exception(f'Unknown NNF solver: {nnf_solver}')

    if verbose: print('a,b,nnf:', a.shape, b.shape, nnf.shape, dist.shape)
    return nnf, a, b, dist


def zlib_score(x):
    x = x.to('cpu', torch.uint8).contiguous().numpy()
    return len(zlib.compress(x)) / x.size


def get_nnfdiv(vid_a, vid_b, patchmatch_iterations=10):
    """
        Implementation of NNFDIV from the paper "SinFusion: Training Diffusion Models on a Single Image or Video"
        (see Section 7.2 of https://arxiv.org/abc/2211.11743)

        @param vid_a: a video of shape NCTHW
        @param vid_b: a video of shape NCTHW
        @param patchmatch_iterations: this is just a heuristic for PatchMatch. PatchMatch comes up with approximate NNF.
                                      The larger this param is, the more accurate the NNF be, but the run time longer.
        @return: NNFDIV

        Compute the NNF from video_a to video_b, and return its NNFDIV score
    """
    nnf, a, b, dist = get_nnf(vid_a, vid_b, ks=(3, 3, 3), nnf_solver='patchmatch', verbose=False,
                              patchmatch_steps=[16, 8, 4, 2, 1, 1, 1, 1, 1, 1, 4, 2, 1, 1, 1] * patchmatch_iterations + [1] * patchmatch_iterations,
                              target_shape=None)

    # the original code in the paper may have used the following for the computation of the NNFDIV (normalizing to 0,1):
    # vv = max(nnf.shape[2:])
    # nnf_vid_np = common_utils.image.tensor2npimg(nnf, vmin=-vv, vmax=vv, normmaxmin=False, to_numpy=False)
    return nnf, zlib_score(nnf)
