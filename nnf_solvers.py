import torch
from pnn.models import PMPNN3D
from pnn import patch_match


### PatchMatch
def compute_nnf_patchmatch(q, k, ks, steps=None):
    """ks (T,H,W) or (H,W)"""
    q_patches = patch_match.tensor2patchesnd(q, kernel_size=ks, stride=1, use_padding=False)
    k_patches = patch_match.tensor2patchesnd(k, kernel_size=ks, stride=1, use_padding=False)

    if steps is None:
        steps = [8, 4, 2, 1] * 50 + [1] * 50

    pmpnn3d = PMPNN3D(kernel_size=ks, steps=steps)
    index, dist, nnf, orig, hist = pmpnn3d.qk_lookup(q_patches, k_patches, compute_dist=True)
    return nnf, dist


# ### FAISS
#
# import faiss
# # the following import is needed in order for faiss/pytorch integration
# import faiss.contrib.torch_utils
#
# def unravel_index(x, shape):
#     h, w = shape
#     #     return torch.cat([(x//w), (x%w)], dim=1)
#     return torch.cat([x.div(w, rounding_mode='trunc'), (x % w)], dim=1)
#
#
# def query_faiss(query, keys, k_nn=1):
#     assert query.is_contiguous()
#     assert keys.is_contiguous()
#     assert query.type() == 'torch.cuda.FloatTensor', query.type()
#     assert keys.type() == 'torch.cuda.FloatTensor', keys.type()
#
#     N, D = keys.shape
#     res = faiss.StandardGpuResources()
#     index = faiss.GpuIndexFlatL2(res, D)
#     index.add(keys)
#
#     D, I = index.search(query, k_nn)
#     return D, I
#
#
# def compute_nnf_faiss_2d(q, k, ks):
#     """ks (H,W)"""
#     q_patches = patch_match.tensor2patchesnd(q, kernel_size=ks, stride=1, use_padding=False)
#     k_patches = patch_match.tensor2patchesnd(k, kernel_size=ks, stride=1, use_padding=False)
#
#     # flatten q/k patches (faiss works on [N,D] inputs)
#     n, h, w, c, kh, kw = q_patches.shape
#     q_flat = q_patches.reshape(n * h * w, c * kh * kw).contiguous()
#     n, h, w, c, kh, kw = k_patches.shape
#     k_flat = k_patches.reshape(n * h * w, c * kh * kw).contiguous()
#     # query faiss
#     D, I = query_faiss(q_flat, k_flat)
#     # reshape I,D back to image
#     I = unravel_index(I, shape=(h, w)).view(h, w, 2).permute(2, 0, 1)
#     D = D.view(h, w, 1).permute(2, 0, 1)
#     # compute NNF from I
#     grid = torch.stack(torch.meshgrid([torch.arange(h), torch.arange(w)])).to(q.device)
#     nnf = I - grid
#     return nnf.unsqueeze(0), D
#
#
# def compute_nnf_faiss_3d(q, k, ks):
#     """ks (T,H,W)"""
#     q_patches = patch_match.tensor2patchesnd(q, kernel_size=ks, stride=1, use_padding=False)
#     k_patches = patch_match.tensor2patchesnd(k, kernel_size=ks, stride=1, use_padding=False)
#
#     # flatten q/k patches (faiss works on [N,D] inputs)
#     n, t, h, w, c, kt, kh, kw = q_patches.shape
#     q_flat = q_patches.reshape(n * t * h * w, c * kt * kh * kw).contiguous()
#     n, t, h, w, c, kt, kh, kw = k_patches.shape
#     k_flat = k_patches.reshape(n * t * h * w, c * kt * kh * kw).contiguous()
#     # query faiss
#     D, I = query_faiss(q_flat, k_flat)
#     # reshape I,D back to image
#     I = unravel_index(I, shape=(t, h, w)).view(t, h, w, 2).permute(2, 0, 1)
#     D = D.view(t, h, w, 1).permute(2, 0, 1)
#     # compute NNF from I
#     grid = torch.stack(torch.meshgrid([torch.arange(t), torch.arange(h), torch.arange(w)])).to(q.device)
#     nnf = I - grid
#     return nnf.unsqueeze(0), D


