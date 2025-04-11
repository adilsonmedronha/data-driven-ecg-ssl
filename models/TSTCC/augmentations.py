import torch
import numpy as np
from tqdm import tqdm

class ConfigAug(object):
    def __init__(self, max_seg, jitter_scale, jitter_ratio) -> None:
        self.jitter_scale_ratio = jitter_scale
        self.jitter_ratio = jitter_ratio
        self.max_seg = max_seg


def DataTransform(sample, config: ConfigAug):

    weak_aug = scaling(sample, config.jitter_scale_ratio)
    strong_aug = jitter(permutation(sample, max_segments=config.max_seg), config.jitter_ratio)

    return weak_aug, strong_aug


def jitter(x, sigma=0.8):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def scaling(x, sigma=1.1):
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[0], x.shape[2]))
    ai = []

    for i in tqdm(range(x.shape[1]), desc="scaling"):
        xi = x[:, i, :]
        ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])

    ret = np.concatenate((ai), axis=1)
    return torch.from_numpy(ret)


def permutation(x, max_segments=5, seg_mode="random"):
    orig_steps = np.arange(x.shape[2])

    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))

    ret = np.zeros_like(x)
    for i, pat in tqdm(enumerate(x), desc="permutation", total=len(x)):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            
            # ----------------------------------
            # [Rafael] Bug fix

            # warp = np.concatenate(np.random.permutation(splits)).ravel()
            # ----------------------------------
            orig_idx = np.arange(len(splits))
            perm_idx = np.random.permutation(orig_idx)

            perm_splits = [splits[idx] for idx in perm_idx]
            warp = np.concatenate(perm_splits).ravel()
            # ----------------------------------

            ret[i] = pat[0,warp]
        else:
            ret[i] = pat
    return torch.from_numpy(ret)

