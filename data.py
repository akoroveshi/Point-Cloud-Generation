"""ImageNet-1k data: a preprocessed uint8 cache (made by prepare_data.py) + dataset helpers."""
import numpy as np
import json
import torch
from torch.utils.data import Dataset

NUM_CLASSES = 1000

def center_crop_arr(pil_image, image_size):
    """ADM-style preprocessing (as used by DiT/ADM on ImageNet).
       Downsizes the image in halves, resizes and takes the centre crop."""
    
    from PIL import Image
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
        
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)
    arr = np.array(pil_image)
    cy, cx = (arr.shape[0] - image_size) // 2, (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[cy:cy + image_size, cx:cx + image_size])


def cache_paths(cache_dir, split, img_size):
    base = f"{cache_dir}/{split}_{img_size}"
    return base + "_imgs.npy", base + "_labels.npy", base + "_meta.json"
 
 
class CachedImageNet(Dataset):
    """Reads (N, S, S, 3) uint8 images from a memory-mapped .npy. Returns floats in [-1, 1]."""
 
    def __init__(self, cache_dir, split, img_size, flip=False):
        self.img_path, lab_path, meta_path = cache_paths(cache_dir, split, img_size)
        self.n = json.load(open(meta_path))["n"]
        self.labels = np.load(lab_path)[: self.n].astype(np.int64)
        self.flip = flip
        self.imgs = None  # opened lazily so each dataloader worker gets its own memmap
 
    def __len__(self):
        return self.n
    
    def __len__(self):
        return self.n
 
    def __getitem__(self, i):
        if self.imgs is None:
            self.imgs = np.load(self.img_path, mmap_mode="r")
        x = torch.from_numpy(np.array(self.imgs[i])).permute(2, 0, 1).float() / 127.5 - 1.0
        if self.flip and torch.rand(1).item() < 0.5:
            x = x.flip(-1)
        return x, int(self.labels[i])


def build_dataset(cache_dir, split, img_size, flip=True):
    return CachedImageNet(cache_dir, split, img_size, flip)


def build_overfit_batch(ds, n, device):
    """First image of n classes spread evenly over the 1000 ImageNet classes (cached on device)."""
    classes = np.linspace(0, NUM_CLASSES - 1, n).round().astype(int)
    idx = [int(np.argmax(ds.labels == c)) for c in classes]
    items = [ds[i] for i in idx]
    X = torch.stack([a for a, _ in items]).to(device)
    Y = torch.tensor([b for _, b in items], device=device)
    return X, Y