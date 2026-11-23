import random
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

def read_rgb(path):
    with Image.open(path) as image:
        if image.mode != "RGB":
            raise ValueError(f"Expected RGB image: {path} ({image.mode})")
        return np.array(image)

class PairedLowLightDataset(Dataset):
    def __init__(self, root, rows, patch=None, cache=False):
        self.root, self.rows, self.patch = Path(root), rows, patch
        self.cache = {} if cache else None
        self.epoch, self.seed = 0, 0
        if patch and any(min(row["height"], row["width"]) < patch for row in rows):
            raise ValueError("Patch exceeds a training image dimension")

    def __len__(self):
        return len(self.rows)

    def crop_parameters(self, index):
        row = self.rows[index]
        rng = random.Random(f"{self.seed}:{self.epoch}:{index}")
        top = rng.randrange(row["height"] - self.patch + 1)
        left = rng.randrange(row["width"] - self.patch + 1)
        return top, left, rng.randrange(4), rng.randrange(2)

    def __getitem__(self, index):
        row = self.rows[index]
        if self.cache is not None and index in self.cache:
            low, high = self.cache[index]
        else:
            low, high = (read_rgb(self.root / row[field]) for field in ("low", "high"))
            if self.cache is not None:
                self.cache[index] = (low, high)
        if self.patch is not None:
            top, left, turns, flip = self.crop_parameters(index)
            region = np.s_[top:top+self.patch, left:left+self.patch]
            low, high = np.rot90(low[region], turns), np.rot90(high[region], turns)
            if flip:
                low, high = low[:, ::-1], high[:, ::-1]
        images = tuple(torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float().div_(255)
                       for image in (low, high))
        return *images, row["id"]
