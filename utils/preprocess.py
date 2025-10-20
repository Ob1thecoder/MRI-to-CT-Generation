import os, re, cv2, math, numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import matplotlib.pyplot as plt

def natural_key(p: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', p)]

def robust_norm(x, p_lo=0.5, p_hi=99.5, eps=1e-6):
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, [p_lo, p_hi])
    x = np.clip(x, lo, hi)
    mu, sd = x.mean(), x.std() + eps
    x = (x - mu) / sd
    x = np.tanh(x)  # squash to [-1, 1]
    return x

class PairedCTT2Loader(keras.utils.Sequence):
    """
    Directory layout (paired by identical sorted order within each patient):
      data/CT/PNG/Patient_001/*.png
      data/T2-MRI/PNG/Patient_001/*.png
    """
    def __init__(self, ct_root, t2_root, image_size=(256, 256), batch_size=8, shuffle=True, aug=True):
        self.ct_root = Path(ct_root)
        self.t2_root = Path(t2_root)
        self.H, self.W = image_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.aug = aug

        # patients present in both dirs
        pats_ct = {p.name for p in self.ct_root.iterdir() if p.is_dir()}
        pats_t2 = {p.name for p in self.t2_root.iterdir() if p.is_dir()}
        self.patients = sorted(list(pats_ct & pats_t2))

        # build slice pairs
        self.pairs = []
        dropped = 0
        for pid in self.patients:
            cdir = self.ct_root / pid
            tdir = self.t2_root / pid
            ct_slices = sorted([f for f in os.listdir(cdir) if f.lower().endswith(('.png','.jpg','.jpeg'))], key=natural_key)
            t2_slices = sorted([f for f in os.listdir(tdir) if f.lower().endswith(('.png','.jpg','.jpeg'))], key=natural_key)
            n = min(len(ct_slices), len(t2_slices))
            for i in range(n):
                self.pairs.append((str(cdir/ct_slices[i]), str(tdir/t2_slices[i])))
            dropped += abs(len(ct_slices)-len(t2_slices))
        if dropped:
            print(f"[Loader] Warning: dropped {dropped} unmatched slices across patients.")

        self.indices = np.arange(len(self.pairs))
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.pairs) / self.batch_size)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, index):
        idx = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        X_t2, Y_ct = [], []
        for i in idx:
            ct_path, t2_path = self.pairs[i]
            ct = cv2.imread(ct_path, cv2.IMREAD_GRAYSCALE)
            t2 = cv2.imread(t2_path, cv2.IMREAD_GRAYSCALE)
            if ct is None or t2 is None:
                continue  # skip broken files

            # Resize (cv2 expects (W,H))
            interp_ct = cv2.INTER_AREA if (ct.shape[0] >= self.H or ct.shape[1] >= self.W) else cv2.INTER_CUBIC
            interp_t2 = cv2.INTER_AREA if (t2.shape[0] >= self.H or t2.shape[1] >= self.W) else cv2.INTER_CUBIC
            ct = cv2.resize(ct, (self.W, self.H), interpolation=interp_ct)
            t2 = cv2.resize(t2, (self.W, self.H), interpolation=interp_t2)

            # Robust normalization → [-1, 1]
            ct = robust_norm(ct)
            t2 = robust_norm(t2)

            # simple paired flips/rotate
            if self.aug:
                if np.random.rand() < 0.5:
                    ct = np.flip(ct, axis=1); t2 = np.flip(t2, axis=1)
                if np.random.rand() < 0.2:
                    ct = np.flip(ct, axis=0); t2 = np.flip(t2, axis=0)
                if np.random.rand() < 0.3:
                    ang = np.random.uniform(-7, 7)
                    sc  = 1.0 + np.random.uniform(-0.05, 0.05)
                    M = cv2.getRotationMatrix2D((self.W/2, self.H/2), ang, sc)
                    ct = cv2.warpAffine(ct, M, (self.W, self.H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    t2 = cv2.warpAffine(t2, M, (self.W, self.H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

            # add channel dim
            X_t2.append(t2[..., None].astype(np.float32))
            Y_ct.append(ct[..., None].astype(np.float32))

        X_t2 = np.stack(X_t2, axis=0)
        Y_ct = np.stack(Y_ct, axis=0)
        return X_t2, Y_ct