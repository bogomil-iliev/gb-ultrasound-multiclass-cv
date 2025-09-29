#Import Needed libraries.
import os, random
import numpy as np
import torch

#Core config
IMG_SIZE = 320
VOL_PERCENTILE = 8.0   #blur threshold percentile for train filtering
MEDIAN_FILTER = True   #apply 3x3 median denoise
BATCH_SIZE = 32
NUM_WORKERS = 2        #adjust to between 2-4.

#Repro seeding
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

#Make CuDNN behavior deterministic (slower but reproducible)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print(f"[CFG] IMG_SIZE={IMG_SIZE}, VOL_PERCENTILE={VOL_PERCENTILE}, MEDIAN_FILTER={MEDIAN_FILTER}")

#Import Needed libraries.

#Restricting the environment to certain versions due to deprications in the latest ones or errors.
!pip -q install albumentations==1.4.4 opencv-python==4.9.0.80

import albumentations as A
from albumentations.pytorch import ToTensorV2

#Definition of Z-score function.
class PerImageZScore(A.ImageOnlyTransform):
    """Per-image z-score: (x - mean) / std, applied after photometric augs."""
    def __init__(self, p=1.0, eps=1e-6):
        super().__init__(always_apply=True, p=p)
        self.eps = eps
    def apply(self, img, **params):
        img = img.astype(np.float32)
        mu = img.mean()
        sigma = img.std()
        sigma = max(sigma, self.eps)
        return (img - mu) / sigma

def make_train_transform():
    return A.Compose([
        #photometric + geometric (small, conservative)
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.10, rotate_limit=10,
                           border_mode=cv2.BORDER_CONSTANT, p=0.8),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.8),
        #standardise after augs
        PerImageZScore(p=1.0),
        ToTensorV2(),
    ])

def make_eval_transform():
    return A.Compose([
        PerImageZScore(p=1.0),
        ToTensorV2(),
    ])

train_tf = make_train_transform()
eval_tf  = make_eval_transform()

print("[OK] Transforms ready (train vs eval).")

#Reuses: safe_imread_gray, letterbox_square, median_3x3, to_rgb3_from_gray (from earlier)

#Function to transform 1-channel (gray) to 3-channels (RGB)
def to_rgb3_from_gray(img_gray: np.ndarray):
    """Stack gray to 3 channels (RGB order)."""
    return cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

#Function for Blur filtering.
def median_3x3(img_gray: np.ndarray):
    return cv2.medianBlur(img_gray, 3)

from torch.utils.data import Dataset, DataLoader

class GBDDataset(Dataset):
    """
    Pipeline (per sample):
      1) read grayscale
      2) letterbox to square
      3) resize to IMG_SIZE
      4) optional 3x3 median denoise
      5) convert to RGB
      6) Albumentations (train or eval)
      7) return tensor, label, filepath
    """
    def __init__(self, df, class_to_idx: dict, transform: A.Compose,
                 img_size: int = 320, use_median: bool = True):
        self.df = df.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.img_size = img_size
        self.use_median = use_median

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        fp = row.filepath
        label = self.class_to_idx[row["class"]]

        #1. read grayscale
        g = safe_imread_gray(fp)
        if g is None:
            raise RuntimeError(f"Failed to read image: {fp}")

        #2. letterbox to square
        g_sq, _ = letterbox_square(g, pad_value=0)

        #3. resize
        g_rz = cv2.resize(g_sq, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        #4. optional median denoise
        if self.use_median:
            g_rz = median_3x3(g_rz)

        #5. to rgb
        rgb = to_rgb3_from_gray(g_rz)  # uint8, HWC

        #6. albumentations
        aug = self.transform(image=rgb)
        img_t = aug["image"]  # CHW float tensor

        return img_t, int(label), fp



def vol_score_at_size(fp: str, size: int = IMG_SIZE) -> float | None:
    """Load gray → letterbox → resize → VoL."""
    g = safe_imread_gray(fp)
    if g is None:
        return None
    g_sq, _ = letterbox_square(g, pad_value=0)
    g_rz = cv2.resize(g_sq, (size, size), interpolation=cv2.INTER_LINEAR)
    return variance_of_laplacian(g_rz)

def compute_vol_scores(filepaths, max_samples=None, seed=SEED):
    """Compute VoL scores for a list of filepaths; optionally subsample for speed."""
    rng = np.random.default_rng(seed)
    fps = list(filepaths)
    if max_samples and len(fps) > max_samples:
        fps = list(rng.choice(fps, size=max_samples, replace=False))
    scores = []
    skipped = 0
    for fp in tqdm(fps, desc="VoL", unit="img"):
        s = vol_score_at_size(fp, size=IMG_SIZE)
        if s is None:
            skipped += 1
            continue
        scores.append(s)
    scores = np.array(scores, dtype=np.float64)
    print(f"[VoL] Scored={len(scores)} | Skipped={skipped}")
    return scores

def choose_threshold_from_train(train_df, percentile: float = VOL_PERCENTILE):
    """Choose τ from the train pool only (e.g., 8th percentile)."""
    scores = compute_vol_scores(train_df["filepath"].tolist(), max_samples=None)
    tau = float(np.percentile(scores, percentile))
    return tau, scores

def filter_train_by_blur(train_df, tau: float, log_dir: Path):
    """
    Keep only images with VoL >= tau.
    Writes CSVs of kept/dropped for reproducibility.
    """
    keep_mask = []
    kept_rows, dropped_rows = [], []
    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="QC filter", unit="img"):
        s = vol_score_at_size(row.filepath, size=IMG_SIZE)
        ok = (s is not None) and (s >= tau)
        keep_mask.append(ok)
        (kept_rows if ok else dropped_rows).append({**row.to_dict(), "vol": None if s is None else float(s)})

    kept_df = pd.DataFrame(kept_rows).reset_index(drop=True)
    dropped_df = pd.DataFrame(dropped_rows).reset_index(drop=True)

    log_dir.mkdir(parents=True, exist_ok=True)
    kept_df.to_csv(log_dir / "train_kept_after_blur.csv", index=False)
    dropped_df.to_csv(log_dir / "train_dropped_blur.csv", index=False)

    print(f"[QC] Kept {len(kept_df):,} | Dropped {len(dropped_df):,} (τ={tau:.2f})")
    return kept_df, dropped_df

#Installation of needed libraries.

from sklearn.model_selection import StratifiedShuffleSplit

#Safety: needs df_all and class_to_idx from EDA Cell 2
try:
    _ = df_all, class_to_idx
except NameError:
    raise SystemExit("Run EDA Cell 2 first (it defines df_all and class_to_idx).")

#(Demo) stratified image-level split 85/15 (we'll replace with patient-level CV later)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
y = df_all["class"].map(class_to_idx).values
tr_idx, te_idx = next(sss.split(df_all, y))
df_dev  = df_all.iloc[tr_idx].reset_index(drop=True)
df_test = df_all.iloc[te_idx].reset_index(drop=True)
print(f"Dev={len(df_dev):,} | Test={len(df_test):,}")

#Choose blur threshold tau from DEV (train pool)
QC_LOG_DIR = (EDA_OUT / "qc_demo")
tau, scores = choose_threshold_from_train(df_dev, percentile=VOL_PERCENTILE)
print(f"[QC] τ (p{VOL_PERCENTILE:.0f}) = {tau:.2f}")

#Filter DEV by tau (train only); Test remains untouched
df_train_qc, df_train_dropped = filter_train_by_blur(df_dev, tau=tau, log_dir=QC_LOG_DIR)

#Build Datasets
train_ds = GBDDataset(df_train_qc, class_to_idx, transform=train_tf,
                      img_size=IMG_SIZE, use_median=MEDIAN_FILTER)
test_ds  = GBDDataset(df_test,      class_to_idx, transform=eval_tf,
                      img_size=IMG_SIZE, use_median=MEDIAN_FILTER)

#DataLoaders
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

#Peek at one batch
xb, yb, fps = next(iter(train_loader))
print("Train batch:", xb.shape, yb.shape, "| dtype", xb.dtype)
print("Classes in this batch:", sorted(set(yb.tolist())))

#Install needed packages and libraries.

import matplotlib.pyplot as plt

def show_sample(df_row, transform):
    g = safe_imread_gray(df_row.filepath)
    g_sq, _ = letterbox_square(g, pad_value=0)
    g_rz = cv2.resize(g_sq, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    if MEDIAN_FILTER:
        g_rz = median_3x3(g_rz)
    rgb = to_rgb3_from_gray(g_rz)

    #Apply transform once (e.g., train_tf with augs)
    out = transform(image=rgb)["image"]
    #convert to HWC for plotting
    out_np = out.permute(1, 2, 0).cpu().numpy()
    #undo z-score for display (min-max to 0-1 for quick view)
    out_disp = (out_np - out_np.min()) / (out_np.max() - out_np.min() + 1e-6)

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(rgb, cmap=None)
    ax[0].set_title("Preprocessed (RGB)"); ax[0].axis("off")
    ax[1].imshow(out_disp, cmap=None)
    ax[1].set_title("After augs + z-score"); ax[1].axis("off")
    plt.show()

#Try it on a random kept training image
if len(df_train_qc):
    show_sample(df_train_qc.sample(1, random_state=SEED).iloc[0], transform=train_tf)

