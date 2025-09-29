#Importing of Libraries and Packages that are needed.
from pathlib import Path
import os, json, hashlib
import numpy as np
import pandas as pd
import cv2
from tqdm.auto import tqdm

#PATHS
PROJ_ROOT   = Path("/content/drive/MyDrive/AI/Project_Summer_Module") #Setting up of the root folder where the project is situated. CHANGE if you are running the code on your machine.
DATA_ROOT   = PROJ_ROOT / "Gallblader Diseases Dataset"   #your class folders
EDA_OUT     = DATA_ROOT / "eda_outputs"                   #setting the path for the EDA output
CACHE_ROOT  = PROJ_ROOT / "cache_320_rgb_v1"              #preprocessed images (RGB 320x320)
SPLITS_ROOT = PROJ_ROOT / "splits_5fold_v1"               #CSVs + metadata

CACHE_ROOT.mkdir(parents=True, exist_ok=True)
SPLITS_ROOT.mkdir(parents=True, exist_ok=True)

#PREPROCESSING SETTINGS
IMG_SIZE          = 320
USE_MEDIAN        = True          # apply 3×3 median denoise
VOL_PERCENTILE    = 8.0           # per-train-fold tau (VoL) percentile
SEED              = 42

#Load manifest from EDA (or fallback to df_all if still in memory)
manifest_csv = EDA_OUT / "eda_report.csv"
if manifest_csv.exists():
    df_all = pd.read_csv(manifest_csv)
    print(f"[OK] Loaded manifest: {manifest_csv}  (rows={len(df_all):,})")
else:
    try:
        _ = df_all
        print("[OK] Using df_all from memory (EDA not saved to CSV).")
    except NameError:
        raise SystemExit("No manifest found. Please run EDA to produce eda_outputs/eda_report.csv first.")

#Sanity: ensure columns
need_cols = {"filepath", "filename", "class", "class_idx", "patient_id"}
missing = need_cols - set(df_all.columns)
if missing:
    raise ValueError(f"Manifest missing columns: {missing}")

#Fallback: if any patient_id is NaN, assign a unique pseudo-ID (safe for grouping)
if df_all["patient_id"].isna().any():
    fill_ids = [f"img_{i}" for i in range(len(df_all))]
    df_all["patient_id"] = df_all["patient_id"].fillna(pd.Series(fill_ids, index=df_all.index))

#Helper: safe grayscale read
def safe_imread_gray(path: str):
    arr = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img

#Helper: letterbox to square
def letterbox_square(img_gray: np.ndarray, pad_value=0):
    h, w = img_gray.shape[:2]
    side = max(h, w)
    top    = (side - h) // 2
    bottom = side - h - top
    left   = (side - w) // 2
    right  = side - w - left
    sq = cv2.copyMakeBorder(img_gray, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_value)
    return sq

#Helper: variance of Laplacian
def vol(gray_img: np.ndarray) -> float:
    return float(cv2.Laplacian(gray_img, cv2.CV_64F).var())

#Safe PNG write (supports non-ASCII paths)
def imwrite_png(path: Path, img_bgr_or_rgb: np.ndarray):
    ext = ".png"
    path = path.with_suffix(ext)
    ok, enc = cv2.imencode(ext, img_bgr_or_rgb)
    if not ok:
        raise IOError(f"Failed to encode {path}")
    enc.tofile(str(path))
    return path

#IF You GET AN ERROR WITH COMPATIBILITY OF CV2 AND NUMPY RUN THIS CODE AND RESTART THE RUNTIME. RUN THE SANITY CHECK BELOW AND START AGAIN FROM 4.1.
#Clean re-pin to a stable stack for Colab
!pip -q uninstall -y opencv-python opencv-python-headless numpy
!pip -q install --upgrade --force-reinstall \
  "numpy==1.26.4" \
  "scikit-learn>=1.6.0" \
  opencv-python==4.9.0.80 \
  albumentations==1.4.4 \
  timm==0.9.12

#(Optional) silence a common IPython warning in some Colab images
!pip -q install "jedi>=0.18.0"

print("Now go to: Runtime then Restart runtime. After it restarts, run your imports again.")

#After the restart, run sanity-check imports
import numpy, cv2, sklearn, albumentations, timm, torch
print("NumPy:", numpy.__version__)          # should be 1.26.4
print("OpenCV:", cv2.__version__)           # 4.9.0.80
print("scikit-learn:", sklearn.__version__) # >=1.6
print("Albumentations:", albumentations.__version__)
print("timm:", timm.__version__)
print("Torch:", torch.__version__)

#Build cache path: CACHE_ROOT/<class>/<filename>.png
def cache_path_for(row) -> Path:
    cls = str(row["class"])
    base = Path(row["filename"]).stem + ".png"   # unify to png
    return CACHE_ROOT / cls / base

#Process and cache (skips if already exists)
cached_paths = []
vol_scores   = []

for i, row in tqdm(df_all.iterrows(), total=len(df_all), desc="Caching 320×320"):
    src = row["filepath"]
    dst = cache_path_for(row)
    if not dst.parent.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        g = safe_imread_gray(src)
        if g is None:
            print(f"[WARN] unreadable: {src}")
            cached_paths.append(None)
            vol_scores.append(np.nan)
            continue
        g_sq = letterbox_square(g, pad_value=0)
        g_rz = cv2.resize(g_sq, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        if USE_MEDIAN:
            g_rz = cv2.medianBlur(g_rz, 3)
        #to RGB (3ch) for backbones expecting 3 channels
        rgb = cv2.cvtColor(g_rz, cv2.COLOR_GRAY2RGB)
        try:
            imwrite_png(dst, rgb)
        except Exception as e:
            print(f"[WARN] failed to write: {dst} :: {e}")
            cached_paths.append(None)
            vol_scores.append(np.nan)
            continue
    #compute/store VoL on the cached (guaranteed 320×320 gray-equivalent)
    #Read back as gray for consistent VoL (fast)
    arr = np.fromfile(str(dst), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    v = np.nan if img is None else vol(img)
    cached_paths.append(str(dst))
    vol_scores.append(v)

df_all["cache_path"] = cached_paths
df_all["vol_320"] = vol_scores

#Save cache mapping
cache_map_csv = CACHE_ROOT / "cache_manifest.csv"
df_all.to_csv(cache_map_csv, index=False)
print(f"[OK] Wrote cache manifest: {cache_map_csv}")
print(f"[INFO] Cached ok: {(df_all['cache_path'].notna()).sum()} / {len(df_all)}")

from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
import numpy as np

#Ensure we only use rows that cached successfully
df_ok = df_all[df_all["cache_path"].notna()].reset_index(drop=True)
print(f"[OK] Images available for splitting: {len(df_ok):,}")

y = df_ok["class_idx"].values
groups = df_ok["patient_id"].astype(str).values

#1. Patient-level 15% hold-out TEST (group-only split; class strat is handled in CV later)
gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
train_dev_idx, test_idx = next(gss.split(df_ok, y, groups))
df_dev  = df_ok.iloc[train_dev_idx].reset_index(drop=True)
df_test = df_ok.iloc[test_idx].reset_index(drop=True)

#Save TEST manifest (no blur filtering)
test_csv = SPLITS_ROOT / "test.csv"
df_test[["cache_path","filepath","class","class_idx","patient_id","vol_320"]].to_csv(test_csv, index=False)
print(f"[OK] Saved held-out test set: {test_csv}  (n={len(df_test)})")

#2. 5-fold StratifiedGroupKFold on DEV (stratify by class, group by patient)
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
folds = list(sgkf.split(df_dev, df_dev["class_idx"], df_dev["patient_id"]))

#Meta info
meta = {
    "img_size": IMG_SIZE,
    "use_median": bool(USE_MEDIAN),
    "vol_percentile": float(VOL_PERCENTILE),
    "n_dev": int(len(df_dev)),
    "n_test": int(len(df_test)),
}
with open(SPLITS_ROOT / "meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print(f"[OK] Wrote meta.json")

def choose_tau_from_train(df_train: pd.DataFrame, percentile: float) -> float:
    scores = df_train["vol_320"].dropna().values
    if len(scores) == 0:
        #Fallback if no VoL available (should not happen with cache step)
        return 0.0
    return float(np.percentile(scores, percentile))

def save_fold_csvs(k: int, train_df: pd.DataFrame, val_df: pd.DataFrame, tau: float):
    fold_dir = SPLITS_ROOT / f"fold_{k}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    #Filter TRAIN by blur threshold (train only)
    keep_mask = (train_df["vol_320"].notna()) & (train_df["vol_320"] >= tau)
    kept  = train_df[keep_mask].copy()
    dropd = train_df[~keep_mask].copy()

    kept_csv   = fold_dir / "train.csv"
    drop_csv   = fold_dir / "train_dropped_blur.csv"
    val_csv    = fold_dir / "val.csv"
    thresh_json= fold_dir / "qc_threshold.json"

    kept[["cache_path","filepath","class","class_idx","patient_id","vol_320"]].to_csv(kept_csv, index=False)
    dropd[["cache_path","filepath","class","class_idx","patient_id","vol_320"]].to_csv(drop_csv, index=False)
    val_df[["cache_path","filepath","class","class_idx","patient_id","vol_320"]].to_csv(val_csv, index=False)

    with open(thresh_json, "w") as f:
        json.dump({"tau_vol_p": VOL_PERCENTILE, "tau_value": tau,
                   "n_train_raw": int(len(train_df)),
                   "n_train_kept": int(len(kept)),
                   "n_train_dropped": int(len(dropd)),
                   "n_val": int(len(val_df))}, f, indent=2)

    print(f"[FOLD {k}] τ={tau:.2f} | train kept={len(kept)} dropped={len(dropd)} | val={len(val_df)}")
    return kept_csv, val_csv, drop_csv, thresh_json

#Iterate folds
for k, (tr_idx, va_idx) in enumerate(folds, start=1):
    df_tr = df_dev.iloc[tr_idx].reset_index(drop=True)
    df_va = df_dev.iloc[va_idx].reset_index(drop=True)

    #Choose per-fold tau from TRAIN only
    tau_k = choose_tau_from_train(df_tr, VOL_PERCENTILE)

    #Save per-fold CSVs + QC logs
    save_fold_csvs(k, df_tr, df_va, tau_k)

print("[OK] All folds saved under:", SPLITS_ROOT)

import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

#Minimal dataset that reads cached 320×320 RGB and applies only transforms (no heavy preprocessing)
class CachedGBDDataset(Dataset):
    def __init__(self, csv_path, transform):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.class_to_idx = {c:i for i,c in enumerate(sorted(self.df["class"].unique()))}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        #read cached RGB directly
        arr = np.fromfile(row.cache_path, dtype=np.uint8)
        rgb = cv2.imdecode(arr, cv2.IMREAD_COLOR)  #BGR
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        aug = self.transform(image=rgb)
        x = aug["image"]
        y = int(row.class_idx)
        return x, y, row.cache_path

#Example: load fold 1
train_csv = SPLITS_ROOT / "fold_1" / "train.csv"
val_csv   = SPLITS_ROOT / "fold_1" / "val.csv"

#Same transforms as before (z-score + light augs for train)
class PerImageZScore(A.ImageOnlyTransform):
    def __init__(self, p=1.0, eps=1e-6):
        super().__init__(always_apply=True, p=p); self.eps = eps
    def apply(self, img, **params):
        img = img.astype(np.float32); mu, sd = img.mean(), img.std()
        return (img - mu) / max(sd, self.eps)

train_tf = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.10, rotate_limit=10,
                       border_mode=cv2.BORDER_CONSTANT, p=0.8),
    A.RandomBrightnessContrast(0.15, 0.15, p=0.8),
    PerImageZScore(p=1.0),
    ToTensorV2(),
])
eval_tf = A.Compose([PerImageZScore(p=1.0), ToTensorV2()])

train_ds = CachedGBDDataset(train_csv, transform=train_tf)
val_ds   = CachedGBDDataset(val_csv,   transform=eval_tf)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

xb, yb, fp = next(iter(train_loader))
print("Batch:", xb.shape, yb.shape)
