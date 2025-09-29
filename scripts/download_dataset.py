#Importing Path from pathlib library - to work easier with file paths.
from pathlib import Path

#Setting Paths
ZIP_URL   = "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/r6h24d2d3y-1.zip" #Do NOT change this path. It is the direct link to download the dataset.
DEST_DIR  = Path("/content/drive/MyDrive/AI/Project_Summer_Module")   #CHANGE your chosen DESTINATION folder where you want to save the dataset.
LOCAL_ZIP = Path("/content") / "UIdataGB_dataset.zip"                  #Temporary download path for the zip.

#Create destination (if it doesn’t exist)
DEST_DIR.mkdir(parents=True, exist_ok=True)

print(f"Zip will be downloaded to: {LOCAL_ZIP}")
print(f"Dataset will be extracted to: {DEST_DIR}")

#Setting up the downloading of the Dataset from the URL safely and resumably, showing a progress bar.

#Importing of Libraries needed.
import requests #requests is a HTTP client library. Used for HEAD/GET requests, streaming content, handling timeouts, and headers.
from tqdm.auto import tqdm #automatically picks a progress-bar backend that works in notebooks or terminals, and updates it as bytes arrive.


"""
    The get_remote_size(url) method tries to obtain the remote file size in bytes.
    - Makes a HEAD request to read the 'Content-Length' header.
    - If the server doesn't return it on HEAD, falls back to a streaming GET.
    Returns: int size in bytes, or None if unknown.
"""
def get_remote_size(url):
    #Try to obtain remote file size (bytes).
    try:
      #Quick metadata check
        r = requests.head(url, allow_redirects=True, timeout=20)
        size = r.headers.get("Content-Length")
        if not size:
          #Some servers don't give size on HEAD; try GET headers instead.
            r = requests.get(url, stream=True, timeout=20)
            size = r.headers.get("Content-Length")
        return int(size) if size else None
    except Exception:
      #If anything goes wrong (e.g., network), treat size as unknown.
        return None





"""
    Stream-download a file with resume support and a progress bar.

    Args:
      url (str): remote file URL
      dst (pathlib.Path): local destination path (final filename)
      chunk_size (int): read size per network iteration (default 1 MiB)

    Behavior:
      - Writes to 'dst.suffix + .part' first (temp file).
      - If that temp file exists, resumes from its current size using HTTP Range.
      - Shows a tqdm progress bar (bytes), with a total if known.
      - On success, renames the temp file to 'dst' atomically.
"""

def download_with_resume(url, dst, chunk_size=2**20):

    #Ensure destination directory exists.
    dst.parent.mkdir(parents=True, exist_ok=True)

    #Temp file we write to while downloading.
    temp_path = dst.with_suffix(dst.suffix + ".part")

    #If a partial file exists, we will try to resume from this position.
    resume_pos = temp_path.stat().st_size if temp_path.exists() else 0

    #Get total size (if the server provides Content-Length).
    total_size = get_remote_size(url)

    #Build headers: only set Range if we can actually resume.
    headers = {}
    if total_size and resume_pos and resume_pos < total_size:
        headers["Range"] = f"bytes={resume_pos}-"

    #If we’re resuming - append to file; otherwise start fresh.
    mode = "ab" if "Range" in headers else "wb"

    #Stream the response so we don’t load the entire file into memory.
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        initial = resume_pos if total_size else 0
        total   = total_size if total_size else None

        #tqdm needs an initial and a total to display a complete progress bar
        with open(temp_path, mode) as f, tqdm(total=total, initial=initial, unit="B", unit_scale=True, desc="Downloading") as pbar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    #Only rename once the whole file is written.
    temp_path.rename(dst)
    print(f"[OK] Download complete: {dst} ({dst.stat().st_size:,} bytes)")

#Run the download
download_with_resume(ZIP_URL, LOCAL_ZIP)

#Unzipping of the Dataset Zips.
import zipfile, shutil
from pathlib import Path
from tqdm.auto import tqdm

def is_within_directory(directory: Path, target: Path) -> bool:
    """Prevent zip-slip by ensuring target stays inside directory."""
    try:
        return directory.resolve() in target.resolve().parents or directory.resolve() == target.resolve()
    except Exception:
        return False

def safe_unzip(zip_path: Path, dest_dir: Path):
    """Safely extract a ZIP file to dest_dir (no path traversal)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for m in tqdm(zf.infolist(), desc=f"Extracting {zip_path.name}", unit="file"):
            target_path = dest_dir / m.filename
            if not is_within_directory(dest_dir, target_path):
                raise RuntimeError(f"Blocked unsafe path: {m.filename}")
            if m.is_dir():
                target_path.mkdir(parents=True, exist_ok=True)
            else:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(m, 'r') as src, open(target_path, 'wb') as dst:
                    shutil.copyfileobj(src, dst)
    print(f"[OK] Extracted to: {dest_dir}")

# Unzip the main archive
safe_unzip(LOCAL_ZIP, DEST_DIR)

"""
    This code block unzips an archive safely into a destination folder, with a progress bar and without loading the whole files into RAM.
    It performs a security check - before writing each file, it verifies the target path is inside the destination directory.
    It loops over all entries in the ZIP, creates any needed folders, then streams each file from the archive to disk.
"""
#zipfile is a standard library module to read ZIP archives; shutil is used to prevent files from loading into the RAM.
import zipfile, shutil

#Provides object-oriented file paths and makes it easy to resolve absolute paths and join paths safely.
from pathlib import Path

#progress bars.
from tqdm.auto import tqdm

#Root where inner zips live (we'll search recursively from DEST_DIR)
SEARCH_ROOT = DEST_DIR

#Temp extraction on VM (fast). Will be cleaned as we go.
TMP_ROOT = Path("/content/tmp_extract_inner")
TMP_ROOT.mkdir(parents=True, exist_ok=True)

def safe_extract_all(zf: zipfile.ZipFile, dest_dir: Path):
    #Extracts all members safely into dest_dir.
    dest_dir.mkdir(parents=True, exist_ok=True)
    for m in zf.infolist():
        target_path = dest_dir / m.filename
        if not is_within_directory(dest_dir, target_path):
            raise RuntimeError(f"Blocked unsafe path: {m.filename}")
        if m.is_dir():
            target_path.mkdir(parents=True, exist_ok=True)
        else:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(m, 'r') as src, open(target_path, 'wb') as dst:
                shutil.copyfileobj(src, dst)

def move_all(src: Path, dst: Path):
    #Move all children of src into dst (merge/overwrite if needed).
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if target.exists():
            if target.is_dir() and item.is_dir():
                #merge directories
                for child in item.iterdir():
                    shutil.move(str(child), str(target / child.name))
                item.rmdir()
                continue
            #remove existing and replace
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        shutil.move(str(item), str(target))

#Find all zip files recursively (inner/class zips)
inner_zips = sorted([p for p in SEARCH_ROOT.rglob("*.zip") if p.is_file()])
print(f"Found {len(inner_zips)} inner zip(s) under {SEARCH_ROOT}")

for zip_path in tqdm(inner_zips):
    #Desired final folder: same name as the zip (without .zip), in the same parent
    final_dir = zip_path.with_suffix("")
    tmp_dir   = TMP_ROOT / final_dir.name

    #Clean temp and extract there first
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zf:
        safe_extract_all(zf, tmp_dir)

    #If the zip contains a single top-level folder, flatten it.
    children = [c for c in tmp_dir.iterdir()]
    inner_root = children[0] if (len(children) == 1 and children[0].is_dir()) else tmp_dir

    #Move everything up to the final_dir (merge/overwrite safe)
    move_all(inner_root, final_dir)

    #Cleanup temp and remove the original zip to save space
    shutil.rmtree(tmp_dir, ignore_errors=True)
    try:
        zip_path.unlink()
    except Exception as e:
        print(f"[WARN] Could not delete {zip_path}: {e}")

    print(f"[OK] {zip_path.name} → {final_dir} (flattened)")

print("All inner zips extracted, flattened, and removed.")

from pathlib import Path

#Provides a summary to double-check what was extracted, if there are any top-level folders and what is the approximate size of the data.
def summarize_directory(root: Path):
    """Lightweight summary: class folders, file/image counts, size."""
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    n_files = 0
    n_images = 0
    total_bytes = 0
    for p in root.rglob("*"):
        if p.is_file():
            n_files += 1
            total_bytes += p.stat().st_size
            if p.suffix.lower() in IMG_EXTS:
                n_images += 1
    top = [p.name for p in root.iterdir() if p.is_dir()]
    print("—"*60)
    print(f"[SUMMARY] Root: {root}")
    print(f"Top-level folders (first 20): {top[:20]}{' ...' if len(top) > 20 else ''}")
    print(f"Total files: {n_files:,} | Image files: {n_images:,}")
    print(f"Approx size: {total_bytes/1e9:.2f} GB")
    print("—"*60)

summarize_directory(DEST_DIR)

#Check if the Dataset .zip was deleted.
try:
    LOCAL_ZIP.unlink()
    print(f"[OK] Deleted local zip: {LOCAL_ZIP}")
except FileNotFoundError:
    pass
