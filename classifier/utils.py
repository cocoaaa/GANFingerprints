from typing import Optional, Union, List
from datetime import datetime
from pathlib import Path
from PIL import Image

IMG_EXTENSIONS = [".jpg", ".jpeg", ".png"] # ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")


def mkdir(p: Path, parents=True):
    if not p.exists():
        p.mkdir(parents=parents)
        print("Created: ", p)

def info(arr, header=None):
    if header is None:
        header = "="*30
    print(header)
    print("shape: ", arr.shape)
    print("dtype: ", arr.dtype)
    print("min, max: ", min(np.ravel(arr)), max(np.ravel(arr)))

def now2str(delimiter: Optional[str]='-'):
    now = datetime.now()
    now_str = now.strftime(f"%Y%m%d{delimiter}%H%M%S")
    return now_str

# === ops on image dir
def pil_rgb_loader(path: Union[Path,str]) -> Image.Image:
    # src: torchvision
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

def has_img_suffix(fp: Path, valid_suffixes:List[str]=IMG_EXTENSIONS):
    return fp.suffix.lower() in valid_suffixes

def is_img_fp(fp:Path, valid_suffixes:List[str]=IMG_EXTENSIONS):
    return fp.is_file() and has_img_suffix(fp, valid_suffixes)

def is_valid_dir(fp: Union[Path,str]) -> bool:
    if isinstance(fp, Path):
        fp = str(fp) 
    return not (fp.startswith('.') and Path(fp).is_file())

def count_imgs(dir_path: Path, valid_suffixes:List[str]=IMG_EXTENSIONS) -> int:
    """ Count the number of images in the directory """
    c = 0
    for img_fp in dir_path.iterdir():
        if img_fp.is_file() and has_img_suffix(img_fp, valid_suffixes):
            c += 1
    return c