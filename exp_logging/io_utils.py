from pathlib import Path
import json, shutil
from PIL import Image
import numpy as np

def ensure_dir(p:Path): p.mkdir(parents=True, exist_ok=True)
def write_json(obj, path:Path): ensure_dir(path.parent); path.write_text(json.dumps(obj, indent=2))
def read_json(path:Path, default=None):
    if not path.exists(): return default
    try: return json.loads(path.read_text())
    except Exception: return default

def rm_tree(path:Path):
    if path.exists(): shutil.rmtree(path)

def colorize_mask(mask_np:np.ndarray, palette:list[list[int]]):
    h,w = mask_np.shape
    out = np.zeros((h,w,3), np.uint8)
    k = min(len(palette), int(mask_np.max())+1)
    for c in range(k):
        out[mask_np==c] = palette[c]
    return Image.fromarray(out)

def stitch_triptych(im1:Image.Image, im2:Image.Image, im3:Image.Image):
    h = max(im1.height, im2.height, im3.height)
    ws = [im1.width, im2.width, im3.width]
    canv = Image.new("RGB", (sum(ws), h), (0,0,0))
    x=0
    for im in [im1, im2, im3]:
        if im.height!=h: im = im.resize((int(im.width*h/im.height), h))
        canv.paste(im, (x,0)); x+=im.width
    return canv
