from pathlib import Path
from typing import Optional, Sequence, Tuple
import torch
from PIL import Image
import numpy as np
import os

from exp_logging.io_utils import ensure_dir, stitch_triptych

# ------------------- batch access helpers -------------------

def _pick(batch: dict, keys):
    for k in keys:
        if k in batch:
            return batch[k]
    return None

def _find_img_path(batch: dict) -> Optional[str]:
    for k in ("img_path", "image_path", "path"):
        p = batch.get(k)
        if isinstance(p, str) and os.path.exists(p):
            return p
    meta = batch.get("meta") or {}
    for k in ("img_path", "image_path", "path"):
        p = meta.get(k)
        if isinstance(p, str) and os.path.exists(p):
            return p
    return None

def _ensure_palette(palette, n: int) -> list:
    if palette and len(palette) >= n:
        return palette
    rng = np.random.default_rng(42)
    pal = [(int(r), int(g), int(b)) for r, g, b in rng.integers(0, 255, size=(max(n, 256), 3))]
    if palette:
        pal[:len(palette)] = palette
    return pal

# ------------------- RGB base acquisition -------------------

def _pil_from_uint8_array(arr: np.ndarray) -> Image.Image:
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    if arr.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3 array, got {arr.shape}")
    return Image.fromarray(arr.astype(np.uint8))

def _get_rgb_pil_from_batch(batch: dict,
                            rgb_indices: Optional[Tuple[int,int,int]],
                            device: torch.device,
                            mean: Optional[Sequence[float]],
                            std: Optional[Sequence[float]]) -> Image.Image:
    """
    Order of preference:
      1) Pre-unnormalized uint8 image in batch (image_uint8/rgb_uint8/image_raw/...).
      2) Load from img_path in batch/meta.
      3) Use model input tensor and minimally undo normalization if needed.
    """
    # 1) Raw uint8 already in batch?
    raw = _pick(batch, ["image_uint8", "rgb_uint8", "image_raw", "rgb_raw"])
    if raw is not None:
        if isinstance(raw, torch.Tensor):
            x = raw.detach().cpu().to(torch.uint8).numpy()
        else:
            x = np.asarray(raw)
        return _pil_from_uint8_array(x)

    # 2) Load from path?
    p = _find_img_path(batch)
    if p:
        im = Image.open(p).convert("RGB")
        return im

    # 3) Fall back to model input tensor
    imgs = _pick(batch, ["image", "img"])
    if imgs is None:
        raise KeyError(f"Expected one of ['image','img'] in batch, got keys={list(batch.keys())}")
    x = imgs[0] if imgs.ndim == 4 else imgs  # allow single or batched outside
    x = x.detach().cpu().float()              # CxHxW (normalized)

    C, H, W = x.shape
    if rgb_indices is None:
        if C < 3:
            raise ValueError(f"Need >=3 channels to visualize (got {C})")
        sel = (0, 1, 2)
    else:
        sel = rgb_indices
        if max(sel) >= C:
            raise ValueError(f"rgb_indices {sel} out of range for C={C}")

    rgb = x[list(sel)]  # 3xHxW

    # If tensor looks normalized (min < 0 or max > 1), undo normalization using provided mean/std.
    # This is the only processing we keep, because otherwise "as-is" would be near-black.
    if (float(rgb.min()) < 0.0 or float(rgb.max()) > 1.0) and mean is not None and std is not None:
        m = torch.tensor(mean, dtype=torch.float32).view(3,1,1)
        s = torch.tensor(std,  dtype=torch.float32).view(3,1,1)
        rgb = (rgb * s + m)

    rgb = rgb.clamp(0.0, 1.0)
    arr = (rgb.permute(1,2,0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr)

# ------------------- overlay blending -------------------

def _overlay_mask_on_rgb(rgb_pil: Image.Image,
                         mask_hw: torch.Tensor,
                         palette: list,
                         alpha: float = 0.55,
                         ignore_index: int = 255) -> Image.Image:
    rgb = np.array(rgb_pil, dtype=np.uint8)
    H, W, _ = rgb.shape
    color = np.zeros((H, W, 3), dtype=np.uint8)

    p = mask_hw.detach().cpu().numpy().astype(np.int32)
    valid = p >= 0
    uniq = np.unique(p[valid])
    for c in uniq:
        if c == ignore_index:
            continue
        color[p == c] = palette[c % len(palette)]

    if np.any(p == ignore_index):
        keep = (p == ignore_index)[..., None]
        blended = np.where(keep, rgb, (alpha * color + (1.0 - alpha) * rgb).astype(np.uint8))
    else:
        blended = (alpha * color + (1.0 - alpha) * rgb).astype(np.uint8)

    return Image.fromarray(blended)

# ------------------- main API -------------------

def _instances_to_semantic_union(
    masks: torch.Tensor,    # [N,H,W] uint8/bool/float
    labels: torch.Tensor,   # [N] int (1..K), 0 is background (unused)
    num_classes: int,
) -> torch.Tensor:
    """Union of instance masks per class -> [H,W] long with ids in 0..K."""
    if masks.numel() == 0:
        # caller should pass the target H,W to create zeros—handle here if provided:
        return None
    m = (masks > 0.5).to(torch.uint8)               # [N,H,W]
    H, W = m.shape[-2], m.shape[-1]
    out = torch.zeros((num_classes, H, W), dtype=torch.uint8)
    for i in range(m.size(0)):
        cid = int(labels[i])
        if 0 < cid < num_classes:
            out[cid] |= m[i]
    return out.argmax(dim=0).to(torch.long)       # [H,W] class ids


def _overlay_instances_unique(
    rgb_pil: Image.Image,
    masks,
    alpha: float = 0.55,
    seed: int = 0,
):
    """Blend each instance with its own random color."""
    if masks is None:
        return rgb_pil

    if isinstance(masks, torch.Tensor):
        masks_np = masks.detach().cpu().numpy()
    else:
        masks_np = np.asarray(masks)

    if masks_np.size == 0:
        return rgb_pil

    if masks_np.ndim == 4 and masks_np.shape[1] == 1:
        masks_np = masks_np[:, 0]

    overlay = np.array(rgb_pil, dtype=np.uint8).copy()
    rng = np.random.default_rng(seed)

    for idx in range(masks_np.shape[0]):
        mask = masks_np[idx]
        if mask.dtype != np.bool_:
            mask = mask > 0.5
        if not mask.any():
            continue
        color = rng.integers(30, 230, size=3, dtype=np.uint8)
        overlay[mask] = (alpha * color + (1.0 - alpha) * overlay[mask]).astype(np.uint8)

    return Image.fromarray(overlay)

def _tensor_chw_to_pil_uint8(x, rgb_indices=(0, 1, 2)) -> Image.Image:
    """
    x: torch.Tensor [C,H,W] or [1,C,H,W], already in [0,1] range.
    Just clamp and convert to uint8 RGB. No color jitter, no denorm.
    """
    import torch
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        if x.dim() == 4:
            x = x[0]
        # pick RGB channels (ignore DSM or extras)
        C = x.size(0)
        r, g, b = [c for c in rgb_indices if c < C][:3] if C >= 3 else (0, 0, 0)
        x = x[[r, g, b], ...] if C >= 3 else x.expand(3, *x.shape[1:])
        x = x.clamp(0.0, 1.0)
        arr = (x.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        return Image.fromarray(arr)
    raise TypeError("Expected a torch.Tensor")
    
@torch.no_grad()
def collect_overlays_for_split_instance(
    model,
    device,
    loader,                         # DataLoader with (list[Tensor], list[target])
    outdir: Path,
    palette,
    n_samples: Optional[int] = 8,
    score_thresh: float = 0.5,      # for visualization only
    rgb_indices: Optional[Tuple[int,int,int]] = None,  # if your tensor isn’t RGB at (0,1,2)
    alpha: float = 0.55,
    num_classes: Optional[int] = None,            # len(dataset.classes)
    instance_colors: bool = False,
):
    """
    Save triptychs (RGB | GT-overlaid | Pred-overlaid) for Mask R-CNN.
    - Uses *input* tensors (0..1) for RGB, no extra processing.
    - When `instance_colors` is False (default), GT/PRED overlays mirror semantic unions.
      When True, every instance gets its own random color to highlight per-object areas.
    """
    ensure_dir(outdir)
    model.eval()
    saved = 0
    pal = _ensure_palette(palette, 256)

    for imgs, targets in loader:
        imgs = [im.to(device) for im in imgs]
        # ---- RGB panel from the *pre-normalized* input (0..1) ----
        # (Mask R-CNN normalizes internally; for viz we want original)
        x = imgs[0].detach().cpu()
        rgb_pil = _tensor_chw_to_pil_uint8(x, rgb_indices=rgb_indices)

        # ---- GT overlay ----
        tgt = targets[0]
        H, W = tgt["masks"].shape[-2], tgt["masks"].shape[-1]
        default_nc = int(tgt["labels"].max().item()) + 1 if tgt["labels"].numel() else 1
        nc = int(num_classes) if num_classes is not None else default_nc
        if instance_colors:
            gt_overlay = _overlay_instances_unique(rgb_pil, tgt["masks"], alpha=alpha, seed=1337 + saved)
        else:
            gt_sem = _instances_to_semantic_union(tgt["masks"].cpu(), tgt["labels"].cpu(), nc)
            if gt_sem is None:
                gt_sem = torch.zeros((H, W), dtype=torch.long)
            gt_overlay = _overlay_mask_on_rgb(rgb_pil, gt_sem, pal, alpha=alpha, ignore_index=255)

        # ---- Pred union -> overlay ----
        outs = model(imgs)
        out = outs[0]
        # threshold for visualization only
        keep = (out["scores"].detach().cpu().numpy() >= score_thresh)
        if keep.any():
            pm = out["masks"].detach().cpu()[keep].squeeze(1)   # [N,H,W]
            pl = out["labels"].detach().cpu()[keep]             # [N]
            if instance_colors:
                pr_overlay = _overlay_instances_unique(rgb_pil, pm, alpha=alpha, seed=7331 + saved)
            else:
                pr_sem = _instances_to_semantic_union(pm, pl, nc)
                if pr_sem is None:
                    pr_sem = torch.zeros((H, W), dtype=torch.long)
                pr_overlay = _overlay_mask_on_rgb(rgb_pil, pr_sem, pal, alpha=alpha, ignore_index=255)
        else:
            if instance_colors:
                pr_overlay = rgb_pil
            else:
                pr_overlay = _overlay_mask_on_rgb(
                    rgb_pil,
                    torch.zeros((H, W), dtype=torch.long),
                    pal,
                    alpha=alpha,
                    ignore_index=255,
                )

        trip = stitch_triptych(rgb_pil, gt_overlay, pr_overlay)
        trip.save(outdir / f"sample_{saved:03d}.png")
        saved += 1
        if n_samples is not None and saved >= n_samples:
            return
        
@torch.no_grad()
def collect_overlays_for_split(
    model,
    device,
    loader,
    outdir: Path,
    palette,
    n_samples: Optional[int] = 8,
    ignore_index: int = 255,
    rgb_indices: Optional[Tuple[int,int,int]] = None,
    alpha: float = 0.55,
    rgb_mean: Optional[Sequence[float]] = None,   # pass your train mean
    rgb_std: Optional[Sequence[float]]  = None,   # pass your train std
):
    """
    Save triptychs: [RGB | GT overlay | Pred overlay]
    - We try to use an **unnormalized** RGB first (uint8 tensor or image path).
    - If not available, we minimally un-normalize the model input (using mean/std)
      just enough to make the image visible (no min-max tricks).
    """
    ensure_dir(outdir)
    model.eval()
    saved = 0
    pal = _ensure_palette(palette, 256)

    for batch in loader:
        logits = model(_pick(batch, ["image", "img"]).to(device, non_blocking=True))
        preds  = torch.argmax(logits, dim=1)

        gts = _pick(batch, ["mask", "gt"])
        if gts is None:
            raise KeyError("Expected 'mask' or 'gt' in batch.")
        gts = gts.to(device, non_blocking=True)

        B = gts.size(0)
        for b in range(B):
            if n_samples is not None and saved >= n_samples:
                return

            rgb_pil = _get_rgb_pil_from_batch(
                {k: (v[b] if isinstance(v, torch.Tensor) and v.ndim>0 and v.size(0)==B else v)
                 for k, v in batch.items()},
                rgb_indices=rgb_indices,
                device=device,
                mean=rgb_mean,
                std=rgb_std
            )

            gt_overlay = _overlay_mask_on_rgb(rgb_pil, gts[b],   pal, alpha=alpha, ignore_index=ignore_index)
            pr_overlay = _overlay_mask_on_rgb(rgb_pil, preds[b], pal, alpha=alpha, ignore_index=ignore_index)

            trip = stitch_triptych(rgb_pil, gt_overlay, pr_overlay)
            trip.save(outdir / f"sample_{saved:03d}.png")
            saved += 1
