import cv2
import numpy as np
from typing import Tuple, List, Union, Dict, Any, Optional
import matplotlib.colors as mcolors
from PIL import Image
from pathlib import Path
from sklearn.cluster import KMeans
from bm3d import bm3d, BM3DStages


def get_color(color_name, normalize=False, bgr=False):
    """
    get the RGB value of a color name
    """
    color = (255, 0, 0) if not normalize else (1.0, 0.0, 0.0)
    try:
        color = mcolors.to_rgb(color_name)
        if not normalize:
            color = tuple(int(x * 255) for x in color)
    except ValueError:
        raise ValueError(
            f"Unable to recognize color '{color_name}'. Please provide a valid color name or RGB tuple."
        )
    color = color[::-1] if bgr else color
    return color


def get_background_color(
    image_path: str, downsample: int = 500_000, bgr=True
) -> tuple[int, int, int]:
    """
    Estimate the background color of a chart image by analyzing the center region.
    """
    # Read image and convert to RGB
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    # Crop to center 50% region to avoid edges
    new_w, new_h = int(w * 0.5), int(h * 0.5)
    left = (w - new_w) // 2
    upper = (h - new_h) // 2
    right = left + new_w
    lower = upper + new_h
    img = img.crop((left, upper, right, lower))

    arr = np.asarray(img)
    pixels = arr.reshape(-1, 3)

    # Downsample if there are too many pixels
    if pixels.shape[0] > downsample:
        idx = np.random.choice(pixels.shape[0], downsample, replace=False)
        pixels = pixels[idx]

    # Get the dominant color by counting unique colors
    colors, counts = np.unique(pixels, axis=0, return_counts=True)
    dominant_rgb = colors[counts.argmax()]
    rgb = tuple(int(c) for c in dominant_rgb)
    return tuple(reversed(rgb)) if bgr else rgb


def denoise_bm3d_luma(
    img_path: str, out_path: str, sigma: float = 25 / 255, one_stage: bool = True
):
    """
    Apply BM3D denoising to the luminance channel only (YCrCb color space).
    Preserves chrominance channels to avoid color shifts.
    """
    # Read image and convert to YCrCb color space
    bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(img_path)
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32) / 255.0
    y, cr, cb = cv2.split(ycrcb)

    # Apply BM3D denoising only to luminance (Y) channel
    stage = BM3DStages.HARD_THRESHOLDING if one_stage else BM3DStages.ALL_STAGES
    y_dn = bm3d(y, sigma_psd=sigma, stage_arg=stage).astype(np.float32)

    # Merge denoised Y with original Cr, Cb and convert back to BGR
    out_mat = cv2.merge((y_dn, cr, cb))
    out_bgr = cv2.cvtColor(
        (out_mat * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_YCrCb2BGR
    )
    cv2.imwrite(out_path, out_bgr)


def adaptive_color_mask(
    roi: np.ndarray,
    chroma_th_low: int = 10,
    chroma_th_high: int = 25,
    obvious_chroma: int = 25,
    obvious_ratio: float = 0.01,
    y_lo: int = 60,
    y_hi: int = 245,
) -> np.ndarray:
    """
    Apply adaptive chroma and luminance thresholds to create a color mask,
    then fill internal holes using flood-fill.
    """
    # Convert to YCrCb and compute chroma magnitude
    ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)
    chroma = cv2.magnitude(Cr.astype(np.float32) - 128, Cb.astype(np.float32) - 128)

    has_obvious = (chroma > obvious_chroma).mean() > obvious_ratio
    chroma_th = chroma_th_high if has_obvious else chroma_th_low

    mask = (chroma > chroma_th) & (Y > y_lo) & (Y < y_hi)
    mask = mask.astype(np.uint8) * 255

    inv = cv2.bitwise_not(mask)

    h, w = inv.shape
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(inv, ff_mask, seedPoint=(0, 0), newVal=255)

    holes = cv2.bitwise_not(inv)

    # Union of original mask and holes
    filled_mask = cv2.bitwise_or(mask, holes)
    filled_bool = filled_mask.astype(bool)

    return filled_bool


def keep_leftmost_component(filled_mask: np.ndarray) -> np.ndarray:
    """
    Keep only the leftmost connected component in filled_mask, set others to 0。
    """
    mask_u8 = filled_mask.astype(np.uint8)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

    if n <= 2:  # Only background + ≤1 region
        return filled_mask.copy()

    # stats: [x, y, w, h, area]
    lefts = stats[1:, cv2.CC_STAT_LEFT]  # Remove background (index 0)
    areas = stats[1:, cv2.CC_STAT_AREA]

    # ① Touching left border (x == 0) 的组件
    border_ids = np.where(lefts == 0)[0]

    if len(border_ids):
        # If multiple, select the largest area
        idx_rel = border_ids[areas[border_ids].argmax()]
    else:
        # ② Otherwise select the one with smallest x
        idx_rel = lefts.argmin()

    idx_abs = idx_rel + 1  # Add back background offset

    # —— Generate output mask ——
    out = lbl == idx_abs
    return out.astype(filled_mask.dtype)


def _lab_medoid_color(pixels_bgr: np.ndarray) -> np.ndarray:
    """
    Get the medoid of pixel set in Lab space。
    Returns uint8 BGR(3,)。
    """

    lab = (
        cv2.cvtColor(pixels_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB)
        .reshape(-1, 3)
        .astype(np.float32)
    )

    d2 = np.sum((lab[:, None, :] - lab[None, :, :]) ** 2, axis=2)
    medoid_idx = np.argmin(d2.sum(axis=1))

    return pixels_bgr[medoid_idx]


def get_main_color(
    roi_bgr,
    mask,
    erode_iters: int = 0,
    sat_thresh: int = 50,
    k: int = 2,
    min_keep: int = 20,
    use_lab_medoid: bool = True,
):
    m = mask.astype(np.uint8)
    if erode_iters:
        m = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=erode_iters)
    m = m.astype(bool)

    if m.sum() < min_keep:
        m = mask.astype(bool)

    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[..., 1][m]
    thr = min(sat_thresh, np.percentile(sat, 50))
    keep = m & (hsv[..., 1] > thr)

    pixels = roi_bgr[keep] if keep.any() else roi_bgr[m]

    if pixels.shape[0] < 20 or k <= 1:
        return (
            _lab_medoid_color(pixels)
            if use_lab_medoid
            else np.median(pixels, axis=0).astype(np.uint8)
        )

    # k-means clustering
    Z = pixels.astype(np.float32)
    _crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(Z, k, None, _crit, 3, cv2.KMEANS_PP_CENTERS)
    main_cluster = centers[np.bincount(labels.flatten()).argmax()].astype(np.uint8)

    if use_lab_medoid:
        cluster_pixels = pixels[
            labels.flatten() == np.bincount(labels.flatten()).argmax()
        ]
        return _lab_medoid_color(cluster_pixels)
    else:
        return main_cluster


def extract_legend_colors(
    image: Union[str, np.ndarray],
    legend_bboxes: List[Tuple[int, int, int, int]],
) -> List[Tuple[int, int, int]]:
    img_bgr = cv2.imread(image) if isinstance(image, str) else image.copy()
    h, w = img_bgr.shape[:2]
    colors = []

    for x0, y0, x1, y1 in legend_bboxes:
        # Clamp coordinates to image bounds
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)

        if x0 >= x1 or y0 >= y1:
            continue

        roi = img_bgr[y0:y1, x0:x1]
        mask = adaptive_color_mask(roi)

        # Expand left if mask is too small
        for _ in range(5):
            if mask.sum() >= 5:
                break
            x0 = max(x0 - 10, 0)
            roi = img_bgr[y0:y1, x0:x1]
            mask = adaptive_color_mask(roi)

        # Expand borders if mask touches edges and is still small
        for _ in range(5):
            if mask.sum() >= 30:
                break
            expanded = False
            if mask[0, :].any():
                y0, expanded = max(y0 - 5, 0), True
            if mask[-1, :].any():
                y1, expanded = min(y1 + 5, h), True
            if mask[:, 0].any():
                x0, expanded = max(x0 - 5, 0), True

            if not expanded:
                break

            roi = img_bgr[y0:y1, x0:x1]
            mask = adaptive_color_mask(roi)

        mask = keep_leftmost_component(mask)

        if mask.sum() < 5:
            continue

        main_color = get_main_color(roi, mask, erode_iters=0, sat_thresh=30, k=2)
        colors.append(tuple(int(c) for c in main_color))

    return colors


def _bgr_list_to_lab(lst: List[Tuple[int, int, int]]) -> np.ndarray:
    if not lst:
        return np.empty((0, 3), np.float32)
    return _bgr_to_lab(np.array(lst, np.uint8))


def _chroma(lab: np.ndarray) -> np.ndarray:
    return np.linalg.norm(lab[:, 1:], axis=1)


def _flood_fill(
    lab_img: np.ndarray,
    seeds: np.ndarray,
    tol: Tuple[int, int, int],
    stop_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    H, W = lab_img.shape[:2]
    if not seeds.size:
        return np.zeros((H, W), bool)

    img_u8 = np.clip(lab_img, 0, 255).astype(np.uint8).copy()
    mask_ff = np.zeros((H + 2, W + 2), np.uint8)  # +2 边框

    # set stop mask in flood-fill mask
    if stop_mask is not None:
        if stop_mask.shape != (H, W):
            raise ValueError("stop_mask must have shape (H, W)")
        mask_ff[1:-1, 1:-1][stop_mask.astype(bool)] = 1

    flags = cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | 8 | (255 << 8)

    for r, c in seeds:
        # skip if in stop_mask
        if stop_mask is not None and stop_mask[r, c]:
            continue
        # skip if already filled or wall
        if mask_ff[r + 1, c + 1]:
            continue

        cv2.floodFill(
            image=img_u8,
            mask=mask_ff,
            seedPoint=(int(c), int(r)),
            newVal=(0, 0, 0),
            loDiff=tol,
            upDiff=tol,
            flags=flags,
        )

    # only keep regions marked as 255 in mask after flood-fill
    filled_only = mask_ff[1:-1, 1:-1] == 255
    return filled_only


def auto_thresholds_lab_v2(
    mask_bgrs: List[Tuple[int, int, int]],
    sel_bgrs: List[Tuple[int, int, int]],
    *,
    T_NEAR: float = 15.0,
) -> Dict[str, Any]:
    """
    Return automatically mapped thresholds based on ΔE between mask and selected colors.
    """
    mask_lab = _bgr_list_to_lab(mask_bgrs)
    sel_lab = _bgr_list_to_lab(sel_bgrs or [(0, 0, 0)])
    d_min = _delta_e(mask_lab, sel_lab).min().item()

    if d_min < T_NEAR:
        dist_thresh_seed = 8.0
        margin_from_selected = 6
        dist_thresh_lock = 6
        flood_tol = (5, 8, 8)

        c_thresh = max(_chroma(mask_lab).mean(), _chroma(sel_lab).mean())
        sat_thresh_seed = 0.25 * c_thresh if c_thresh > 15 else None

    else:
        dist_thresh_seed = float(np.clip(0.5 * d_min + 5, 5, 60))
        margin_from_selected = float(np.clip(0.1 * d_min, 5, 10))
        dist_thresh_lock = dist_thresh_seed * 0.6
        flood_tol = (
            int(np.clip(0.4 * d_min, 5, 10)),
            int(np.clip(0.6 * d_min, 10, 15)),
            int(np.clip(0.6 * d_min, 10, 15)),
        )
        c_thresh = max(_chroma(mask_lab).mean(), _chroma(sel_lab).mean())
        sat_thresh_seed = 0.25 * c_thresh if c_thresh > 25 else None

    return dict(
        dist_thresh_seed=dist_thresh_seed,
        margin_from_selected=margin_from_selected,
        dist_thresh_lock=dist_thresh_lock,
        sat_thresh_seed=sat_thresh_seed,
        flood_tol=flood_tol,
    )


def erode_reconstruct(
    mask: np.ndarray, ksize: int = 3, it: int = 1, keep_skeleton: bool = True
) -> np.ndarray:
    """
    Morphological erosion followed by geodesic dilation until stability.
    """
    # ker = cv2.getStructuringElement(cv2.MORPH_CROSS, (ksize, ksize))
    ker = np.ones((3, 3), np.uint8)
    marker = cv2.erode(mask, ker, iterations=it)

    if keep_skeleton:
        # 1. Skeletonization (uint8 0/255)
        skel = cv2.ximgproc.thinning(mask)
        marker = cv2.bitwise_or(marker, skel)

    # 2. geodesic dilation until stability
    while True:
        prev = marker.copy()
        marker = cv2.dilate(marker, ker, iterations=1)
        marker &= mask
        if np.array_equal(marker, prev):
            break
    return marker


def chroma_mag_ycrcb(bgr: np.ndarray) -> np.ndarray:
    """
    calc chroma magnitude in YCrCb space.
    """
    # bgr: (N,3) uint8
    ycrcb = cv2.cvtColor(bgr.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2YCrCb)
    _, Cr, Cb = cv2.split(ycrcb)
    chroma = cv2.magnitude(Cr.astype(np.float32) - 128, Cb.astype(np.float32) - 128)
    return chroma.reshape(bgr.shape[:-1])


def are_all_vivid(mask_legend_bgrs, *, chroma_thresh: float = 20.0) -> bool:
    """
    judge if all colors in mask_legend_bgrs are vivid (high chroma)。
    """
    bgr_arr = np.asarray(mask_legend_bgrs, dtype=np.uint8)
    chroma = chroma_mag_ycrcb(bgr_arr)
    return np.all(chroma > chroma_thresh)


# ──────────────────────── Main function (Lab only) ──────────────────────────
def compute_mask_region(
    img_path: str | Path,
    mask_legend_bgrs: List[Tuple[int, int, int]],
    selected_legend_bgrs: List[Tuple[int, int, int]] | None = None,
    *,
    dist_thresh_seed: float | None = None,
    margin_from_selected: float | None = None,
    dist_thresh_lock: float | None = None,
    sat_thresh_seed: float | None = None,
    flood_tol: Tuple[int, int, int] | None = None,
    core_ratio: float = 0.4,
    extend_to_light_area: bool = False,
    min_conn_pixels: int = 2,
    max_conn_ratio: float = 1,
    drop_border: bool = False,
    dilate_iter: int = 3,
    close_iter: int = 3,
    erode_iter: int = 1,
    kernel_size: int = 5,
) -> np.ndarray:
    """
    Return True for pixels to be masked; all operations are done in CIELab space.
    If dist_thresh_seed=None etc. are passed, thresholds are automatically set based on mask/selected ΔE.
    """
    if not mask_legend_bgrs:
        raise ValueError("mask_legend_bgrs cannot be empty.")
    if kernel_size % 2 == 0 or kernel_size < 3:
        raise ValueError("kernel_size must be odd and ≥3.")

    selected_legend_bgrs = selected_legend_bgrs or []

    # ---------- If thresholds are None → auto-generate ----------
    if None in (dist_thresh_seed, margin_from_selected, dist_thresh_lock, flood_tol):
        auto = auto_thresholds_lab_v2(mask_legend_bgrs, selected_legend_bgrs)
        dist_thresh_seed = dist_thresh_seed or auto["dist_thresh_seed"]
        margin_from_selected = margin_from_selected or auto["margin_from_selected"]
        dist_thresh_lock = dist_thresh_lock or auto["dist_thresh_lock"]
        sat_thresh_seed = (
            sat_thresh_seed if sat_thresh_seed is not None else auto["sat_thresh_seed"]
        )
        flood_tol = flood_tol or auto["flood_tol"]

    if extend_to_light_area:
        flood_tol = (75, 20, 20)

    # ---------- Read image & convert to Lab ----------
    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(img_path)
    H, W = bgr.shape[:2]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    flat = lab.reshape(-1, 3)
    coords = np.column_stack(np.unravel_index(np.arange(flat.shape[0]), (H, W)))

    msk_c = _bgr_list_to_lab(mask_legend_bgrs)
    sel_c = _bgr_list_to_lab(selected_legend_bgrs)

    d_mask = _delta_e(flat, msk_c).min(1)
    d_sel = np.full(flat.shape[0], np.inf, np.float32)
    if sel_c.size:
        d_sel = _delta_e(flat, sel_c).min(1)

    seed_mask = (d_mask <= dist_thresh_seed) & (d_mask + margin_from_selected < d_sel)
    if sat_thresh_seed is not None:
        c_flat = np.linalg.norm(flat[:, 1:], axis=1)
        seed_mask &= c_flat >= sat_thresh_seed

    if seed_mask.any():
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        Y, Cr, Cb = cv2.split(ycrcb)
        chroma = cv2.magnitude(Cr.astype(np.float32) - 128, Cb.astype(np.float32) - 128)
        obvious_chroma = 40
        obvious_ratio = 0.01

        has_obvious = (chroma > obvious_chroma).mean() > obvious_ratio

        has_obvious = are_all_vivid(mask_legend_bgrs)

        if has_obvious & ~extend_to_light_area:
            chroma_p95 = np.percentile(chroma, 95)
            chroma_th = max(25, min(30, chroma_p95))
            low_chroma = chroma <= chroma_th

            y_lo = 50
            y_hi = 50
            y_black_hi = 50

            gray_mask = low_chroma & ((Y <= y_lo) | (Y >= y_hi))
            black_white_mask = low_chroma & ((Y <= 50) | (Y >= 220))
        else:
            ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
            Y, Cr, Cb = cv2.split(ycrcb)
            Cr = Cr.astype(np.int16) - 128
            Cb = Cb.astype(np.int16) - 128

            # ---------- Adaptive threshold ----------
            cr_p10 = np.percentile(np.abs(Cr), 10)
            cb_p10 = np.percentile(np.abs(Cb), 10)
            delta = int(min(10, max(cr_p10, cb_p10)))

            low_chroma = (np.abs(Cr) <= delta) & (np.abs(Cb) <= delta)

            # ---------- Brightness threshold ----------
            y_lo, y_hi = 50, 50
            gray_mask = low_chroma & ((Y <= y_lo) | (Y >= y_hi))
            black_white_mask = low_chroma & ((Y <= 50) | (Y >= 220))

        # === 3. Update seed_mask ===
        seed_mask &= ~gray_mask.ravel()

    seed_coords = coords[seed_mask]

    if not seed_coords.size:
        return np.zeros((H, W), bool)

    lock_mask = (d_sel.reshape(H, W) <= dist_thresh_lock) & (
        d_sel.reshape(H, W) + 5 < d_mask.reshape(H, W)
    )
    seed_mask &= ~lock_mask.ravel()

    core_tol = tuple(int(t * core_ratio) for t in flood_tol)
    core = _flood_fill(
        lab.copy(), seed_coords, core_tol, stop_mask=black_white_mask.reshape(H, W)
    )
    grown = (
        _flood_fill(
            lab.copy(),
            np.column_stack(np.nonzero(core)),
            flood_tol,
            stop_mask=black_white_mask.reshape(H, W),
        )
        | core
    )

    if kernel_size and (dilate_iter or close_iter or erode_iter):
        ker = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        g = grown.astype(np.uint8)
        if close_iter:
            g = cv2.morphologyEx(g, cv2.MORPH_CLOSE, ker, close_iter)
        if erode_iter:
            g = erode_reconstruct(g, ksize=3, it=erode_iter)

        if dilate_iter:
            g = cv2.dilate(g, ker, dilate_iter)
        grown = g.astype(bool)

    # ---------- Connected components filtering ----------
    if min_conn_pixels > 1 or max_conn_ratio < 1.0 or drop_border:
        g = grown.astype(np.uint8)
        n, lbl = cv2.connectedComponents(g, 8)
        keep = np.zeros_like(g, bool)
        border = np.zeros_like(g, bool)
        border[[0, -1], :] = True
        border[:, [0, -1]] = True
        max_px = int(H * W * max_conn_ratio)

        for i in range(1, n):
            comp = lbl == i
            area = comp.sum()
            if area < min_conn_pixels or area > max_px:
                continue
            if drop_border and np.any(comp & border):
                continue
            keep |= comp

        grown = keep

    # ---------- Final removal of selected ----------
    if sel_c.size:
        use_conn_lock = True
        flood_tol_lock = tuple(t // 2 for t in flood_tol)
        core_ratio_lock = 0.3
        dist_thresh_seed_lock = dist_thresh_lock
        d_sel_full = _delta_e(flat, sel_c).min(1)

        eps = 6
        seed_lock = (d_sel_full <= dist_thresh_seed_lock) & (d_sel_full + eps < d_mask)

        remove_mask = (d_mask <= dist_thresh_seed) & (d_mask + eps < d_sel_full)
        rm_2d = remove_mask.reshape(H, W)

        radius = 1
        ker = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
        )

        ker_lock = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        rm_dil = cv2.morphologyEx(
            rm_2d.astype(np.uint8), cv2.MORPH_CLOSE, ker_lock, iterations=1
        )
        rm_dil = erode_reconstruct(rm_dil, ksize=3, it=1)
        rm_dil = cv2.dilate(rm_dil, ker, iterations=1)

        remove_mask = rm_dil.astype(bool).ravel()
        seed_lock &= ~remove_mask
        if ((seed_lock.sum() / grown.sum()) < 0.02) and (not extend_to_light_area):
            flood_tol_lock = (30, 30, 30)
        elif ((seed_lock.sum() / grown.sum()) < 0.002) and extend_to_light_area:
            flood_tol_lock = (20, 20, 20)
        else:
            flood_tol_lock = (1, 2, 2)

        seed_lock_coords = coords[seed_lock]

        if use_conn_lock and seed_lock_coords.size:
            core_tol_lock = tuple(int(t * core_ratio_lock) for t in flood_tol_lock)
            core_lock = _flood_fill(lab.copy(), seed_lock_coords, core_tol_lock)
            lock = (
                _flood_fill(
                    lab.copy(), np.column_stack(np.nonzero(core_lock)), flood_tol_lock
                )
                | core_lock
            ).astype(np.uint8)
        else:
            lock = seed_lock.astype(np.uint8)

        # --- Update lock mask with morphology ---
        ker_lock = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        if lock.ndim == 1:
            lock = lock.reshape(H, W)
        if close_iter:
            lock = cv2.morphologyEx(
                lock, cv2.MORPH_CLOSE, ker_lock, iterations=close_iter
            )
        if erode_iter:
            g = erode_reconstruct(g, ksize=3, it=erode_iter)
        if dilate_iter:
            lock = cv2.dilate(lock, ker_lock, iterations=1)

        lock = lock.astype(bool)
        grown &= ~lock

    return grown


def _bgr_to_lab(bgr: np.ndarray) -> np.ndarray:
    """(N,3)[uint8] → (N,3)[float32] / Lab"""
    return (
        cv2.cvtColor(bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB)
        .reshape(-1, 3)
        .astype(np.float32)
    )


def _delta_e(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)


def _delta_e_lab(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """
    CIE-76 ΔE。
    lab1: (N,3) 或 (H,W,3)
    lab2: (M,3) 或 (3,)
    """
    diff = lab1[..., None, :] - lab2  # → (..., M, 3)
    return np.sqrt(np.sum(diff**2, axis=-1))


def bgr2lab_real(bgr: np.ndarray) -> np.ndarray:
    """BGR → Lab (real range)"""
    lab_cv = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_cv[..., 0] *= 100 / 255.0  # L*  0-100
    lab_cv[..., 1:] -= 128.0  # a*, b*  -128 – 127
    return lab_cv


def get_center_colors_auto_merge(
    img_path: str,
    selected_legend_bgrs: List[Tuple[int, int, int]],
    *,
    Y_hi: int = 230,
    exclude_delta_e: float = 35,
    merge_thr: float = 35,
    max_k: int = 10,
    min_pixels: int = 10,
) -> List[Tuple[int, int, int]]:
    """
    return list of main colors in BGR format.
    """
    bgr = cv2.imread(img_path)
    if bgr is None:
        raise FileNotFoundError(img_path)
    H, W = bgr.shape[:2]

    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)
    chroma = cv2.magnitude(Cr.astype(np.float32) - 128, Cb.astype(np.float32) - 128)

    chroma_th_low = 10
    chroma_th_high = 30
    obvious_chroma = 40
    obvious_ratio = 1e-4
    has_obvious = (chroma > obvious_chroma).mean() > obvious_ratio
    chroma_th = chroma_th_high if has_obvious else chroma_th_low
    mask = (chroma > chroma_th) & (Y < Y_hi)  # bool (H,W)

    # exclude selected colors
    if selected_legend_bgrs:
        sel_lab = bgr2lab_real(
            np.asarray(selected_legend_bgrs, np.uint8).reshape(-1, 1, 3)
        ).reshape(-1, 3)  # (K,3)
        dsel = _delta_e_lab(
            bgr2lab_real(bgr),
            sel_lab,  # (H,W,K)
        ).min(axis=-1)
        mask &= dsel > exclude_delta_e

    if not np.any(mask):
        return []

    # connected components: remove border-touching and small areas
    mask_u8 = mask.astype(np.uint8)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    drop_idx: List[int] = []
    for i in range(1, n):  # 0 是背景
        x, y, w, h, area = stats[i]
        touch_border = (x == 0) | (y == 0) | (x + w == W) | (y + h == H)
        if touch_border or area < min_pixels:
            drop_idx.append(i)
    for i in drop_idx:
        mask[lbl == i] = False

    if not np.any(mask):
        return []

    # exclude edge light colors / noise
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    edge_px = 1.0  # edge shrink thickness, adjustable
    thin_thr = 1.5  # connected component max_dist < thin_thr → considered thin line, skip shrink
    if dist.max() >= thin_thr:  # only shrink thick color blocks
        core = dist >= edge_px  # ≥1 px considered core pixel
        mask &= core

    if not np.any(mask):
        return []

    # Morphology: opening + light erosion
    ker_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    ker_erod = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, ker_open, 1)
    mask = cv2.erode(mask, ker_erod, 1).astype(bool)

    if not np.any(mask):
        return []

    # K-Means clustering
    pixels = bgr[mask].astype(np.float32)  # (N,3)
    N = len(pixels)
    k0 = min(max_k, max(2, int(np.sqrt(N / 500))))
    k0 = min(k0, N)
    kmeans = KMeans(n_clusters=k0, n_init="auto", random_state=0).fit(pixels)
    centers_bgr = kmeans.cluster_centers_  # (k0,3)
    counts = np.bincount(kmeans.labels_, minlength=k0)

    centers_lab = bgr2lab_real(centers_bgr.astype(np.uint8)[None])[0]

    # recursively merge similar clusters
    while len(centers_lab) > 1:
        d = _delta_e_lab(centers_lab, centers_lab)
        np.fill_diagonal(d, np.inf)
        i, j = np.unravel_index(np.argmin(d), d.shape)
        if d[i, j] >= merge_thr:
            break
        n_i, n_j = counts[i], counts[j]
        new_bgr = (centers_bgr[i] * n_i + centers_bgr[j] * n_j) / (n_i + n_j)
        new_lab = (centers_lab[i] * n_i + centers_lab[j] * n_j) / (n_i + n_j)

        keep = np.ones(len(centers_lab), bool)
        keep[[i, j]] = False
        centers_bgr = np.vstack([centers_bgr[keep], new_bgr])
        centers_lab = np.vstack([centers_lab[keep], new_lab])
        counts = np.concatenate([counts[keep], [n_i + n_j]])

    # output: sort by cluster size & exclude selected colors
    order = counts.argsort()[::-1]
    sel_lab = bgr2lab_real(
        np.asarray(selected_legend_bgrs, np.uint8).reshape(-1, 1, 3)
    ).reshape(-1, 3)

    out: List[Tuple[int, int, int]] = []
    for idx in order:
        c_bgr = centers_bgr[idx]
        c_lab = bgr2lab_real(np.uint8([[c_bgr]]))[0, 0]
        if _delta_e_lab(c_lab, sel_lab).min() > exclude_delta_e:
            out.append(tuple(map(int, c_bgr)))

    return out


def mask_chart_legend(
    image_path,
    mask_legend_bboxes,
    selected_legend_bboxes,
    output_path,
    kenel_size=5,
    hsv_tolerance=(10, 40, 40),
    mask_padding=0,
    orig_image_path=None,
    return_mask=False,
    extend_to_light_area=False,
):
    img_bgr = cv2.imread(image_path)
    if orig_image_path is None:
        orig_image_path = image_path
    if len(selected_legend_bboxes) == 0:
        raise ValueError("Bounding boxes of the focused legend cannot be found.")
    else:
        selected_legend_bgrs = extract_legend_colors(
            orig_image_path, selected_legend_bboxes
        )

    if len(mask_legend_bboxes) == 0:
        mask_legend_bgrs = get_center_colors_auto_merge(
            image_path, selected_legend_bgrs
        )
    else:
        mask_legend_bgrs = extract_legend_colors(orig_image_path, mask_legend_bboxes)

    if len(mask_legend_bgrs) == 0 and len(mask_legend_bboxes) == 0:
        raise ValueError(
            "No masked legend found. Please list all the other legend names besides the focused ones."
        )
    if len(mask_legend_bgrs) == 0 or len(selected_legend_bgrs) == 0:
        raise ValueError("Cannot get the legend colors from the image.")

    b_color = get_background_color(image_path=image_path)

    final_mask = compute_mask_region(
        image_path,
        mask_legend_bgrs,
        selected_legend_bgrs,
        extend_to_light_area=extend_to_light_area,
    )

    if return_mask:
        return final_mask

    # update image with mask
    img_bgr[final_mask > 0] = np.array(b_color, np.uint8)
    if output_path:
        cv2.imwrite(output_path, img_bgr)
    return output_path
