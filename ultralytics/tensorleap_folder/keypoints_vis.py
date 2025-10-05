import numpy as np
import cv2

try:
    from PIL import Image
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False

# === Ultralytics pose palette (RGB) ===
_POSE_PALETTE = np.array([
    [255, 128, 0],  [255, 153, 51],  [255, 178, 102], [230, 230, 0],
    [255, 153, 255],[153, 204, 255],[255, 102, 255], [255, 51, 255],
    [102, 178, 255],[51, 153, 255], [255, 153, 153],[255, 102, 102],
    [255, 51, 51],  [153, 255, 153],[102, 255, 102], [51, 255, 51],
    [0, 255, 0],    [0, 0, 255],    [255, 0, 0],     [255, 255, 255],
], dtype=np.uint8)

# === Ultralytics skeleton (1-indexed in the class; we keep 1-index here to mirror logic) ===
# Will subtract 1 when indexing into kpts
_ULTRA_SKELETON_1IDX = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
    [6, 12],  [7, 13],  [6, 7],   [6, 8],   [7, 9],
    [8, 10],  [9, 11],  [2, 3],   [1, 2],   [1, 3],
    [2, 4],   [3, 5],   [4, 6],   [5, 7],
]

# === Ultralytics limb and keypoint color selections ===
_LIMB_COLOR = _POSE_PALETTE[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
_KPT_COLOR  = _POSE_PALETTE[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]

def draw_ultralytics_keypoints(
    image,
    kpts,
    shape=(640, 640),
    radius=None,
    kpt_line=True,
    conf_thres=0.25,
    kpt_color=None,   # optional override: (B,G,R)
):
    """
    Independent function that mirrors Ultralytics Annotator.kpts() behavior.

    Args:
        image: np.ndarray BGR (HxWx3) or PIL.Image (RGB).
        kpts: array-like [17,2] or [17,3] with (x, y, [conf]).
        shape: (H, W) used only for boundary checks (same as Ultralytics).
        radius: circle radius; auto-scales if None like Ultralytics' line width heuristic.
        kpt_line: draw skeleton lines (Ultralytics 17-key human pose only).
        conf_thres: confidence threshold when kpts include conf.
        kpt_color: optional single (B,G,R) color for points & limbs (overrides palettes).

    Returns:
        Image in the same type as input (np.ndarray BGR or PIL.Image).
    """
    # --- normalize image to cv2-friendly array and remember original type ---
    pil_input = _HAS_PIL and isinstance(image, Image.Image)
    if pil_input:
        np_im = np.asarray(image)  # RGB
        if np_im.ndim != 3 or np_im.shape[2] != 3:
            raise ValueError("PIL image must be RGB.")
        np_im = np_im[:, :, ::-1].copy()  # -> BGR for cv2
    else:
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a NumPy array (BGR) or a PIL.Image.")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("NumPy image must be HxWx3.")
        np_im = image.copy()

    H, W = np_im.shape[:2]

    # --- keypoints array ---
    if hasattr(kpts, "detach"):  # torch.Tensor
        kpts = kpts.detach().cpu().numpy()
    else:
        kpts = np.asarray(kpts)
    if kpts.shape[0] != 17 or kpts.shape[1] not in (2, 3):
        raise ValueError(f"kpts must be shape [17,2] or [17,3]. got {kpts.shape}")

    has_conf = (kpts.shape[1] == 3)

    # --- line width & radius to mimic Ultralytics scaling ---
    lw = max(round((H + W) / 2 * 0.003), 2)
    radius = radius if radius is not None else lw

    # --- draw keypoints (Ultralytics logic for bounds/conf) ---
    for i, k in enumerate(kpts):
        x_coord, y_coord = float(k[0]), float(k[1])
        # same boundary style: skip points lying exactly on image edges via modulo test against model 'shape'
        if (x_coord % shape[1] != 0) and (y_coord % shape[0] != 0):
            if has_conf and float(k[2]) < conf_thres:
                continue
            col = (kpt_color if kpt_color is not None
                   else tuple(int(c) for c in _KPT_COLOR[i][::-1]))  # RGB->BGR
            cv2.circle(np_im, (int(x_coord), int(y_coord)), radius, col, -1, lineType=cv2.LINE_AA)

    # --- draw skeleton (uses 1-indexed pairs in the original; subtract 1 here) ---
    if kpt_line:
        for idx, (a1, a2) in enumerate(_ULTRA_SKELETON_1IDX):
            i1, i2 = a1 - 1, a2 - 1
            x1, y1 = int(kpts[i1, 0]), int(kpts[i1, 1])
            x2, y2 = int(kpts[i2, 0]), int(kpts[i2, 1])

            if has_conf:
                if kpts[i1, 2] < conf_thres or kpts[i2, 2] < conf_thres:
                    continue

            # replicate the edge checks from Ultralytics
            if x1 % shape[1] == 0 or y1 % shape[0] == 0 or x1 < 0 or y1 < 0:
                continue
            if x2 % shape[1] == 0 or y2 % shape[0] == 0 or x2 < 0 or y2 < 0:
                continue

            col = (kpt_color if kpt_color is not None
                   else tuple(int(c) for c in _LIMB_COLOR[idx][::-1]))  # RGB->BGR
            cv2.line(np_im, (x1, y1), (x2, y2), col, thickness=int(np.ceil(lw / 2)), lineType=cv2.LINE_AA)

    # --- back to original type ---
    if pil_input:
        return Image.fromarray(np_im[:, :, ::-1])  # BGR->RGB -> PIL
    return np_im
