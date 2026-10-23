"""box_draw.py — self-contained 3D-box drawing for the qualitative videos.

Pure numpy + cv2 (NO mmdet3d / mmengine), so overlays can be regenerated on any
machine from the saved videos + data.pkl.

The FRONT face of each box (the +x / heading direction) is filled translucently
so orientation is obvious on a 2D image; wireframe edges are drawn on top.

Corner order:
  0 b-front-left  1 b-front-right  2 b-rear-right  3 b-rear-left
  4 t-front-left  5 t-front-right  6 t-rear-right   7 t-rear-left
Front (heading, +x) face = corners [0, 1, 5, 4].
"""
import numpy as np
import cv2

# Fig-2 colors (RGB 0-1): Vehicle yellow, Pedestrian orange, Cyclist magenta
CLASS_COLORS = {
    0: (1.00, 0.85, 0.00),
    1: (1.00, 0.55, 0.00),
    2: (1.00, 0.00, 1.00),
}
FRONT_FACE = [0, 1, 5, 4]
_BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),   # bottom
    (4, 5), (5, 6), (6, 7), (7, 4),   # top
    (0, 4), (1, 5), (2, 6), (3, 7),   # verticals
]


def box_corners_3d(box):
    """8 corners (8,3) in LiDAR frame for [x,y,z,l,w,h,yaw] (z = bottom center)."""
    x, y, z, l, w, h, yaw = box[:7]
    x_c = np.array([l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2])
    y_c = np.array([w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2])
    z_c = np.array([0, 0, 0, 0, h, h, h, h])
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    corners = R @ np.vstack([x_c, y_c, z_c])
    corners[0, :] += x
    corners[1, :] += y
    corners[2, :] += z
    return corners.T


def project(corners_3d, lidar2img):
    """Project (N,3) LiDAR points to image (N,2) plus depth (N,)."""
    n = corners_3d.shape[0]
    pts4 = np.concatenate([corners_3d, np.ones((n, 1))], axis=1)
    l2i = np.array(lidar2img)
    if l2i.shape == (3, 4):
        l2i = np.concatenate([l2i, [[0, 0, 0, 1]]], axis=0)
    uvw = pts4 @ l2i.T
    depth = uvw[:, 2]
    uvw[:, 2] = np.clip(uvw[:, 2], 1e-5, 1e6)
    return np.stack([uvw[:, 0] / uvw[:, 2], uvw[:, 1] / uvw[:, 2]], axis=1), depth


def draw_boxes_on_image(img_bgr, boxes, labels, lidar2img,
                        thickness=2, fill_alpha=0.32):
    """Draw projected 3D boxes with a translucent shaded FRONT face. Returns BGR."""
    H, W = img_bgr.shape[:2]
    base = img_bgr.copy()
    overlay = img_bgr.copy()
    todo = []
    for box, lab in zip(boxes, labels):
        corners = box_corners_3d(box)
        pts2d, depth = project(corners, lidar2img)
        in_front = depth > 0.1
        if in_front.sum() < 2:
            continue
        col_rgb = CLASS_COLORS.get(int(lab), (1, 1, 1))
        col_bgr = tuple(int(255 * c) for c in col_rgb[::-1])
        todo.append((pts2d, in_front, col_bgr))

    def _sane(p):
        return abs(p[0]) < 5 * W and abs(p[1]) < 5 * H

    for pts2d, in_front, col_bgr in todo:        # pass 1: fill heading face
        if all(in_front[i] for i in FRONT_FACE):
            quad = pts2d[FRONT_FACE]
            if all(_sane(p) for p in quad):
                cv2.fillPoly(overlay, [quad.round().astype(np.int32)], col_bgr)
    img = cv2.addWeighted(overlay, fill_alpha, base, 1 - fill_alpha, 0)

    for pts2d, in_front, col_bgr in todo:        # pass 2: wireframe
        for a, b in _BOX_EDGES:
            if not (in_front[a] and in_front[b]):
                continue
            pa, pb = pts2d[a], pts2d[b]
            if not (_sane(pa) and _sane(pb)):
                continue
            cv2.line(img, (int(round(pa[0])), int(round(pa[1]))),
                     (int(round(pb[0])), int(round(pb[1]))),
                     col_bgr, thickness, cv2.LINE_AA)
    return img
