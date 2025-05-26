# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import Tensor

def box_iou_rotated(bboxes1: torch.Tensor,
                    bboxes2: torch.Tensor,
                    mode: str = 'iou',
                    aligned: bool = False,
                    clockwise: bool = True) -> torch.Tensor:
    """Return intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in
    (x_center, y_center, width, height, angle) format.

    If ``aligned`` is ``False``, then calculate the ious between each bbox
    of bboxes1 and bboxes2, otherwise the ious between each aligned pair of
    bboxes1 and bboxes2.

    .. note::
        The operator assumes:

        1) The positive direction along x axis is left -> right.

        2) The positive direction along y axis is top -> down.

        3) The w border is in parallel with x axis when angle = 0.

        However, there are 2 opposite definitions of the positive angular
        direction, clockwise (CW) and counter-clockwise (CCW). MMCV supports
        both definitions and uses CW by default.

        Please set ``clockwise=False`` if you are using the CCW definition.

        The coordinate system when ``clockwise`` is ``True`` (default)

            .. code-block:: none

                0-------------------> x (0 rad)
                |  A-------------B
                |  |             |
                |  |     box     h
                |  |   angle=0   |
                |  D------w------C
                v
                y (pi/2 rad)

            In such coordination system the rotation matrix is

            .. math::
                \\begin{pmatrix}
                \\cos\\alpha & -\\sin\\alpha \\\\
                \\sin\\alpha & \\cos\\alpha
                \\end{pmatrix}

            The coordinates of the corner point A can be calculated as:

            .. math::
                P_A=
                \\begin{pmatrix} x_A \\\\ y_A\\end{pmatrix}
                =
                \\begin{pmatrix} x_{center} \\\\ y_{center}\\end{pmatrix} +
                \\begin{pmatrix}\\cos\\alpha & -\\sin\\alpha \\\\
                \\sin\\alpha & \\cos\\alpha\\end{pmatrix}
                \\begin{pmatrix} -0.5w \\\\ -0.5h\\end{pmatrix} \\\\
                =
                \\begin{pmatrix} x_{center}-0.5w\\cos\\alpha+0.5h\\sin\\alpha
                \\\\
                y_{center}-0.5w\\sin\\alpha-0.5h\\cos\\alpha\\end{pmatrix}


        The coordinate system when ``clockwise`` is ``False``

            .. code-block:: none

                0-------------------> x (0 rad)
                |  A-------------B
                |  |             |
                |  |     box     h
                |  |   angle=0   |
                |  D------w------C
                v
                y (-pi/2 rad)

            In such coordination system the rotation matrix is

            .. math::
                \\begin{pmatrix}
                \\cos\\alpha & \\sin\\alpha \\\\
                -\\sin\\alpha & \\cos\\alpha
                \\end{pmatrix}

            The coordinates of the corner point A can be calculated as:

            .. math::
                P_A=
                \\begin{pmatrix} x_A \\\\ y_A\\end{pmatrix}
                =
                \\begin{pmatrix} x_{center} \\\\ y_{center}\\end{pmatrix} +
                \\begin{pmatrix}\\cos\\alpha & \\sin\\alpha \\\\
                -\\sin\\alpha & \\cos\\alpha\\end{pmatrix}
                \\begin{pmatrix} -0.5w \\\\ -0.5h\\end{pmatrix} \\\\
                =
                \\begin{pmatrix} x_{center}-0.5w\\cos\\alpha-0.5h\\sin\\alpha
                \\\\
                y_{center}+0.5w\\sin\\alpha-0.5h\\cos\\alpha\\end{pmatrix}

    Args:
        boxes1 (torch.Tensor): rotated bboxes 1. It has shape (N, 5),
            indicating (x, y, w, h, theta) for each row. Note that theta is in
            radian.
        boxes2 (torch.Tensor): rotated bboxes 2. It has shape (M, 5),
            indicating (x, y, w, h, theta) for each row. Note that theta is in
            radian.
        mode (str): "iou" (intersection over union) or iof (intersection over
            foreground).
        clockwise (bool): flag indicating whether the positive angular
            orientation is clockwise. default True.
            `New in version 1.4.3.`

    Returns:
        torch.Tensor: Return the ious betweens boxes. If ``aligned`` is
        ``False``, the shape of ious is (N, M) else (N,).
    """
    assert mode in ['iou', 'iof']
    mode_dict = {'iou': 0, 'iof': 1}
    mode_flag = mode_dict[mode]
    rows = bboxes1.size(0)
    cols = bboxes2.size(0)
    if aligned:
        ious = bboxes1.new_zeros(rows)
    else:
        if bboxes1.device.type == 'mlu':
            ious = bboxes1.new_zeros([rows, cols])
        else:
            ious = bboxes1.new_zeros(rows * cols)
    if not clockwise:
        flip_mat = bboxes1.new_ones(bboxes1.shape[-1])
        flip_mat[-1] = -1
        bboxes1 = bboxes1 * flip_mat
        bboxes2 = bboxes2 * flip_mat
    if bboxes1.device.type == 'npu':
        scale_mat = bboxes1.new_ones(bboxes1.shape[-1])
        scale_mat[-1] = 1.0 / 0.01745329252
        bboxes1 = bboxes1 * scale_mat
        bboxes2 = bboxes2 * scale_mat
    bboxes1 = bboxes1.contiguous()
    bboxes2 = bboxes2.contiguous()
    # ext_module.box_iou_rotated(
    #     bboxes1, bboxes2, ious, mode_flag=mode_flag, aligned=aligned)
    if not aligned:
        ious = ious.view(rows, cols)
    return ious

def points_in_boxes_part(points: Tensor, boxes: Tensor) -> Tensor:
    """Find the box in which each point is (CUDA).

    Args:
        points (torch.Tensor): [B, M, 3], [x, y, z] in LiDAR/DEPTH coordinate.
        boxes (torch.Tensor): [B, T, 7],
            num_valid_boxes <= T, [x, y, z, x_size, y_size, z_size, rz] in
            LiDAR/DEPTH coordinate, (x, y, z) is the bottom center.

    Returns:
        torch.Tensor: Return the box indices of points with the shape of
        (B, M). Default background = -1.
    """
    assert points.shape[0] == boxes.shape[0], \
        'Points and boxes should have the same batch size, ' \
        f'but got {points.shape[0]} and {boxes.shape[0]}'
    assert boxes.shape[2] == 7, \
        'boxes dimension should be 7, ' \
        f'but got unexpected shape {boxes.shape[2]}'
    assert points.shape[2] == 3, \
        'points dimension should be 3, ' \
        f'but got unexpected shape {points.shape[2]}'
    batch_size, num_points, _ = points.shape

    box_idxs_of_pts = points.new_zeros((batch_size, num_points),
                                       dtype=torch.int).fill_(-1)

    # If manually put the tensor 'points' or 'boxes' on a device
    # which is not the current device, some temporary variables
    # will be created on the current device in the cuda op,
    # and the output will be incorrect.
    # Therefore, we force the current device to be the same
    # as the device of the tensors if it was not.
    # Please refer to https://github.com/open-mmlab/mmdetection3d/issues/305
    # for the incorrect output before the fix.
    points_device = points.get_device()
    assert points_device == boxes.get_device(), \
        'Points and boxes should be put on the same device'
    if torch.cuda.current_device() != points_device:
        torch.cuda.set_device(points_device)

    # ext_module.points_in_boxes_part_forward(boxes.contiguous(),
    #                                         points.contiguous(),
    #                                         box_idxs_of_pts)

    return box_idxs_of_pts


def points_in_boxes_cpu(points: Tensor, boxes: Tensor) -> Tensor:
    """Find all boxes in which each point is (CPU). The CPU version of
    :meth:`points_in_boxes_all`.

    Args:
        points (torch.Tensor): [B, M, 3], [x, y, z] in
            LiDAR/DEPTH coordinate
        boxes (torch.Tensor): [B, T, 7],
            num_valid_boxes <= T, [x, y, z, x_size, y_size, z_size, rz],
            (x, y, z) is the bottom center.

    Returns:
        torch.Tensor: Return the box indices of points with the shape of
        (B, M, T). Default background = 0.
    """
    assert points.shape[0] == boxes.shape[0], \
        'Points and boxes should have the same batch size, ' \
        f'but got {points.shape[0]} and {boxes.shape[0]}'
    assert boxes.shape[2] == 7, \
        'boxes dimension should be 7, ' \
        f'but got unexpected shape {boxes.shape[2]}'
    assert points.shape[2] == 3, \
        'points dimension should be 3, ' \
        f'but got unexpected shape {points.shape[2]}'
    batch_size, num_points, _ = points.shape
    num_boxes = boxes.shape[1]

    point_indices = points.new_zeros((batch_size, num_boxes, num_points),
                                     dtype=torch.int)
    # for b in range(batch_size):
    #     ext_module.points_in_boxes_cpu_forward(boxes[b].float().contiguous(),
    #                                            points[b].float().contiguous(),
    #                                            point_indices[b])
    point_indices = point_indices.transpose(1, 2)

    return point_indices


def points_in_boxes_all(points: Tensor, boxes: Tensor) -> Tensor:
    """Find all boxes in which each point is (CUDA).

    Args:
        points (torch.Tensor): [B, M, 3], [x, y, z] in LiDAR/DEPTH coordinate
        boxes (torch.Tensor): [B, T, 7],
            num_valid_boxes <= T, [x, y, z, x_size, y_size, z_size, rz],
            (x, y, z) is the bottom center.

    Returns:
        torch.Tensor: Return the box indices of points with the shape of
        (B, M, T). Default background = 0.
    """
    assert boxes.shape[0] == points.shape[0], \
        'Points and boxes should have the same batch size, ' \
        f'but got {boxes.shape[0]} and {boxes.shape[0]}'
    assert boxes.shape[2] == 7, \
        'boxes dimension should be 7, ' \
        f'but got unexpected shape {boxes.shape[2]}'
    assert points.shape[2] == 3, \
        'points dimension should be 3, ' \
        f'but got unexpected shape {points.shape[2]}'
    batch_size, num_points, _ = points.shape
    num_boxes = boxes.shape[1]

    box_idxs_of_pts = points.new_zeros((batch_size, num_points, num_boxes),
                                       dtype=torch.int).fill_(0)

    # Same reason as line 25-32
    points_device = points.get_device()
    assert points_device == boxes.get_device(), \
        'Points and boxes should be put on the same device'
    if torch.cuda.current_device() != points_device:
        torch.cuda.set_device(points_device)

    # ext_module.points_in_boxes_all_forward(boxes.contiguous(),
    #                                        points.contiguous(),
    #                                        box_idxs_of_pts)

    return box_idxs_of_pts
