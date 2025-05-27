import argparse
import os
import os.path as osp
import sys # For logger
import logging # For logger

import torch
from mmengine.config import Config
from mmengine.registry import RUNNERS, TASK_UTILS
from mmengine.runner import Runner
from mmengine.utils import is_list_of

# It's generally better to set CUDA_VISIBLE_DEVICES outside the script
# e.g., CUDA_VISIBLE_DEVICES=1 python your_script.py
# However, if needed programmatically and early:
# os.environ['CUDA_VISIBLE_DEVICES'] = '1' # Set via CLI or env var is preferred

# Try to import mmdet3d and mmdet specific utilities
try:
    from mmdet3d.utils import replace_ceph_backend
    from mmdet3d.utils import register_all_modules as register_all_modules_3d
    register_all_modules_3d(init_default_scope=False)
except ImportError:
    logging.warning("mmdet3d.utils not found. Ensure mmdet3d is installed if 3D specific modules are critical.")
    replace_ceph_backend = None # Define a placeholder if not available

try:
    from mmdet.utils import register_all_modules as register_all_modules_det
    register_all_modules_det(init_default_scope=False)
except ImportError:
    logging.warning("mmdet.utils not found. Ensure mmdet is installed if 2D detection base modules are critical.")

# --- Custom Anchor Generator Classes (from your script) ---
# These classes seem well-defined for MMLab. Minor typing/docstring improvements.

@TASK_UTILS.register_module()
class Anchor3DRangeGenerator:
    """3D Anchor Generator by range.
    (Docstring from original script - consider expanding if needed)
    """
    def __init__(self,
                 ranges: List[List[float]],
                 sizes: List[List[float]] = [[3.9, 1.6, 1.56]],
                 scales: List[int] = [1],
                 rotations: List[float] = [0, 1.5707963],
                 custom_values: Tuple[float, ...] = (),
                 reshape_out: bool = True,
                 size_per_range: bool = True):
        assert is_list_of(ranges, list)
        if size_per_range:
            if len(sizes) != len(ranges):
                assert len(ranges) == 1, \
                    "If sizes and ranges have different lengths and size_per_range is True, ranges must have length 1 to be duplicated."
                ranges = ranges * len(sizes)
            assert len(ranges) == len(sizes), \
                "If size_per_range is True, ranges and sizes must have the same length after potential duplication of ranges."
        else:
            assert len(ranges) == 1, \
                "If size_per_range is False, ranges must have length 1."
        assert is_list_of(sizes, list)
        assert isinstance(scales, list)

        self.sizes = sizes
        self.scales = scales
        self.ranges = ranges
        self.rotations = rotations
        self.custom_values = custom_values # Tuple of floats
        self.reshape_out = reshape_out
        self.size_per_range = size_per_range
        # self.cached_anchors = None # Caching can be complex with device changes, MMLab might handle it.

    def __repr__(self) -> str:
        s = self.__class__.__name__ + '('
        s += f'anchor_range={self.ranges},\n'
        s += f'scales={self.scales},\n'
        s += f'sizes={self.sizes},\n'
        s += f'rotations={self.rotations},\n'
        s += f'custom_values={self.custom_values},\n'
        s += f'reshape_out={self.reshape_out},\n'
        s += f'size_per_range={self.size_per_range})'
        return s

    @property
    def num_base_anchors(self) -> int:
        """list[int]: Total number of base anchors in a feature grid."""
        num_rot = len(self.rotations)
        num_size = torch.tensor(self.sizes).reshape(-1, 3).size(0)
        return num_rot * num_size

    @property
    def num_levels(self) -> int:
        """int: Number of feature levels that the generator is applied to."""
        return len(self.scales)

    def grid_anchors(self, featmap_sizes: List[Tuple[int, ...]], device: str = 'cuda') -> List[torch.Tensor]:
        """Generate grid anchors in multiple feature levels."""
        assert self.num_levels == len(featmap_sizes), \
            f"Number of levels mismatch: {self.num_levels} vs {len(featmap_sizes)}"
        multi_level_anchors = []
        for i in range(self.num_levels):
            anchors = self.single_level_grid_anchors(
                featmap_sizes[i], self.scales[i], device=device)
            if self.reshape_out:
                anchors = anchors.reshape(-1, anchors.size(-1))
            multi_level_anchors.append(anchors)
        return multi_level_anchors

    def single_level_grid_anchors(self, featmap_size: Tuple[int, ...], scale: float, device: str = 'cuda') -> torch.Tensor:
        """Generate grid anchors of a single level feature map."""
        if not self.size_per_range:
            return self.anchors_single_range(
                featmap_size, self.ranges[0], scale, self.sizes, self.rotations, device=device
            )

        mr_anchors = []
        for anchor_range, anchor_size_group in zip(self.ranges, self.sizes): # anchor_size_group is a list of sizes
            mr_anchors.append(
                self.anchors_single_range(
                    featmap_size, anchor_range, scale, anchor_size_group, self.rotations, device=device
                )
            )
        # Concatenate along the dimension that represents different anchor types (num_sizes * num_rots)
        # Original was dim=-3 which corresponds to feature_size[0] (depth) if permuted later.
        # The output of anchors_single_range is [*feature_size, num_sizes_for_this_group, num_rots, 7(+C)]
        # We want to stack these different groups of sizes.
        return torch.cat(mr_anchors, dim=-3) # Stacking along the num_sizes dimension

    def anchors_single_range(self,
                             feature_size: Tuple[int, ...],
                             anchor_range: List[float],
                             scale: float = 1.0,
                             sizes: List[List[float]] = [[3.9, 1.6, 1.56]], # Now this is a group of sizes for this range
                             rotations: List[float] = [0, 1.5707963],
                             device: str = 'cuda') -> torch.Tensor:
        """Generate anchors in a single range for a given group of sizes."""
        if len(feature_size) == 2: # H, W
            feature_size = (1, *feature_size) # D, H, W (D=1)
        
        anchor_range_t = torch.tensor(anchor_range, device=device, dtype=torch.float32)
        
        # Create centers for z, y, x based on feature_size and anchor_range
        # The linspace endpoint for feature_size[dim] corresponds to number of anchor centers.
        z_centers = torch.linspace(anchor_range_t[2], anchor_range_t[5], int(feature_size[0]), device=device)
        y_centers = torch.linspace(anchor_range_t[1], anchor_range_t[4], int(feature_size[1]), device=device)
        x_centers = torch.linspace(anchor_range_t[0], anchor_range_t[3], int(feature_size[2]), device=device)

        sizes_t = torch.tensor(sizes, device=device, dtype=torch.float32).reshape(-1, 3) * scale
        rotations_t = torch.tensor(rotations, device=device, dtype=torch.float32)

        # Create meshgrid
        # Output order from meshgrid: x, y, z, rot (if indexing='ij' - default for torch)
        # Or y, x, z, rot (if indexing='xy') - typically used in image contexts for H,W
        # MMLab often uses x,y,z for LiDAR. Let's assume default 'ij' for meshgrid.
        centers_x, centers_y, centers_z, current_rotations = torch.meshgrid(
            x_centers, y_centers, z_centers, rotations_t, indexing='ij' # Ensure order is as expected
        )
        # Result shapes: [*feature_size_permuted_by_meshgrid, num_rotations]

        # Expand dimensions for broadcasting with sizes
        # Target shape for centers/rot: [Fx, Fy, Fz, 1 (for sizes), num_rot, 1 (for attribute)]
        centers_x = centers_x.unsqueeze(-2).unsqueeze(-1)
        centers_y = centers_y.unsqueeze(-2).unsqueeze(-1)
        centers_z = centers_z.unsqueeze(-2).unsqueeze(-1)
        current_rotations = current_rotations.unsqueeze(-2).unsqueeze(-1)

        # Prepare sizes for broadcasting: [1, 1, 1, num_sizes, 1 (for rots), 3 (for dim)]
        expanded_sizes = sizes_t.view(1, 1, 1, -1, 1, 3)
        # Repeat sizes to match the grid dimensions
        # Shape of centers_x: [Fx, Fy, Fz, 1, num_rot, 1]
        # Tile shape for sizes needs to match this up to the num_sizes dim
        tile_shape_for_sizes = list(centers_x.shape) # [Fx, Fy, Fz, 1, num_rot, 1]
        tile_shape_for_sizes[3] = 1 # This will be broadcasted by sizes_t own num_sizes dim
        expanded_sizes = expanded_sizes.repeat(*tile_shape_for_sizes[:3], 1, tile_shape_for_sizes[4], 1)

        # Repeat centers and rotations to match num_sizes dimension
        tile_shape_for_centers = [1] * 6 # For broadcasting centers_x, etc.
        tile_shape_for_centers[3] = sizes_t.size(0) # num_sizes

        centers_x = centers_x.repeat(tile_shape_for_centers)
        centers_y = centers_y.repeat(tile_shape_for_centers)
        centers_z = centers_z.repeat(tile_shape_for_centers)
        current_rotations = current_rotations.repeat(tile_shape_for_centers)

        # Concatenate: [x, y, z, w, l, h, rot] (MMLab order for sizes is often w,l,h or l,w,h)
        # Original code uses sizes.insert(3, sizes_t_expanded) which implies order in rets
        # Original `rets` was [x_centers, y_centers, z_centers, rotations]
        # Then `rets.insert(3, sizes)` -> [x,y,z, sizes, rot]
        # Let's stick to (x,y,z, dim_x, dim_y, dim_z, rot) = (x,y,z, w,l,h, rot) if sizes are w,l,h
        # sizes_t is [num_sizes, 3 (w,l,h)] -> expanded_sizes is [Fx,Fy,Fz,num_sizes,num_rot,3]
        anchors = torch.cat([centers_x, centers_y, centers_z, expanded_sizes, current_rotations], dim=-1)
        # Anchors shape: [Fx, Fy, Fz, num_sizes_for_this_group, num_rotations, 7]
        # Permute to match MMLab convention if necessary, often [Fz, Fy, Fx, num_anchors_per_loc, 7]
        # Original: ret = torch.cat(rets, dim=-1).permute([2, 1, 0, 3, 4, 5])
        # This implies rets was [x, y, z, sizes, rot] and each item had shape [Fx, Fy, Fz, N_size, N_rot, 1 or 3]
        # The permutation [2,1,0,...] changes Fx,Fy,Fz to Fz,Fy,Fx
        anchors = anchors.permute(2, 1, 0, 3, 4, 5) # [Fz, Fy, Fx, num_sizes, num_rot, 7]

        if self.custom_values: # custom_values is a tuple
            custom_t = torch.tensor(self.custom_values, device=device, dtype=torch.float32)
            custom_ndim = len(self.custom_values)
            # Create tensor matching anchor dims, broadcast custom_values
            expanded_custom_values = custom_t.view((1,) * (anchors.ndim - 1) + (custom_ndim,))
            expanded_custom_values = expanded_custom_values.repeat(*(anchors.shape[:-1] + (1,)))
            anchors = torch.cat([anchors, expanded_custom_values], dim=-1)
        
        return anchors


@TASK_UTILS.register_module()
class AlignedAnchor3DRangeGenerator(Anchor3DRangeGenerator):
    """Aligned 3D Anchor Generator by range.
    (Docstring from original script - consider expanding)
    """
    def __init__(self, align_corner: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.align_corner = align_corner

    def anchors_single_range(self,
                             feature_size: Tuple[int, ...],
                             anchor_range: List[float],
                             scale: float,
                             sizes: List[List[float]] = [[3.9, 1.6, 1.56]],
                             rotations: List[float] = [0, 1.5707963],
                             device: str = 'cuda') -> torch.Tensor:
        if len(feature_size) == 2:
            feature_size = (1, *feature_size)
        
        anchor_range_t = torch.tensor(anchor_range, device=device, dtype=torch.float32)
        sizes_t = torch.tensor(sizes, device=device, dtype=torch.float32).reshape(-1, 3) * scale
        rotations_t = torch.tensor(rotations, device=device, dtype=torch.float32)

        # Generate centers based on feature_size + 1 points to define voxel boundaries
        z_coords = torch.linspace(anchor_range_t[2], anchor_range_t[5], int(feature_size[0]) + 1, device=device)
        y_coords = torch.linspace(anchor_range_t[1], anchor_range_t[4], int(feature_size[1]) + 1, device=device)
        x_coords = torch.linspace(anchor_range_t[0], anchor_range_t[3], int(feature_size[2]) + 1, device=device)

        # Calculate voxel centers if not aligning to corners
        if not self.align_corner:
            z_centers = (z_coords[:-1] + z_coords[1:]) / 2
            y_centers = (y_coords[:-1] + y_coords[1:]) / 2
            x_centers = (x_coords[:-1] + x_coords[1:]) / 2
        else: # Align to the "bottom-left-front" corner of voxels
            z_centers = z_coords[:-1]
            y_centers = y_coords[:-1]
            x_centers = x_coords[:-1]
        
        # The rest of the logic is identical to the parent's anchors_single_range
        # Duplicating for clarity, or could call super().anchors_single_range with modified centers.
        # For this refactor, I'll reuse by creating these centers and then proceeding as parent:
        
        centers_x_mesh, centers_y_mesh, centers_z_mesh, current_rotations_mesh = torch.meshgrid(
            x_centers, y_centers, z_centers, rotations_t, indexing='ij'
        )
        centers_x_mesh = centers_x_mesh.unsqueeze(-2).unsqueeze(-1)
        centers_y_mesh = centers_y_mesh.unsqueeze(-2).unsqueeze(-1)
        centers_z_mesh = centers_z_mesh.unsqueeze(-2).unsqueeze(-1)
        current_rotations_mesh = current_rotations_mesh.unsqueeze(-2).unsqueeze(-1)

        expanded_sizes = sizes_t.view(1, 1, 1, -1, 1, 3)
        tile_shape_for_sizes = list(centers_x_mesh.shape)
        tile_shape_for_sizes[3] = 1
        expanded_sizes = expanded_sizes.repeat(*tile_shape_for_sizes[:3], 1, tile_shape_for_sizes[4], 1)

        tile_shape_for_centers = [1] * 6
        tile_shape_for_centers[3] = sizes_t.size(0)

        centers_x_mesh = centers_x_mesh.repeat(tile_shape_for_centers)
        centers_y_mesh = centers_y_mesh.repeat(tile_shape_for_centers)
        centers_z_mesh = centers_z_mesh.repeat(tile_shape_for_centers)
        current_rotations_mesh = current_rotations_mesh.repeat(tile_shape_for_centers)

        anchors = torch.cat([centers_x_mesh, centers_y_mesh, centers_z_mesh, expanded_sizes, current_rotations_mesh], dim=-1)
        anchors = anchors.permute(2, 1, 0, 3, 4, 5)

        if self.custom_values:
            custom_t = torch.tensor(self.custom_values, device=device, dtype=torch.float32)
            custom_ndim = len(self.custom_values)
            expanded_custom_values = custom_t.view((1,) * (anchors.ndim - 1) + (custom_ndim,))
            expanded_custom_values = expanded_custom_values.repeat(*(anchors.shape[:-1] + (1,)))
            anchors = torch.cat([anchors, expanded_custom_values], dim=-1)
        return anchors

@TASK_UTILS.register_module()
class AlignedAnchor3DRangeGeneratorPerCls(AlignedAnchor3DRangeGenerator):
    """Aligned 3D Anchor Generator by range for per class.
    (Docstring from original - check if scales assertion is still desired)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Original assertion: assert len(self.scales) == 1 ...
        # If multi-scale per class is needed, this might be revisited.
        # For now, keeping it to indicate it expects single scale factor for all per-class generations.
        if len(self.scales) != 1:
            logging.warning("AlignedAnchor3DRangeGeneratorPerCls typically expects a single scale value for all classes.")


    def grid_anchors(self, featmap_sizes_per_class: List[Tuple[int, ...]], device: str = 'cuda') -> List[List[torch.Tensor]]:
        """
        Generates grid anchors for multiple classes, where each class can have a different feature map size.
        Assumes a single feature level (scale).
        """
        # featmap_sizes_per_class is a list of feature map sizes, one for each class/range/size group.
        # The output structure from MMLab for such anchor generators is often List[List[Tensor]],
        # where the outer list is for feature levels (here, only 1 level)
        # and the inner list is for anchors per class (or per anchor group).
        if len(self.scales) != 1:
             raise ValueError("AlignedAnchor3DRangeGeneratorPerCls currently supports only a single scale for all classes.")
        scale = self.scales[0]

        anchors_per_class_group = self.multi_cls_grid_anchors(featmap_sizes_per_class, scale, device=device)
        return [anchors_per_class_group] # Wrap in an outer list for feature levels

    def multi_cls_grid_anchors(self, featmap_sizes_per_class: List[Tuple[int, ...]], scale: float, device: str = 'cuda') -> List[torch.Tensor]:
        """Generates grid anchors for multiple classes with potentially different feature map sizes."""
        if not (len(featmap_sizes_per_class) == len(self.sizes) == len(self.ranges)):
            raise ValueError("The number of feature_map_sizes, anchor_sizes, and anchor_ranges must be the same for per-class generation.")

        all_class_anchors_list = []
        for i in range(len(featmap_sizes_per_class)):
            # self.sizes[i] should be a list of sizes for the i-th class/group
            # self.ranges[i] is the range for the i-th class/group
            anchors_for_class = self.anchors_single_range(
                featmap_sizes_per_class[i],
                self.ranges[i],
                scale,
                self.sizes[i], # Pass the specific size group for this class
                self.rotations, # Rotations are typically shared
                device=device
            )
            # Reshape to [N, box_dim] as often expected by MMLab detectors
            all_class_anchors_list.append(anchors_for_class.reshape(-1, anchors_for_class.size(-1)))
        
        return all_class_anchors_list # List of Tensors, one per class/group


# --- Custom BBox Coder & IoU Calculators (from your script, assuming they are MMLab compatible) ---
# These seem to be direct copies/adaptations from MMLab or similar.
# Minor refactoring for clarity if needed, but structure is MMLab standard.

try:
    from mmdet.models.task_modules.coders import BaseBBoxCoder # For DeltaXYZWLHRBBoxCoder
    from mmdet.evaluation.functional import bbox_overlaps # For BboxOverlapsNearest3D (2D BEV part)
    from mmdet3d.structures import get_box_type # For IoU calculators, was mmdet3d.core.bbox.structures
    # For BboxOverlaps3D, the original calls bboxes1.overlaps(bboxes1, bboxes2, mode=mode)
    # This implies bboxes1 is an instance of a Box3DMode class from mmdet3D which has an overlaps method.
except ImportError as e:
    logging.error(f"Failed to import MMDetection/MMDetection3D components: {e}. Some classes might not work.")
    BaseBBoxCoder = object # Placeholder
    bbox_overlaps = None
    get_box_type = None


@TASK_UTILS.register_module()
class DeltaXYZWLHRBBoxCoder(BaseBBoxCoder):
    """Bbox Coder for 3D boxes. Encodes/decodes deltas for regression.
    (Docstring from original script)
    """
    def __init__(self, code_size: int = 7):
        super().__init__() # BaseBBoxCoder takes no args in MMDetection 2.x/3.x
        self.code_size = code_size

    @staticmethod
    def encode(src_boxes: torch.Tensor, dst_boxes: torch.Tensor) -> torch.Tensor:
        # (Implementation from original script - seems standard for this type of coder)
        box_ndim = src_boxes.shape[-1]
        # xc, yc, zc, w, l, h, r, *custom_values_src
        # For encoding, z is bottom center, but for calculation with height, often center of z is used.
        # Original: za = za + ha / 2
        
        # Decompose boxes
        if box_ndim > 7: # Has custom values
            xa, ya, za_bottom, wa, la, ha, ra, *cas_src = torch.split(src_boxes, 1, dim=-1)
            xg, yg, zg_bottom, wg, lg, hg, rg, *cas_dst = torch.split(dst_boxes, 1, dim=-1)
            custom_deltas = [dst_val - src_val for dst_val, src_val in zip(cas_dst, cas_src)]
        else:
            xa, ya, za_bottom, wa, la, ha, ra = torch.split(src_boxes, 1, dim=-1)
            xg, yg, zg_bottom, wg, lg, hg, rg = torch.split(dst_boxes, 1, dim=-1)
            custom_deltas = []

        za = za_bottom + ha * 0.5 # Center z for anchor
        zg = zg_bottom + hg * 0.5 # Center z for gt

        diagonal = torch.sqrt(la**2 + wa**2)
        denominators = torch.where(diagonal == 0, torch.ones_like(diagonal), diagonal) # Avoid div by zero

        xt = (xg - xa) / denominators
        yt = (yg - ya) / denominators
        zt = (zg - za) / torch.where(ha == 0, torch.ones_like(ha), ha) # Avoid div by zero for ha

        lt = torch.log(torch.where(la == 0, torch.ones_like(la), lg / torch.where(la == 0, torch.ones_like(la), la)))
        wt = torch.log(torch.where(wa == 0, torch.ones_like(wa), wg / torch.where(wa == 0, torch.ones_like(wa), wa)))
        ht = torch.log(torch.where(ha == 0, torch.ones_like(ha), hg / torch.where(ha == 0, torch.ones_like(ha), ha)))
        
        rt = rg - ra # Simpler representation for rotation delta
        
        return torch.cat([xt, yt, zt, wt, lt, ht, rt, *custom_deltas], dim=-1)

    @staticmethod
    def decode(anchors: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        # (Implementation from original script - seems standard)
        box_ndim = anchors.shape[-1]
        if box_ndim > 7: # Has custom values
            xa, ya, za_bottom, wa, la, ha, ra, *cas_anchor = torch.split(anchors, 1, dim=-1)
            xt, yt, zt, wt, lt, ht, rt, *cts_delta = torch.split(deltas, 1, dim=-1)
            custom_pred = [delta_val + anchor_val for delta_val, anchor_val in zip(cts_delta, cas_anchor)]
        else:
            xa, ya, za_bottom, wa, la, ha, ra = torch.split(anchors, 1, dim=-1)
            xt, yt, zt, wt, lt, ht, rt = torch.split(deltas[: , :7], 1, dim=-1) # Ensure only 7 for core deltas
            custom_pred = []
            if deltas.size(-1) > 7: # Handle cases where deltas might have more than 7 but anchors dont
                custom_pred = list(torch.split(deltas[:, 7:], 1, dim=-1))


        za = za_bottom + ha * 0.5 # Anchor z center

        diagonal = torch.sqrt(la**2 + wa**2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * ha + za # Predicted z center

        lg = torch.exp(lt) * la
        wg = torch.exp(wt) * wa
        hg = torch.exp(ht) * ha
        rg = rt + ra
        
        zg_bottom_pred = zg - hg * 0.5 # Predicted z bottom

        return torch.cat([xg, yg, zg_bottom_pred, wg, lg, hg, rg, *custom_pred], dim=-1)

# Placeholder for MMLab's actual bbox_overlaps_3d if direct import/use is intended
# This usually requires compiled CUDA ops from MMLab.
def _placeholder_bbox_overlaps_3d(bboxes1, bboxes2, mode='iou', coordinate='lidar'):
    logging.warning("Using placeholder for 3D bbox overlaps. Install MMLab for full functionality.")
    # Simple BEV IoU as a very rough placeholder if needed for basic runs
    if bbox_overlaps and get_box_type:
        box_type, _ = get_box_type(coordinate)
        b1 = box_type(bboxes1, box_dim=bboxes1.shape[-1]).nearest_bev
        b2 = box_type(bboxes2, box_dim=bboxes2.shape[-1]).nearest_bev
        return bbox_overlaps(b1, b2, mode=mode)
    return torch.zeros((bboxes1.size(0), bboxes2.size(0)), device=bboxes1.device)


@TASK_UTILS.register_module()
class BboxOverlapsNearest3D:
    """Nearest 3D IoU Calculator (BEV IoU)."""
    def __init__(self, coordinate: str = 'lidar'):
        assert coordinate in ['camera', 'lidar', 'depth']
        self.coordinate = coordinate

    def __call__(self, bboxes1: torch.Tensor, bboxes2: torch.Tensor, mode: str = 'iou', is_aligned: bool = False) -> torch.Tensor:
        if not (bbox_overlaps and get_box_type):
            raise ImportError("MMDetection components (bbox_overlaps, get_box_type) not available for BboxOverlapsNearest3D.")
        
        assert bboxes1.size(-1) == bboxes2.size(-1) and bboxes1.size(-1) >= 7
        box_type_cls, _ = get_box_type(self.coordinate)
        
        # Ensure tensor inputs
        if not isinstance(bboxes1, torch.Tensor): bboxes1 = torch.tensor(bboxes1)
        if not isinstance(bboxes2, torch.Tensor): bboxes2 = torch.tensor(bboxes2)

        # Instantiate box objects
        mm_bboxes1 = box_type_cls(bboxes1, box_dim=bboxes1.shape[-1], origin=(0.5,0.5,0.5) if self.coordinate=='lidar' else (0.5,0.5,0.0))
        mm_bboxes2 = box_type_cls(bboxes2, box_dim=bboxes2.shape[-1], origin=(0.5,0.5,0.5) if self.coordinate=='lidar' else (0.5,0.5,0.0))

        bboxes1_bev = mm_bboxes1.nearest_bev
        bboxes2_bev = mm_bboxes2.nearest_bev
        
        # mmdet.evaluation.functional.bbox_overlaps expects xyxy format for BEV boxes
        # The .bev or .nearest_bev from MMLab box structures should already provide this.
        return bbox_overlaps(bboxes1_bev, bboxes2_bev, mode=mode, is_aligned=is_aligned)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(coordinate={self.coordinate})'

@TASK_UTILS.register_module()
class BboxOverlaps3D:
    """3D IoU Calculator using volume-based IoU."""
    def __init__(self, coordinate: str):
        assert coordinate in ['camera', 'lidar', 'depth']
        self.coordinate = coordinate

    def __call__(self, bboxes1: torch.Tensor, bboxes2: torch.Tensor, mode: str = 'iou') -> torch.Tensor:
        if not get_box_type: # Check if critical MMLab util is available
            raise ImportError("MMDetection3D 'get_box_type' not available. Cannot perform 3D IoU.")

        # Ensure tensor inputs
        if not isinstance(bboxes1, torch.Tensor): bboxes1 = torch.tensor(bboxes1)
        if not isinstance(bboxes2, torch.Tensor): bboxes2 = torch.tensor(bboxes2)

        assert bboxes1.size(-1) == bboxes2.size(-1) and bboxes1.size(-1) >= 7

        box_type_cls, _ = get_box_type(self.coordinate)
        
        # Instantiate MMLab box objects
        # Origin might need adjustment based on how your boxes are defined (center vs. bottom-center)
        # For LiDAR, (0.5, 0.5, 0.5) for center of box, (0.5,0.5,0) for bottom center.
        # The DeltaXYZWLHRBBoxCoder uses bottom center for z and adds h/2.
        # Let's assume the box definition for IoU expects center for z if using LiDAR boxes directly.
        origin = (0.5, 0.5, 0.0) if self.coordinate == 'camera' else (0.5, 0.5, 0.5) # Adjust if z is bottom for lidar boxes
        
        mm_bboxes1 = box_type_cls(bboxes1, box_dim=bboxes1.shape[-1], origin=origin)
        mm_bboxes2 = box_type_cls(bboxes2, box_dim=bboxes2.shape[-1], origin=origin)

        # The .overlaps method should exist on MMLab box instances if using specific types like LiDARInstance3DBoxes
        if hasattr(mm_bboxes1, 'overlaps'):
            return mm_bboxes1.overlaps(mm_bboxes1, mm_bboxes2, mode=mode) # MMLab's internal method
        else:
            # Fallback to placeholder if .overlaps method is not found (e.g. base Box3DMode)
            # This indicates a potential mismatch in box type or MMLab version.
            logging.warning("Using placeholder 3D IoU as '.overlaps' method not found on box type. Results may be inaccurate.")
            return _placeholder_bbox_overlaps_3d(bboxes1, bboxes2, mode, self.coordinate)


    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(coordinate={self.coordinate})'


# --- Main Execution Logic ---

def parse_args():
    parser = argparse.ArgumentParser(description='Test a 3D object detection model using MMEngine.')
    parser.add_argument('config', help='Path to the model config file (e.g., configs/your_algo.py)')
    parser.add_argument('checkpoint', help='Path to the checkpoint file to load weights from.')
    parser.add_argument('--work-dir', help='Directory to save logs and output files.')
    parser.add_argument('--show', action='store_true', help='Show results visuall_tool_codey (if backend allows).')
    parser.add_argument('--show-dir', help='Directory where painted images will be saved.')
    parser.add_argument('--wait-time', type=float, default=0, help='The interval of show (s).')
    parser.add_argument('--score-thr', type=float, default=0.1, help='Score threshold for visualization.')
    parser.add_argument(
        '--task', default='mono_det', # Default from original context, make sure it's appropriate
        choices=['mono_det', 'multi-view_det', 'lidar_det', 'lidar_seg', 'multi-modality_det'],
        help='Task type for visualization hook.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='Job launcher for distributed testing.')
    # Add any other MMEngine specific args or cfg-overrides if needed
    # e.g., --cfg-options model.param=value data.samples_per_gpu=2
    # args.cfg_options = None # MMEngine handles this
    return parser.parse_args()

def trigger_visualization_hook(cfg: Config, args: argparse.Namespace) -> Config:
    """Modifies config to enable and configure the visualization hook."""
    if 'visualization' in cfg.default_hooks:
        vis_hook_cfg = cfg.default_hooks['visualization']
        vis_hook_cfg['draw'] = True # Enable drawing predictions
        if args.show:
            vis_hook_cfg['show'] = True
            vis_hook_cfg['wait_time'] = args.wait_time
        if args.show_dir:
            vis_hook_cfg['test_out_dir'] = args.show_dir
        
        vis_hook_cfg['vis_task'] = args.task
        vis_hook_cfg['score_thr'] = args.score_thr
    elif args.show or args.show_dir: # If user wants viz but hook isn't defined
        logging.warning(
            "'visualization' hook not found in default_hooks but visualization options were requested. "
            "Please ensure 'visualization=dict(type=\'VisualizationHook\')' is in default_hooks in your config."
        )
    return cfg

def main():
    args = parse_args()

    # Basic logging setup
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    # Load config
    cfg = Config.fromfile(args.config)
    logging.info(f"Loaded config from: {args.config}")

    # Apply command-line arguments to config
    cfg.launcher = args.launcher
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None: # Set default if not in config or args
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])
    
    if args.checkpoint is not None:
        cfg.load_from = args.checkpoint

    # Handle data backend (e.g., Ceph) if utility is available
    if replace_ceph_backend is not None:
        cfg = replace_ceph_backend(cfg)

    # Configure visualization hook
    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    # Build the runner
    if 'runner_type' not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)
    
    logging.info("Runner built. Starting test process...")
    runner.test()
    logging.info("Test process completed.")

if __name__ == '__main__':
    main()