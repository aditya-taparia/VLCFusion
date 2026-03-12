import cv2
import numpy as np
import open3d as o3d
import os
from utils import bbox3d2corners


COLORS = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]]
COLORS_IMG = [[0, 0, 255], [0, 255, 0], [255, 0, 0], [0, 255, 255]] # BGR format -> Red: Pedestrian, Green: Cyclist, Blue: Car, Yellow: Ground Truth

LINES = [
        [0, 1],
        [1, 2], 
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [2, 6],
        [7, 3],
        [1, 5],
        [4, 0]
    ]


def npy2ply(npy):
    ply = o3d.geometry.PointCloud()
    ply.points = o3d.utility.Vector3dVector(npy[:, :3])
    density = npy[:, 3]
    colors = [[item, item, item] for item in density]
    ply.colors = o3d.utility.Vector3dVector(colors)
    return ply


def ply2npy(ply):
    return np.array(ply.points)


def bbox_obj(points, color=[1, 0, 0]):
    colors = [color for i in range(len(LINES))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(LINES),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def vis_core(plys):
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    PAR = os.path.dirname(os.path.abspath(__file__))
    ctr = vis.get_view_control()
    param = o3d.io.read_pinhole_camera_parameters(os.path.join(PAR, 'viewpoint.json'))
    for ply in plys:
        vis.add_geometry(ply)
    ctr.convert_from_pinhole_camera_parameters(param)

    vis.run()
    # param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    # o3d.io.write_pinhole_camera_parameters(os.path.join(PAR, 'viewpoint.json'), param)
    vis.destroy_window()


def vis_pc(pc, bboxes=None, labels=None):
    '''
    pc: ply or np.ndarray (N, 4)
    bboxes: np.ndarray, (n, 7) or (n, 8, 3)
    labels: (n, )
    '''
    if isinstance(pc, np.ndarray):
        pc = npy2ply(pc)
    
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=10, origin=[0, 0, 0])

    if bboxes is None:
        vis_core([pc, mesh_frame])
        return
    
    if len(bboxes.shape) == 2:
        bboxes = bbox3d2corners(bboxes)
    
    vis_objs = [pc, mesh_frame]
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        if labels is None:
            color = [1, 1, 0]
        else:
            if labels[i] >= 0 and labels[i] < 3:
                color = COLORS[labels[i]]
            else:
                color = COLORS[-1]
        vis_objs.append(bbox_obj(bbox, color=color))
    vis_core(vis_objs)


def vis_img_3d(img, image_points, labels, label_name=None, rt=True):
    '''
    img: (h, w, 3)
    image_points: (n, 8, 2)
    labels: (n, )
    label_name: dict
    '''

    if label_name is None:
        print('label_name is None')

    for i in range(len(image_points)):
        label = labels[i]
        bbox_points = image_points[i] # (8, 2)
        if label >= 0 and label < 3:
            color = COLORS_IMG[label]
        else:
            color = COLORS_IMG[-1]
        for line_id in LINES:
            x1, y1 = bbox_points[line_id[0]]
            x2, y2 = bbox_points[line_id[1]]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.line(img, (x1, y1), (x2, y2), color, 1)
        
        if label_name and label in label_name:
            label_txt = label_name[label]

            text_position = (int(bbox_points[0][0]), int(bbox_points[0][1]) - 10)
            cv2.putText(img, label_txt, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    if rt:
        return img
    cv2.imshow('bbox', img)
    cv2.waitKey(0)


################################## Aditya Added this ##################################

# Save the lidar visualization as image without rendering it
def vis_pc_without_render(pc, bboxes=None, labels=None):
    
    if isinstance(pc, np.ndarray):
        pc = npy2ply(pc)
    
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=10, origin=[0, 0, 0])
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    render_option = vis.get_render_option()
    render_option.background_color = np.array((1,1,1), dtype=np.float32)
    
    PAR = os.path.dirname(os.path.abspath(__file__))
    ctr = vis.get_view_control()
    param = o3d.io.read_pinhole_camera_parameters(os.path.join(PAR, 'viewpoint.json'))
    ctr.convert_from_pinhole_camera_parameters(param)
    
    vis_objs = [pc, mesh_frame]
    
    if len(bboxes.shape) == 2:
        bboxes = bbox3d2corners(bboxes)
    
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        if labels is None:
            color = [1, 1, 0]
        else:
            if labels[i] >= 0 and labels[i] < 3:
                color = COLORS[labels[i]]
            else:
                color = COLORS[-1]
        vis_objs.append(bbox_obj(bbox, color=color))
    
    for ply in vis_objs:
        vis.add_geometry(ply)
    
    vis.run()
    img_o3d = vis.capture_screen_float_buffer(do_render=True)
    
    vis.destroy_window()
    
    return img_o3d
    
    # # Output
    # if bboxes is None:
    #     vis_core([pc, mesh_frame])
    #     return
    
    # if len(bboxes.shape) == 2:
    #     bboxes = bbox3d2corners(bboxes)
    
    # vis_objs = [pc, mesh_frame]
    # for i in range(len(bboxes)):
    #     bbox = bboxes[i]
    #     if labels is None:
    #         color = [1, 1, 0]
    #     else:
    #         if labels[i] >= 0 and labels[i] < 3:
    #             color = COLORS[labels[i]]
    #         else:
    #             color = COLORS[-1]
    #     vis_objs.append(bbox_obj(bbox, color=color))
    # vis_core(vis_objs)
    
    # # Vis core
    # vis = o3d.visualization.Visualizer()
    # vis.create_window(visible=False)

    # PAR = os.path.dirname(os.path.abspath(__file__))
    # ctr = vis.get_view_control()
    # param = o3d.io.read_pinhole_camera_parameters(os.path.join(PAR, 'viewpoint.json'))
    # for ply in plys:
    #     vis.add_geometry(ply)
    # ctr.convert_from_pinhole_camera_parameters(param)

    # vis.run()
    # # param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    # # o3d.io.write_pinhole_camera_parameters(os.path.join(PAR, 'viewpoint.json'), param)
    # vis.destroy_window()