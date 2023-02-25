import time
from pathlib import Path
import numpy as np
import math
import open3d as o3d
import json
import csv
import pandas as pd

# import yaml
# import tqdm
# import cv2
# import torch
# import json
# import pickle

import os


def box_center_to_corner(box, i):

   # to do: verify if this is the format
   translation = [box['location'][i][0], box['location'][i][1], box['location'][i][2]]
   l,h,w = box['dimensions'][i]
   rotation = box['rotation_y'][i]


   # create a bounding box outline
   bounding_box = np.array([
       [-l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2, l/2],
       [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2],
       [-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2]
   ])


   # standard 3x3 rotation matrix around the z axis
   rotation_matrix = np.array([
       [np.cos(rotation), -np.sin(rotation), 0.0],
       [np.sin(rotation), np.cos(rotation), 0.0],
       [0.0, 0.0, 1.0]
   ])


   # repeat the [x, y, z] eight times
   eight_points = np.tile(translation, (8,1))


   # translate the rotated bounding box by the
   # original center position to obtain the final box
   corner_box = np.dot(
       rotation_matrix, bounding_box
   ) + eight_points.transpose()


   return corner_box.transpose()


def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
   """
   Aligns vector a to vector b with axis angle rotation
   """
   if np.array_equal(a, b):
       return None, None
   axis_ = np.cross(a, b)
   axis_ = axis_ / np.linalg.norm(axis_)
   angle = np.arccos(np.dot(a, b))
   return axis_, angle


def normalized(a, axis=-1, order=2):
   """normalizes a numpy array of points"""
   l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
   l2[l2 == 0] = 1
   return a / np.expand_dims(l2, axis), l2


class LineMesh(object):
   def __init__(self, points, lines=None, colors=[0,1,0], radius=0.15):
       """creates a line represented as sequence of cylinder triangular meshes
       arguments:
           points {ndarray} -- Numpy array of points Nx3.
       keyword arguments:
           lines {list[list] or None} -- List of point index pairs denoting line segments.
           colors {list} -- list of colours, or single colour of the line
           radius {float} -- radius of cylinder (default: {0.15})
       """
       self.points = np.array(points)
       self.lines = np.array(
           lines) if lines is not None else self.lines_from_ordered_points(self.points)
       self.colors = np.array(colors)
       self.radius = radius
       self.cylinder_segments = []

       self.create_line_mesh()


   def lines_from_ordered_points(points):
       lines = [[i, i+1] for i in range(0, points.shape[0] -1, 1)]
       return np.array(lines)


   def create_line_mesh(self):
       first_points = self.points[self.lines[:, 0], :]
       second_points = self.points[self.lines[:, 1], :]
       line_segments = second_points - first_points
       line_segments_unit, line_lengths = normalized(line_segments)


       z_axis = np.array([0, 0, 1])
       # create triangular mesh cylinder segments of line
       for i in range(line_segments_unit.shape[0]):
           line_segment = line_segments_unit[i, :]
           line_length = line_lengths[i]
           # get axis angle rotation to align cylinder with line segment
           axis, angle = align_vector_to_another(z_axis, line_segment)
           # get translation vector
           translation = first_points[i, :] + line_segment * line_length * 0.5
           # create cylinder and apply transformation
           cylinder_segment = o3d.geometry.TriangleMesh.create_cylinder(
               self.radius, line_length)
           cylinder_segment = cylinder_segment.translate(
               translation, relative=False)
           if axis is not None:
               axis_a = axis * angle
               cylinder_segment = cylinder_segment.rotate(
                   R=o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a), center=cylinder_segment.get_center())
           # color cylinder
           color = self.colors if self.colors.ndim == 1 else self.colors[i, :]
           cylinder_segment.paint_uniform_color(color)


           self.cylinder_segments.append(cylinder_segment)


   def add_line(self, vis):
       """add this line to the visualizer"""
       for cylinder in self.cylinder_segments:
           vis.add_geometry(cylinder)


   def remove_line(self, vis):
       """remove this line from the visualizer"""
       for cylinder in self.cylinder_segments:
           vis.remove_geometry(cylinder)


def return_pcd(scan_name, calib, obj_name, line_thickness=0.15,
              threshold_score=0.4, threed=True):
   with open(calib, 'r') as f:
        calib_infos = f.readlines()
   calib_content = [calib_info.strip().split(' ') for calib_info in calib_infos]
   Tr_velo_to_cam = np.array(calib_content[-2][1:], dtype=np.float32).reshape((3,4))
   R0_rect = np.array(calib_content[4][1:], dtype=np.float32).reshape((3,3))
   pcd_np = np.fromfile(scan_name, dtype=np.float32).reshape((-1, 6))

   local_xyz = pcd_np[:,0:3]


   if threed == False:
       local_xyz[:,2] = -0.1

   raw_pcd = o3d.geometry.PointCloud()
   R = np.array([220.0])
   G = np.array([220.0])
   B = np.array([220.0])
   rgb = np.asarray([R, G, B])
   rgb_t = np.transpose(rgb)
   rgb_t = np.repeat(rgb_t, local_xyz.shape[0], axis=0)
   raw_pcd.points = o3d.utility.Vector3dVector(local_xyz)
   raw_pcd.colors = o3d.utility.Vector3dVector(rgb_t.astype('float') / 255.0)


   # colouring object detction ground truth and predictions
   gt_boxes = obj_name # list of numpy arrays (x, y, z, w, l, h, ry, velocity_x, velocity_y, class)
   # gt_boxes_labels = (obj_name['gt_boxes']).cpu().numpy()[:,-1]
   #
   #
   # pred_boxes = obj_name['boxes_lidar'][obj_name['score'] > threshold_score] # list of arrays (x, y, z, w, l, h, ry, velocity_x, velocity_y)
   # pred_boxes_labels = obj_name['pred_labels'][obj_name['score'] > threshold_score] # list of array


   # ground truth object detection labels
   lines = []
   corners = np.array([])
   if len(gt_boxes['name']) != 0:
   # if len(gt_boxes) != 0 and len(pred_boxes) != 0:
       for i in range(len(gt_boxes['name'])):


           if len(corners) == 0:
               # only draw from first four corners
               if threed == False:
                   corners = box_center_to_corner(gt_boxes, i)[4:8, :]
               else:
                   corners = box_center_to_corner(gt_boxes, i)


           else:
               # only draw from first four corners
               if threed == False:
                   corners = np.vstack((corners, box_center_to_corner(gt_boxes, i)[4:8, :]))
               else:
                   corners = np.vstack((corners, box_center_to_corner(gt_boxes, i)))

           corners = np.matmul(np.linalg.inv(R0_rect), corners.T).T
           corners = np.matmul(np.linalg.inv(Tr_velo_to_cam[:, :3]), corners.T).T
           # corners[:, 2] += 1.6
           if threed == False:
               corners[:,2] = 0

           # only draw from first four corners
           if threed == False:
               lines += [[0+4*i, 1+4*i], [1+4*i, 2+4*i], [2+4*i, 3+4*i], [0+4*i, 3+4*i]]
           else:
               lines += [[0+8*i, 1+8*i], [1+8*i, 2+8*i], [2+8*i, 3+8*i], [0+8*i, 3+8*i],
                         [4+8*i, 5+8*i], [5+8*i, 6+8*i], [6+8*i, 7+8*i], [4+8*i, 7+8*i],
                         [0+8*i, 4+8*i], [1+8*i, 5+8*i], [2+8*i, 6+8*i], [3+8*i, 7+8*i]]


       # use the same color for all lines
       colors_red = [[255.0/255.0, 51.0/255.0, 51.0/255.0] for _ in range(len(lines))]


       # create line mesh (thickend lines)
       pcd_label_obj = LineMesh(corners, lines, colors_red, line_thickness)
       pcd_label_obj = pcd_label_obj.cylinder_segments


       # lines = []
       # corners = np.array([])
       # for i in range(len(pred_boxes)):
       #
       #
       #     if len(corners) == 0:
       #         # only draw from first four corners
       #         if threed == False:
       #             corners = box_center_to_corner(pred_boxes, i)[4:8, :]
       #         else:
       #             corners = box_center_to_corner(pred_boxes, i)
       #
       #
       #     else:
       #         # only draw from first four corners
       #         if threed == False:
       #             corners = np.vstack((corners, box_center_to_corner(pred_boxes, i)[4:8, :]))
       #         else:
       #             corners = np.vstack((corners, box_center_to_corner(pred_boxes, i)))
       #
       #
       #     if threed == False:
       #         corners[:,2] = 0
       #
       #
       #     # only draw from first four corners
       #     if threed == False:
       #         lines += [[0 + 4 * i, 1 + 4 * i], [1 + 4 * i, 2 + 4 * i], [2 + 4 * i, 3 + 4 * i], [0 + 4 * i, 3 + 4 * i]]
       #     else:
       #         lines += [[0 + 8 * i, 1 + 8 * i], [1 + 8 * i, 2 + 8 * i], [2 + 8 * i, 3 + 8 * i], [0 + 8 * i, 3 + 8 * i],
       #                   [4 + 8 * i, 5 + 8 * i], [5 + 8 * i, 6 + 8 * i], [6 + 8 * i, 7 + 8 * i], [4 + 8 * i, 7 + 8 * i],
       #                   [0 + 8 * i, 4 + 8 * i], [1 + 8 * i, 5 + 8 * i], [2 + 8 * i, 6 + 8 * i], [3 + 8 * i, 7 + 8 * i]]
       #                   [0+4*i, 4+4*i], [1+4*i, 5+4*i], [2+4*i, 6+4*i], [3+4*i, 7+4*i]]
       #
       #
       #     # use the same color for all lines
       #     colors_blue = [[0.0/255.0, 102.0/255.0, 204.0/255.0] for _ in range(len(lines))]


       # # create line mesh (thickend lines)
       # pcd_pred_obj = LineMesh(corners, lines, colors_blue, radius = line_thickness)
       # pcd_pred_obj = pcd_pred_obj.cylinder_segments
       pcd_pred_obj = []

   else:
       pcd_pred_obj = []
       pcd_label_obj = []


   return raw_pcd, pcd_label_obj, pcd_pred_obj


def gen_figures(batch_dict, calib_dict, annos, vis_dir, idx, threshold_score=0.40, threed=False):
   # # to do: classwise bounding box, generate specific samples

   # obtain open3d objects (point clouds and bounding boxes)
   raw_pcd, pcd_label_obj, pcd_pred_obj = return_pcd(batch_dict, calib_dict, annos, line_thickness=0.15, threshold_score=threshold_score, threed=threed)


   # rotate open3d objects
   if threed:
       # adjust the rotation on Z axis
       _theta = math.pi
       R_z = np.array([[math.cos(_theta), -math.sin(_theta), 0],
                       [math.sin(_theta), math.cos(_theta), 0],
                       [0, 0, 1]])


       # adjust the camera angle (how much it is with respective to the ground)
       _theta = - math.pi / 3
       R_x = np.array([[1, 0, 0],
                       [0, math.cos(_theta), -math.sin(_theta)],
                       [0, math.sin(_theta), math.cos(_theta)]])


       raw_pcd.rotate(R_z, [0, 0, 0])
       if len(pcd_label_obj) != 0:
           for i in range(len(pcd_label_obj)):
               pcd_label_obj[i].rotate(R_z, [0, 0, 0])
       if len(pcd_pred_obj) != 0:
           for i in range(len(pcd_pred_obj)):
               pcd_pred_obj[i].rotate(R_z, [0, 0, 0])


       raw_pcd.rotate(R_x, [0, 0, 0])
       if len(pcd_label_obj) != 0:
           for i in range(len(pcd_label_obj)):
               pcd_label_obj[i].rotate(R_x, [0, 0, 0])
       if len(pcd_pred_obj) != 0:
           for i in range(len(pcd_pred_obj)):
               pcd_pred_obj[i].rotate(R_x, [0, 0, 0])
   if not os.path.exists(vis_dir):
       os.makedirs(vis_dir)
       os.makedirs(os.path.join(vis_dir, 'detection_gt'))
       os.makedirs(os.path.join(vis_dir, 'detection_pred'))
   if len(pcd_label_obj) != 0:
       o3d.visualization.draw_geometries(pcd_label_obj + [raw_pcd])
       save_to_png(pcd=raw_pcd, filename=os.path.join(vis_dir, 'detection_gt', str(idx) + '.png'), ld=pcd_label_obj)
   if len(pcd_pred_obj) != 0:
       save_to_png(pcd=raw_pcd, filename=os.path.join(vis_dir, 'detection_pred', str(idx) + '.png'), ld=pcd_label_obj,
                   lpd=pcd_pred_obj)

def save_to_png(pcd, filename, ld= None, lpd = None, zoom_level = 1.5):
   o3d.visualization.draw_geometries(ld + [pcd])
   vis = o3d.visualization.Visualizer()
   vis.create_window(window_name='TopMiddleRight', width=2640, height=2340, left=50, top=50)


   vis.add_geometry(pcd)

   if ld != None:
       for i in range(len(ld)):
           vis.add_geometry(ld[i])


   if lpd != None:
       for i in range(len(lpd)):
           vis.add_geometry(lpd[i])


   o3d.visualization.ViewControl.set_zoom(vis.get_view_control(), zoom_level)
   o3d.visualization.ViewControl.set_lookat(vis.get_view_control(), [0,0,0])
   o3d.visualization.ViewControl.set_front(vis.get_view_control(), [0,0,1])
   o3d.visualization.ViewControl.rotate(vis.get_view_control(), 0,0)


   vis.update_geometry(pcd)
   # if ld != None:
   #     for i in range(len(ld)):
   #         vis.update_geometry(ld[i])
   #
   #
   # if lpd != None:
   #     for i in range(len(lpd)):
   #         vis.update_geometry(lpd[i])


   vis.poll_events()
   vis.update_renderer()
   vis.capture_screen_image(filename, do_render=True)
   vis.destroy_window()


def get_label_anno(label_path):
    annotations = {}
    annotations.update({
        'name': [],
        'truncated': [],
        'occluded': [],
        'alpha': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_y': []
    })
    with open(label_path, 'r') as f:
        lines = f.readlines()
    # if len(lines) == 0 or len(lines[0]) < 15:
    #     content = []
    # else:
    content = [line.strip().split(' ') for line in lines]
    filtered_content = list()
    for item in content:
        if item[0] in ['Car', 'Pedestrian', 'Bike']:
            filtered_content.append(item)

    num_objects = len([x[0] for x in filtered_content if x[0] != 'DontCare'])
    annotations['name'] = np.array([x[0] for x in filtered_content])
    num_gt = len(annotations['name'])
    annotations['truncated'] = np.array([float(x[1]) for x in filtered_content])
    annotations['occluded'] = np.array([int(x[2]) for x in filtered_content])
    annotations['alpha'] = np.array([float(x[3]) for x in filtered_content])
    annotations['bbox'] = np.array([[float(info) for info in x[4:8]]
                                    for x in filtered_content]).reshape(-1, 4)
    # dimensions will convert hwl format to standard lhw(camera) format.
    annotations['dimensions'] = np.array([[float(info) for info in x[8:11]]
                                          for x in filtered_content
                                          ]).reshape(-1, 3)[:, [2, 0, 1]]
    annotations['location'] = np.array([[float(info) for info in x[11:14]]
                                        for x in filtered_content]).reshape(-1, 3)
    annotations['rotation_y'] = np.array([float(x[14])
                                          for x in filtered_content]).reshape(-1)
    if len(filtered_content) != 0 and len(filtered_content[0]) == 16:  # have score
        annotations['score'] = np.array([float(x[15]) for x in filtered_content])
    else:
        annotations['score'] = np.zeros((annotations['bbox'].shape[0], ))
    index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
    annotations['index'] = np.array(index, dtype=np.int32)
    annotations['group_ids'] = np.arange(num_gt, dtype=np.int32)
    return annotations
def main():
    root_dir = '/home/user/Documents/carla_data/DataGenerator/data/sem_test/train/velodyne'
    file_dir = os.listdir(root_dir)
    # file_dir.sort()
    anno_dir = '/home/user/Documents/carla_data/DataGenerator/data/sem_test/train/kitti_label'
    calib_dir = '/home/user/Documents/carla_data/DataGenerator/data/sem_test/train/calib'
    vis_dir = 'vis_dir'
    idx = 0
    for file in file_dir:
        batch_dict = os.path.join(root_dir, file)
        anno_dict = os.path.join(anno_dir, file.split('.')[0]+'.txt')
        calib_dict = os.path.join(calib_dir, '000633.txt')
        annos = get_label_anno(anno_dict)
        gen_figures(batch_dict, calib_dict, annos, vis_dir, idx, threed=True)
        idx += 1




if __name__ == '__main__':
    main()