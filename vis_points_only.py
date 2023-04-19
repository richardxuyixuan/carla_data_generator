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

LABEL_COLORS = np.array([
    (255, 255, 255), # None
    (70, 70, 70),    # Building
    (100, 40, 40),   # Fences
    (55, 90, 80),    # Other
    (220, 20, 60),   # Pedestrian
    (153, 153, 153), # Pole
    (157, 234, 50),  # RoadLines
    (128, 64, 128),  # Road
    (244, 35, 232),  # Sidewalk
    (107, 142, 35),  # Vegetation
    (0, 0, 142),     # Vehicle
    (102, 102, 156), # Wall
    (220, 220, 0),   # TrafficSign
    (70, 130, 180),  # Sky
    (81, 0, 81),     # Ground
    (150, 100, 100), # Bridge
    (230, 150, 140), # RailTrack
    (180, 165, 180), # GuardRail
    (250, 170, 30),  # TrafficLight
    (110, 190, 160), # Static
    (170, 120, 50),  # Dynamic
    (45, 60, 150),   # Water
    (145, 170, 100), # Terrain
]) / 255.0 # normalize each channel [0-1] since is what Open3D uses

def box_center_to_corner(box):

   # to do: verify if this is the format
   translation = [box['position']['x'], box['position']['y'], box['position']['z']]
   w, l, h = box['dimensions']['x'], box['dimensions']['y'], box['dimensions']['z']
   rotation = box['yaw']


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


def return_pcd(scan_name, line_thickness=0.15,
              threshold_score=0.4, threed=True):
   # fetching point cloud
   # with open(scan_name, 'r') as file:
   #     reader = csv.reader(file, delimiter=',')
   #     for row in reader:
   #         print(row)
   #         break

   pcd_np = np.fromfile(scan_name, dtype=np.float32).reshape((-1, 6))
   # pcd_extra = np.fromfile('/home/user/Documents/carla_data/DataGenerator/data/sem_test/train/kitti_velodyne/000092.bin', dtype=np.float32).reshape((-1, 4))
   labels = pcd_np[:, -2]
   int_color = LABEL_COLORS[labels.astype(int)]
   local_xyz = pcd_np[:,0:3]
   # # adjust the camera angle (how much it is with respective to the ground)
   # _theta = - math.pi / 2
   # # R_x = np.array([[math.cos(_theta), -math.sin(_theta), 0],
   # #                     [math.sin(_theta), math.cos(_theta), 0],
   # #                     [0, 0, 1]])
   #
   # R_x = np.array([[math.cos(_theta), 0, math.sin(_theta)],
   #                 [0, 1, 0],
   #                 [-math.sin(_theta), 0, math.cos(_theta)]])
   #
   #
   # # R_x = np.array([[1, 0, 0],
   # #                 [0, math.cos(_theta), -math.sin(_theta)],
   # #                 [0, math.sin(_theta), math.cos(_theta)]])
   # local_xyz = np.matmul(local_xyz, R_x)
   # local_xyz[:, 2] += 70.0 - 1.6
   # pcd_extra_xyz = pcd_extra[:, 0:3]
   # local_xyz = np.concatenate((local_xyz, pcd_extra_xyz), axis=0).copy()
   # # not_roof = original_depth > 2.5
   # # local_xyz = local_xyz[not_roof]


   if threed == False:
       local_xyz[:,2] = -0.1

   raw_pcd = o3d.geometry.PointCloud()
   # R = np.array([220.0])
   # G = np.array([220.0])
   # B = np.array([220.0])
   # rgb = np.asarray([R, G, B])
   # rgb_t = np.transpose(rgb)
   # rgb_t = np.repeat(rgb_t, local_xyz.shape[0], axis=0)
   raw_pcd.points = o3d.utility.Vector3dVector(local_xyz)
   # raw_pcd.colors = o3d.utility.Vector3dVector(rgb_t.astype('float') / 255.0)
   raw_pcd.colors = o3d.utility.Vector3dVector(int_color)

   return raw_pcd


def gen_figures(batch_dict, vis_dir, idx, threshold_score=0.40, threed=False):
   # # to do: classwise bounding box, generate specific samples

   # obtain open3d objects (point clouds and bounding boxes)
   raw_pcd = return_pcd(batch_dict, line_thickness=0.15, threshold_score=threshold_score, threed=threed)


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


       raw_pcd.rotate(R_x, [0, 0, 0])
   if not os.path.exists(vis_dir):
       os.makedirs(vis_dir)
   if not os.path.exists(os.path.join(vis_dir, 'raw_pcd')):
       os.makedirs(os.path.join(vis_dir, 'raw_pcd'))
   o3d.visualization.draw_geometries([raw_pcd])
   save_to_png(pcd=raw_pcd, filename=os.path.join(vis_dir, 'raw_pcd', str(idx) + '.png'))

def save_to_png(pcd, filename, ld= None, lpd = None, zoom_level = 1.5):
   vis = o3d.visualization.Visualizer()
   # vis.create_window(window_name='TopMiddleRight', width=2640, height=2340, left=50, top=50)
   vis.create_window(
       window_name='Carla Lidar',
       width=960,
       height=540,
       left=480,
       top=270)
   vis.get_render_option().background_color = [0.05, 0.05, 0.05]
   vis.get_render_option().point_size = 1
   vis.get_render_option().show_coordinate_frame = True

   vis.add_geometry(pcd)

   if ld != None:
       for i in range(len(ld)):
           vis.add_geometry(ld[i])


   if lpd != None:
       for i in range(len(lpd)):
           vis.add_geometry(lpd[i])


   # o3d.visualization.ViewControl.set_zoom(vis.get_view_control(), zoom_level)
   # o3d.visualization.ViewControl.set_lookat(vis.get_view_control(), [0,0,0])
   # o3d.visualization.ViewControl.set_front(vis.get_view_control(), [0,0,1])
   # o3d.visualization.ViewControl.rotate(vis.get_view_control(), 0,0)

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


def main():
    root_dir = '/media/user/storage/data/sem_test/train/velodyne'
    file_dir = os.listdir(root_dir)
    idx = 0
    for file in file_dir:
        batch_dict = os.path.join(root_dir, file)
        gen_figures(batch_dict, 'vis_dir', idx, threed=True)
        idx += 1

if __name__ == '__main__':
    main()