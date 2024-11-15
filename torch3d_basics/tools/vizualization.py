#!/usr/bin/env python
# -*- coding: utf-8 -*-

import open3d

def visualize_point_cloud(point_cloud):
    print(f"Visualizing the point cloud {point_cloud} using open3D")
    pcd = open3d.io.read_point_cloud(point_cloud)
    open3d.visualization.draw_geometries([pcd], mesh_show_wireframe=True)

def visualize_mesh(mesh):
    print(f"Visualizing the mesh {mesh} using open3D")
    mesh = open3d.io.read_triangle_mesh(mesh, enable_post_processing=True)
    open3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True, mesh_show_back_face=True)