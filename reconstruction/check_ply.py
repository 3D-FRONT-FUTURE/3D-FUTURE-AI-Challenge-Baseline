import open3d as o3d
import os,pdb

ply_dir = 'reconstruction_results'
ply_names = os.listdir(ply_dir)
for ply_name in ply_names:
    ply_file = os.path.join(ply_dir, ply_name)
    pcd = o3d.io.read_point_cloud(ply_file)
    if len(pcd.points) > 2048 or len(pcd.points)< 1000:
        print(len(pcd.points))
        print('wrong: ', ply_name)
