import numpy as np
import xml.etree.ElementTree as ET
import copy
import cv2
import os
import matplotlib.pyplot as plt
import open3d as o3d
import torch


def calc_euclidean_distance_1D_gpu(x,y):
    xt = x.unsqueeze(dim=1)
    yt = y.unsqueeze(dim=0)
    
    xx = xt**2
    yy = yt**2
    
    S = torch.matmul(xt,yt)

    xx = xx.repeat(1,S.size(1))

    yy = yy.repeat(S.size(0), 1)
    S = xx + yy - 2*S
    return S



dataset_path = 'sport_2/' #dataset path
sfs_path = 'sport2_sfs/'     #sfs output dir
ii = 0                                          # frame id


if not os.path.exists(os.path.join(sfs_path,'holes')):
    os.makedirs(os.path.join(sfs_path,'holes')) 



tmp =  np.load(os.path.join(dataset_path,'pointclouds/frame%d.npy' % (ii+1)))    # original point cloud
origin_pc = o3d.geometry.PointCloud()
origin_pc.points = o3d.utility.Vector3dVector(tmp[:,0:3])
origin_pc.colors = o3d.utility.Vector3dVector(tmp[:,3:6])

sfs_mesh = o3d.io.read_triangle_mesh(os.path.join(sfs_path,'sfs_model/frame%d.ply' % (ii+1)))   # sfs result
sfs_pc = sfs_mesh.sample_points_uniformly(600000)
sfs_pc, ind = sfs_pc.voxel_down_sample_and_trace(voxel_size=0.0033,min_bound=np.min(np.asarray(sfs_pc.points).astype(np.float32), axis = 0), 
                                                    max_bound= np.max(np.asarray(sfs_pc.points).astype(np.float32), axis = 0))

print('origin',len(origin_pc.points))
print('sfs',len(sfs_pc.points))
downpcd, ind = origin_pc.voxel_down_sample_and_trace(voxel_size=0.004,min_bound=np.min(tmp[:,0:3], axis = 0), max_bound= np.max(tmp[:,0:3], axis = 0))
print(len(downpcd.points))

dis = np.asarray(sfs_pc.compute_point_cloud_distance(downpcd))

threshold = (np.max(dis)-np.median(dis)) *0.2 + np.median(dis)
print('threshold', threshold)


points_ori_np = np.asarray(downpcd.points).astype(np.float32)
colors_ori_np = np.asarray(downpcd.colors).astype(np.float32)
points_sfs_np = np.asarray(sfs_pc.points).astype(np.float32)
colors_sfs_np = np.zeros_like(points_sfs_np)

points_ori_gpu = torch.tensor(points_ori_np).cuda()
points_sfs_gpu = torch.tensor(points_sfs_np).cuda()


cnt = 0
step = 200
nums = []
inds = []
while cnt < points_sfs_np.shape[0]:

    diffs = []
    for i in range(3):
        diff = calc_euclidean_distance_1D_gpu( points_sfs_gpu[cnt:cnt+step,i].clone(),points_ori_gpu[:,i].clone())
        diffs.append(diff)

    for i in range(1,3):
        diffs[0] = diffs[0]+diffs[i]

    res = torch.sqrt(diffs[0])
    
    

    mask = res<threshold
    inds.append(torch.argmin(res,dim=1))
    nums.append(torch.sum(mask,dim=1).cpu())

    cnt =cnt + step
    if cnt%100000 ==0:
        print(cnt)

nums = torch.cat(nums)
inds = torch.cat(inds)

colors_sfs_np = colors_ori_np[inds.cpu().numpy(),:]


data = nums.numpy().tolist()
p = nums.clone().tolist()
p.sort()
binwidth = torch.max(nums)/15

t = plt.hist(data, bins=range(min(data), max(data) + binwidth, binwidth))
plt.show()

tau_2 = p[int(t[0][0])]
print('tau_2', tau_2)


res_pc = o3d.geometry.PointCloud()
res_pc.points = o3d.utility.Vector3dVector(points_sfs_np[nums<tau_2,:])
res_pc.colors = o3d.utility.Vector3dVector(colors_sfs_np[nums<tau_2,:])

res_pc_d, ind = res_pc.voxel_down_sample_and_trace(voxel_size=0.0033,min_bound=np.min(points_sfs_np[nums<tau_2,:], axis = 0), max_bound= np.max(points_sfs_np[nums<tau_2,:], axis = 0))

res_p = np.asarray(res_pc_d.points).astype(np.float32)
res_c = np.asarray(res_pc_d.colors).astype(np.float32)

np.savetxt(os.path.join(sfs_path,'holes/frame%d.txt' %(ii+1)),np.concatenate([res_p, res_c],axis=1))
np.save(os.path.join(sfs_path,'holes/frame%d.npy' %(ii+1)),np.concatenate([res_p, res_c],axis=1))
print(res_p.shape)
print('finished. %d'%i)
print('---------------------------')