# encoding: utf-8

import torch

def static_collate(batch):

    point_features = []
    in_points = []
    Ks = []
    Ts = []
    inds = []
    num_points = []
    rgbs = []

    imgs = []

    # need to be exposed outside
    #near_far_max_splatting_size = torch.Tensor([ [1000,3600,2] ]).repeat(len(batch),1)
    near_far_max_splatting_size = []
    
    for item in batch:
        #point_features.append(item[2])
        in_points.append(item[1])
        Ks.append(item[4])
        Ts.append(item[3])
        imgs.append(item[0])
        inds.append(item[2])
        near_far_max_splatting_size.append(item[5])
        rgbs.append(item[6])

        num_points.append(item[1].size(0))

    #point_features = torch.cat(point_features, dim=1)
    in_points = torch.cat(in_points, dim=0)
    rgbs = torch.cat(rgbs, dim=0)

    Ks = torch.stack(Ks, dim=0)
    Ts = torch.stack(Ts, dim=0)
    near_far_max_splatting_size = torch.stack(near_far_max_splatting_size, dim=0)

    num_points = torch.Tensor(num_points).view(-1)

    inds = torch.Tensor(inds).int()

    imgs = torch.stack(imgs, dim=0)

    return [0]*len(batch), in_points, Ks, Ts, num_points, near_far_max_splatting_size, inds, imgs, rgbs

