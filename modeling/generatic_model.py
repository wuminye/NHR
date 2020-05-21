import torch
from .UNet import UNet
from .PCPR import PCPRender
from .PCPR import PCPRParameters
from .Pointnet import get_pn2_model
from utils.instant_norm import instant_norm_1d


class Generatic_Model(torch.nn.Module):


    def __init__(self, cfg, tar_width, tar_height, feature_dim, use_dir, dataset_num=0, use_rgb = False , use_mask=False, use_pointnet = True, use_depth = False, use_pc_norm = False):
        super(Generatic_Model, self).__init__()

        addition = 0
        if cfg.INPUT.RGB_MAP:
            addition = 3


        if not use_pointnet:
            feature_dim = 3

        self.pcpr_parameters = PCPRParameters(feature_dim + addition)
        self.render = PCPRender(cfg,feature_dim,tar_width,tar_height, use_mask=use_mask, use_dir_in_world = (use_dir=='MAPS'), use_depth = use_depth)
        
        self.use_dir = use_dir
        self.use_pointnet = use_pointnet
        self.use_pc_norm = use_pc_norm
        self.use_rgb_map = cfg.INPUT.RGB_MAP

        self.multiple_pointnet = cfg.MODEL.MUL_POINTNET


        input_channels = 0
        self.use_rgb = use_rgb
        if use_rgb:
            input_channels = 3
        if use_dir == 'POINT':
            input_channels = input_channels + 3

        
        if use_pointnet:  
            if self.multiple_pointnet:
                self.pointnets = torch.nn.ModuleList()
                for i in range(dataset_num):
                    self.pointnets.append(get_pn2_model(cfg, input_channels = input_channels, out_dim = feature_dim).cuda())
            else:
                self.pointnet = get_pn2_model(cfg, input_channels = input_channels, out_dim = feature_dim)
                self.pointnet = self.pointnet.cuda()


    def forward(self, in_points, K, T,
           near_far_max_splatting_size, num_points, rgbs, inds=None):
        
        

        


        num_points = num_points.int()

        _,default_features = self.pcpr_parameters()


        batch_size = K.size(0)
        dim_features = default_features.size(0)

        m_point_features = []
        beg = 0

       
        if self.use_pointnet:
            for i in range(batch_size):
                
                cur_points = in_points[beg:beg+num_points[i],:].view(1,num_points[i],3)
                if self.use_dir=='POINT':
                    cam_pose = T[i,0:3,2].view(1,1,3).repeat(1,num_points[i],1)
                    cur_point_dir = cur_points - cam_pose
                    
                    cur_point_dir = cur_point_dir / cur_point_dir.norm(dim=2).view(1,num_points[i],1).repeat(1,1,3)
                    cur_points = torch.cat([cur_points, cur_point_dir], dim=2 )



                if self.use_rgb:
                    cur_rgbs = rgbs[beg:beg+num_points[i],:].view(1,num_points[i],3)
                    cur_points = torch.cat([cur_points, cur_rgbs], dim=2 )

                if self.use_pc_norm:
                    cur_points[:,:,0:3] = instant_norm_1d(cur_points[:,:,0:3])

                if self.multiple_pointnet:
                    m_point_features.append(self.pointnets[inds[i]](cur_points)[0])
                else:
                    m_point_features.append(self.pointnet(cur_points)[0])
            


                beg = beg + num_points[i]

        if self.use_pointnet:
            point_features = torch.cat(m_point_features, dim = 1).requires_grad_()
        else:
            point_features = rgbs.transpose(0,1)

        if self.use_rgb_map:
            point_features = torch.cat([ point_features, rgbs.transpose(0,1)], dim = 0).requires_grad_()

        
   
        res,depth,features,dir_in_world, rgb = self.render(point_features, default_features,
                             in_points,
                             K, T,
                             near_far_max_splatting_size, num_points)

        
        return res,depth, features, dir_in_world, rgb, point_features

        
