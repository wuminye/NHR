import numpy as np


# feats (N, feature_dim)
def feats_pca_projection(feats,pre = None):
    data = np.mat(feats.T)
    if pre is None:
        data_mean = data - data.mean(axis=1)
        data_cov = np.cov(data_mean)
        tzz,tzxl = np.linalg.eig(data_cov)
        xl = tzxl.T[0:3]
        res =  data_mean.T.__mul__(np.mat(xl).T)
    else:
        data_mean = data - pre[0]
        xl = pre[1]
        res =  data_mean.T.__mul__(np.mat(xl).T)
    #res = res/np.std(res) +0.6
    res = np.array(res)

    return res,(data.mean(axis=1),xl )


def feats_map_pca_projection(feats_map, pre = None):
    feats = feats_map.transpose(1,2,0)
    ori_shape = feats.shape
    feats = np.reshape(feats,(-1,feats.shape[2]))

    res, ss = feats_pca_projection(feats,pre)
    res = res.reshape((ori_shape[0],ori_shape[1],3))
    return res,ss
