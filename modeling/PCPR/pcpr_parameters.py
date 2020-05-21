import torch

class PCPRParameters(torch.nn.Module):
    def __init__(self, feature_dim):
        super(PCPRParameters, self).__init__()
        self.feature_dim = feature_dim
        self.p_parameters = torch.nn.ParameterList()
        self.p_parameters = None
        self.default_features = torch.nn.Parameter(torch.randn(feature_dim, 1).cuda())

         
        #for i in range(self.vertices_num.size(0)):
        #    self.p_parameters.append(torch.nn.Parameter(torch.randn(feature_dim, self.vertices_num[i]).cuda()))
        
        ## just for test, need reimplement here.
        #self.p_parameters = torch.nn.Parameter(torch.randn(feature_dim, self.vertices_num[0]).cuda())



        
    def forward(self):

        return self.p_parameters, self.default_features

    def setPointNum(self, num):
        if self.p_parameters is None:
            self.p_parameters = torch.nn.Parameter(torch.randn(self.feature_dim, num).cuda())




