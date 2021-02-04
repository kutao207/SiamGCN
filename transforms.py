import torch

class Center(object):

    def __call__(self, data):
        data.x[:,:3] = data.x[:,:3] - data.x[:,:3].mean(dim=-2, keepdim=True)
        data.x2[:,:3] = data.x2[:,:3] - data.x2[:,:3].mean(dim=-2, keepdim=True)
        return data
    
    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class SamplePoints(object):
    def __init__(self, num: int) -> None:
        super().__init__()
        self.num = num
    def __call__(self, data):
        x1, x2 = data.x, data.x2
        assert x1.size(1) >= 3 and x2.size(1) >= 3

        if self.num < x1.size(0):
            idx = torch.randperm(x1.size(0))[:self.num]
            x1 = x1[idx]
        if self.num < x2.size(0):
            idx = torch.randperm(x2.size(0))[:self.num]
            x2 = x2[idx]
        
        data.x, data.x2 = x1, x2
        return data

class NormalizeScale(object):
    def __init__(self) -> None:
        super().__init__()
        self.center = Center()   

    def __call__(self, data):

        data = self.center(data)
        maxv = max(data.x[:,:3].abs().max(), data.x2[:,:3].abs().max())
        scale = (1 / maxv) * 0.999999 

        data.x[:,:3] *= scale
        data.x2[:,:3] *= scale

        return data

    def __repr__(self) -> str:
        return '{}()'.format(self.__class__.__name__)