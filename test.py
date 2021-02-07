import torch
import numpy as np
from imblearn.over_sampling import RandomOverSampler


class Data(object):
    def __init__(self, x) -> None:
        super().__init__()
        self.x = x
    


ros = RandomOverSampler(random_state=0)


# X = torch.tensor([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15],[16,17,18],[19,20,21],[0.1,0.2,0.3],[0.4,0.5,0.6]])
# y = torch.tensor([0,0,0,0,0,0,0,1,1])

X = [np.array([1,2,3]), np.array([4,5,6]), np.array([7,8,9]), np.array([10,11,12]), np.array([0.1,0.2,0.3])]
y = [0,0,0,0,1]

# X2 = [Data(k) for k in X]


X_resampled, y_resampled = ros.fit_resample(X, y)


print("Done!")